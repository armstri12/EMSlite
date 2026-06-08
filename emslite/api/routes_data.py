"""Data and metrics API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from ..core import (
    amps_to_kw,
    build_meter_factor_maps,
    excluded_columns,
    get_device_meter_map,
    get_device_voltage_map,
    load_csv,
    meter_columns,
    parse_window_to_hours,
)
from ..metrics import compute_department_breakdown, compute_kpi, compute_panel_rankings, integrate_kwh

router = APIRouter(tags=["data"])


def _get_master_path() -> Path:
    from .app import get_app_config, get_project_root

    cfg = get_app_config()
    root = get_project_root()
    data_dir = root / cfg.get("data_dir", "data")
    return data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")


def _get_config() -> dict:
    from .app import get_app_config
    return get_app_config()


@router.get("/data")
def get_data(
    start: str | None = Query(None, description="Start timestamp ISO format"),
    end: str | None = Query(None, description="End timestamp ISO format"),
    panels: str | None = Query(None, description="Comma-separated panel IDs"),
    department: str | None = Query(None, description="Filter by department ID"),
) -> dict[str, Any]:
    """Return time-series data for the dashboard."""
    master = _get_master_path()
    if not master.exists():
        return {"timestamps": [], "total_kw": [], "panel_series": {}, "panel_names": []}

    cfg = _get_config()
    df = load_csv(master)
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    calibration_factor = float(cfg.get("calibration_factor", 1.0))
    price_per_kwh = float(cfg.get("price_per_kwh", 0.25))
    carbon_kg_per_kwh = float(cfg.get("carbon_kg_per_kwh", 0.4))
    rolling_hours = parse_window_to_hours(cfg.get("rolling_window", "1h"))
    voltage_map = get_device_voltage_map()
    pf_map, cal_map = build_meter_factor_maps(cfg, get_device_meter_map())

    # Date filtering
    if start:
        import pandas as pd
        start_dt = pd.to_datetime(start, utc=True)
        df = df[df["Timestamp"] >= start_dt]
    if end:
        import pandas as pd
        end_dt = pd.to_datetime(end, utc=True)
        df = df[df["Timestamp"] <= end_dt]

    all_panels = meter_columns(df.columns, exclude=excluded_columns(cfg))

    # Filter by specific panels
    if panels:
        selected = [p.strip() for p in panels.split(",")]
        all_panels = [p for p in all_panels if p in selected]

    # Filter by department
    if department:
        dept_panels = _get_department_panels(department)
        if dept_panels is not None:
            all_panels = [p for p in all_panels if p in dept_panels]

    timestamps = df["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    # Build panel series (convert amps to kW)
    panel_series = {}
    for col in all_panels:
        if col in df.columns:
            v = voltage_map.get(col, line_voltage)
            panel_series[col] = amps_to_kw(
                df[col].fillna(0), v,
                pf_map.get(col, power_factor), cal_map.get(col, calibration_factor),
            ).fillna(0).round(3).tolist()

    # Total kW across selected panels
    import pandas as pd
    total_kw_series = pd.Series(0.0, index=df.index)
    for col in all_panels:
        if col in df.columns:
            v = voltage_map.get(col, line_voltage)
            total_kw_series += amps_to_kw(
                df[col].fillna(0), v,
                pf_map.get(col, power_factor), cal_map.get(col, calibration_factor),
            ).fillna(0)

    # Build group (combo_columns) series — these are pre-computed kW columns
    combo_cols = cfg.get("combo_columns", {})
    group_series = {}
    group_names = []
    for col_name in combo_cols:
        if col_name in df.columns:
            group_names.append(col_name)
            group_series[col_name] = df[col_name].fillna(0).round(3).tolist()

    return {
        "timestamps": timestamps,
        "total_kw": total_kw_series.round(3).tolist(),
        "panel_series": panel_series,
        "panel_names": all_panels,
        "group_series": group_series,
        "group_names": group_names,
        "rolling_hours": rolling_hours,
        "price_per_kwh": price_per_kwh,
        "carbon_kg_per_kwh": carbon_kg_per_kwh,
        "utility_meters": cfg.get("utility_meters", []),
    }


@router.get("/metrics")
def get_metrics(
    start: str | None = Query(None),
    end: str | None = Query(None),
    department: str | None = Query(None),
) -> dict[str, Any]:
    """Return computed KPI metrics."""
    master = _get_master_path()
    if not master.exists():
        return {"kpi": {}, "rankings": [], "departments": []}

    cfg = _get_config()
    df = load_csv(master)
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    calibration_factor = float(cfg.get("calibration_factor", 1.0))
    price_per_kwh = float(cfg.get("price_per_kwh", 0.25))
    carbon_kg_per_kwh = float(cfg.get("carbon_kg_per_kwh", 0.4))
    voltage_map = get_device_voltage_map()
    pf_map, cal_map = build_meter_factor_maps(cfg, get_device_meter_map())

    # Date filtering
    if start:
        import pandas as pd
        df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        import pandas as pd
        df = df[df["Timestamp"] <= pd.to_datetime(end, utc=True)]

    all_panels = meter_columns(df.columns, exclude=excluded_columns(cfg))

    # Department filter
    if department:
        dept_panels = _get_department_panels(department)
        if dept_panels is not None:
            all_panels = [p for p in all_panels if p in dept_panels]

    kpi = compute_kpi(df, line_voltage, power_factor, price_per_kwh, carbon_kg_per_kwh, all_panels, voltage_map=voltage_map, calibration_factor=calibration_factor, pf_map=pf_map, calibration_map=cal_map)
    rankings = compute_panel_rankings(df, line_voltage, power_factor, all_panels, top_n=10, voltage_map=voltage_map, calibration_factor=calibration_factor, pf_map=pf_map, calibration_map=cal_map)

    # Department breakdown.
    # When a department filter is active, return breakdown only for that department
    # so the response is consistent with the filtered KPI and rankings.
    # Without a filter, return all departments.
    dept_map = _get_all_department_panels()
    if department and department in dept_map:
        dept_map = {department: dept_map[department]}
    elif department:
        # Filter by display name as fallback (dept param may be display name)
        filtered = {k: v for k, v in dept_map.items() if k == department}
        if filtered:
            dept_map = filtered
    dept_breakdown = compute_department_breakdown(
        df,
        dept_map,
        line_voltage,
        power_factor,
        price_per_kwh,
        carbon_kg_per_kwh,
        voltage_map=voltage_map,
        calibration_factor=calibration_factor,
        pf_map=pf_map,
        calibration_map=cal_map,
    )

    # Enrich rankings with display names
    device_names = _get_device_display_names()
    for r in rankings:
        r["display_name"] = device_names.get(r["panel_id"], r["panel_id"])

    return {
        "kpi": kpi,
        "rankings": rankings,
        "departments": dept_breakdown,
        "carbon_kg_per_kwh": carbon_kg_per_kwh,
    }


@router.get("/meter-coverage")
def get_meter_coverage(
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> dict[str, Any]:
    """Reconcile the facility total against per-meter (billed) coverage.

    Computes each panel's kWh with the SAME factors the dashboard uses
    (per-device voltage + per-meter power-factor / calibration) and groups them
    by their device's meter assignment. The facility total a meter-less bill can
    never explain is the sum of the ``unassigned`` panels — the usual reason a
    comparison-view total exceeds the combined meter bills.
    """
    master = _get_master_path()
    if not master.exists():
        return {"meters": [], "unassigned": [], "facility_total_kwh": 0.0}

    cfg = _get_config()
    df = load_csv(master)
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    calibration_factor = float(cfg.get("calibration_factor", 1.0))
    voltage_map = get_device_voltage_map()
    device_meter_map = get_device_meter_map()
    pf_map, cal_map = build_meter_factor_maps(cfg, device_meter_map)

    if start:
        import pandas as pd
        df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        import pandas as pd
        end_ts = end + "T23:59:59" if len(end) == 10 else end
        df = df[df["Timestamp"] <= pd.to_datetime(end_ts, utc=True)]

    if df.empty:
        return {"meters": [], "unassigned": [], "facility_total_kwh": 0.0}

    hours = df["Timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0
    all_panels = meter_columns(df.columns, exclude=excluded_columns(cfg))

    # Per-panel kWh using the dashboard's factor resolution.
    panels: list[dict[str, Any]] = []
    for col in all_panels:
        if col not in df.columns:
            continue
        v = voltage_map.get(col, line_voltage)
        kw = amps_to_kw(
            df[col].fillna(0), v, pf_map.get(col, power_factor), cal_map.get(col, calibration_factor)
        ).fillna(0)
        panels.append({
            "panel_id": col,
            "meter_name": device_meter_map.get(col),
            "total_kwh": round(integrate_kwh(kw, hours), 2),
            "voltage": v,
            "calibration_factor": cal_map.get(col, calibration_factor),
            "power_factor": pf_map.get(col, power_factor),
        })

    # Group by meter assignment; meter-less panels are the reconciliation gap.
    by_meter: dict[str, list[dict[str, Any]]] = {}
    unassigned: list[dict[str, Any]] = []
    for p in panels:
        if p["meter_name"]:
            by_meter.setdefault(p["meter_name"], []).append(p)
        else:
            unassigned.append(p)

    meters = [
        {
            "meter_name": name,
            "panel_count": len(rows),
            "total_kwh": round(sum(r["total_kwh"] for r in rows), 2),
            "panels": sorted(rows, key=lambda r: r["total_kwh"], reverse=True),
        }
        for name, rows in sorted(by_meter.items())
    ]
    unassigned.sort(key=lambda r: r["total_kwh"], reverse=True)
    facility_total = round(sum(p["total_kwh"] for p in panels), 2)

    # Panels the user explicitly excluded — compute their kWh too so they're
    # visible (and re-includable) even though they're out of the facility total.
    excluded_panel_ids = cfg.get("excluded_panels", []) or []
    excluded_panels = []
    for col in excluded_panel_ids:
        if col not in df.columns:
            continue
        v = voltage_map.get(col, line_voltage)
        kw = amps_to_kw(
            df[col].fillna(0), v, pf_map.get(col, power_factor), cal_map.get(col, calibration_factor)
        ).fillna(0)
        excluded_panels.append({
            "panel_id": col,
            "meter_name": device_meter_map.get(col),
            "total_kwh": round(integrate_kwh(kw, hours), 2),
        })
    excluded_panels.sort(key=lambda r: r["total_kwh"], reverse=True)

    return {
        "facility_total_kwh": facility_total,
        "assigned_total_kwh": round(sum(m["total_kwh"] for m in meters), 2),
        "unassigned_total_kwh": round(sum(p["total_kwh"] for p in unassigned), 2),
        "meters": meters,
        "unassigned": unassigned,
        # User-excluded panels (with kWh) so they can be reviewed / re-included.
        "excluded_panels": excluded_panels,
        # All columns kept out of the facility total (combo/aggregate/excluded),
        # surfaced so a hidden double-counting feed is easy to spot.
        "excluded_columns": sorted(excluded_columns(cfg)),
        "range": {
            "start": df["Timestamp"].min().isoformat(),
            "end": df["Timestamp"].max().isoformat(),
        },
    }


def _get_department_panels(department_id: str) -> list[str] | None:
    """Get panel IDs belonging to a department."""
    from ..database import get_session
    from ..models import Device

    session = get_session()
    try:
        devices = session.query(Device).filter(
            Device.department_id == department_id,
            Device.enabled.is_(True),
        ).all()
        return [d.id for d in devices] if devices else []
    finally:
        session.close()


def _get_all_department_panels() -> dict[str, list[str]]:
    """Get all department -> panel mappings."""
    from ..database import get_session
    from ..models import Department, Device

    session = get_session()
    try:
        departments = session.query(Department).all()
        result = {}
        for dept in departments:
            devices = session.query(Device).filter(
                Device.department_id == dept.id,
                Device.enabled.is_(True),
            ).all()
            result[dept.display_name] = [d.id for d in devices]

        # Also include unassigned
        unassigned = session.query(Device).filter(
            Device.department_id.is_(None),
            Device.enabled.is_(True),
        ).all()
        if unassigned:
            result["Unassigned"] = [d.id for d in unassigned]

        return result
    finally:
        session.close()


def _get_device_display_names() -> dict[str, str]:
    """Get mapping of device ID -> display name."""
    from ..database import get_session
    from ..models import Device

    session = get_session()
    try:
        devices = session.query(Device).all()
        return {d.id: d.display_name for d in devices}
    finally:
        session.close()
