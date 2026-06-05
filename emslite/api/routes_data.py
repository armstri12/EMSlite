"""Data and metrics API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from ..core import amps_to_kw, excluded_columns, get_device_voltage_map, load_csv, meter_columns, parse_window_to_hours
from ..metrics import compute_department_breakdown, compute_kpi, compute_panel_rankings

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
                df[col].fillna(0), v, power_factor, calibration_factor
            ).fillna(0).round(3).tolist()

    # Total kW across selected panels
    import pandas as pd
    total_kw_series = pd.Series(0.0, index=df.index)
    for col in all_panels:
        if col in df.columns:
            v = voltage_map.get(col, line_voltage)
            total_kw_series += amps_to_kw(df[col].fillna(0), v, power_factor, calibration_factor).fillna(0)

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

    kpi = compute_kpi(df, line_voltage, power_factor, price_per_kwh, carbon_kg_per_kwh, all_panels, voltage_map=voltage_map, calibration_factor=calibration_factor)
    rankings = compute_panel_rankings(df, line_voltage, power_factor, all_panels, top_n=10, voltage_map=voltage_map, calibration_factor=calibration_factor)

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
