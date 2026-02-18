"""Metric calculations for energy data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .core import amps_to_kw, meter_columns, resolve_columns


def compute_kpi(
    df: pd.DataFrame,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
    panel_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compute high-level KPI metrics from a wide-format dataframe.

    Returns dict with: total_kwh, total_cost, peak_kw, avg_kw, load_factor,
    device_count, latest_timestamp, date_range_days.
    """
    if panel_cols is None:
        panel_cols = meter_columns(df.columns)

    # Convert all panels from amps to kW and sum
    kw_df = pd.DataFrame()
    for col in panel_cols:
        if col in df.columns:
            kw_df[col] = amps_to_kw(df[col].fillna(0), line_voltage, power_factor)

    if kw_df.empty:
        return _empty_kpi()

    total_kw = kw_df.sum(axis=1)

    ts = df["Timestamp"]
    dt_hours = ts.diff().dt.total_seconds().fillna(0) / 3600.0
    total_kwh = (total_kw * dt_hours).sum()
    peak_kw = float(total_kw.max())
    avg_kw = float(total_kw.mean())
    load_factor = (avg_kw / peak_kw * 100.0) if peak_kw > 0 else 0.0
    date_range = (ts.max() - ts.min()).total_seconds() / 86400.0

    return {
        "total_kwh": round(total_kwh, 2),
        "total_cost": round(total_kwh * price_per_kwh, 2),
        "total_carbon_kg": round(total_kwh * carbon_kg_per_kwh, 2),
        "total_carbon_tonnes": round(total_kwh * carbon_kg_per_kwh / 1000.0, 3),
        "peak_kw": round(peak_kw, 2),
        "avg_kw": round(avg_kw, 2),
        "load_factor": round(load_factor, 1),
        "device_count": len(panel_cols),
        "latest_timestamp": ts.max().isoformat() if not ts.empty else None,
        "date_range_days": round(date_range, 1),
    }


def compute_panel_rankings(
    df: pd.DataFrame,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    panel_cols: list[str] | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Rank panels by total kWh consumption."""
    if panel_cols is None:
        panel_cols = meter_columns(df.columns)

    ts = df["Timestamp"]
    dt_hours = ts.diff().dt.total_seconds().fillna(0) / 3600.0

    rankings = []
    for col in panel_cols:
        if col not in df.columns:
            continue
        kw = amps_to_kw(df[col].fillna(0), line_voltage, power_factor)
        kwh = float((kw * dt_hours).sum())
        peak = float(kw.max())
        rankings.append({
            "panel_id": col,
            "total_kwh": round(kwh, 2),
            "peak_kw": round(peak, 2),
        })

    rankings.sort(key=lambda x: x["total_kwh"], reverse=True)
    return rankings[:top_n]


def compute_department_breakdown(
    df: pd.DataFrame,
    department_panels: dict[str, list[str]],
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
) -> list[dict[str, Any]]:
    """Compute energy/cost metrics grouped by department."""
    ts = df["Timestamp"]
    dt_hours = ts.diff().dt.total_seconds().fillna(0) / 3600.0

    results = []
    for dept_name, panels in department_panels.items():
        valid_panels = [p for p in panels if p in df.columns]
        if not valid_panels:
            results.append({
                "department": dept_name,
                "total_kwh": 0,
                "total_cost": 0,
                "total_carbon_kg": 0,
                "total_carbon_tonnes": 0,
                "peak_kw": 0,
                "device_count": 0,
            })
            continue

        kw_df = pd.DataFrame()
        for col in valid_panels:
            kw_df[col] = amps_to_kw(df[col].fillna(0), line_voltage, power_factor)

        total_kw = kw_df.sum(axis=1)
        total_kwh = float((total_kw * dt_hours).sum())

        results.append({
            "department": dept_name,
            "total_kwh": round(total_kwh, 2),
            "total_cost": round(total_kwh * price_per_kwh, 2),
            "total_carbon_kg": round(total_kwh * carbon_kg_per_kwh, 2),
            "total_carbon_tonnes": round(total_kwh * carbon_kg_per_kwh / 1000.0, 3),
            "peak_kw": round(float(total_kw.max()), 2),
            "device_count": len(valid_panels),
        })

    results.sort(key=lambda x: x["total_kwh"], reverse=True)
    return results


def _empty_kpi() -> dict[str, Any]:
    return {
        "total_kwh": 0,
        "total_cost": 0,
        "total_carbon_kg": 0,
        "total_carbon_tonnes": 0,
        "peak_kw": 0,
        "avg_kw": 0,
        "load_factor": 0,
        "device_count": 0,
        "latest_timestamp": None,
        "date_range_days": 0,
    }
