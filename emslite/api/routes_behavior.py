"""Behavior analysis API endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..behavior import analyze_behavior, compute_phantom_rankings
from ..core import load_csv, meter_columns
from .routes_data import _get_config, _get_master_path

router = APIRouter(tags=["behavior"])


def _get_all_device_display_names() -> dict[str, str]:
    """Get mapping of device ID -> display name for all devices."""
    from ..database import get_session
    from ..models import Device

    session = get_session()
    try:
        devices = session.query(Device).all()
        return {d.id: d.display_name for d in devices}
    finally:
        session.close()


def _get_device_display_name(panel_id: str) -> str | None:
    """Look up device display name from DB."""
    from ..database import get_session
    from ..models import Device

    session = get_session()
    try:
        device = session.get(Device, panel_id)
        return device.display_name if device else None
    finally:
        session.close()


@router.get("/behavior/rankings")
def get_behavior_rankings(
    start: str | None = Query(None, description="Start timestamp ISO format"),
    end: str | None = Query(None, description="End timestamp ISO format"),
) -> dict[str, Any]:
    """Rank all panels by phantom draw waste."""
    master = _get_master_path()
    if not master.exists():
        raise HTTPException(status_code=404, detail="No data available.")

    cfg = _get_config()
    df = load_csv(master)

    # Date filtering
    if start:
        df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["Timestamp"] <= pd.to_datetime(end, utc=True)]

    if df.empty:
        raise HTTPException(
            status_code=404, detail="No data for selected date range."
        )

    all_panels = meter_columns(
        df.columns, exclude=set(cfg.get("combo_columns", {}).keys())
    )
    display_names = _get_all_device_display_names()

    return compute_phantom_rankings(
        df=df,
        panel_cols=all_panels,
        line_voltage=float(cfg.get("line_voltage", 480.0)),
        power_factor=float(cfg.get("power_factor", 1.0)),
        price_per_kwh=float(cfg.get("price_per_kwh", 0.25)),
        carbon_kg_per_kwh=float(cfg.get("carbon_kg_per_kwh", 0.4)),
        display_names=display_names,
    )


@router.get("/behavior")
def get_behavior(
    panel: str = Query(..., description="Panel/device column ID to analyze"),
    start: str | None = Query(None, description="Start timestamp ISO format"),
    end: str | None = Query(None, description="End timestamp ISO format"),
) -> dict[str, Any]:
    """Return behavior analysis for a single panel."""
    master = _get_master_path()
    if not master.exists():
        raise HTTPException(status_code=404, detail="No data available.")

    cfg = _get_config()
    df = load_csv(master)

    # Validate panel exists
    available = meter_columns(
        df.columns, exclude=set(cfg.get("combo_columns", {}).keys())
    )
    if panel not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Panel '{panel}' not found. Available: {available[:10]}",
        )

    # Date filtering
    if start:
        df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["Timestamp"] <= pd.to_datetime(end, utc=True)]

    if df.empty:
        raise HTTPException(
            status_code=404, detail="No data for selected date range."
        )

    display_name = _get_device_display_name(panel) or panel

    result = analyze_behavior(
        df=df,
        panel_col=panel,
        line_voltage=float(cfg.get("line_voltage", 480.0)),
        power_factor=float(cfg.get("power_factor", 1.0)),
        price_per_kwh=float(cfg.get("price_per_kwh", 0.25)),
        carbon_kg_per_kwh=float(cfg.get("carbon_kg_per_kwh", 0.4)),
        panel_display_name=display_name,
    )

    return result
