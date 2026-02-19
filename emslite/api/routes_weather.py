"""Weather data API endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(tags=["weather"])


@router.get("/weather")
def get_weather(
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Return cached/fetched weather data for the configured station."""
    from .app import get_app_config
    from ..weather import get_weather_for_range

    cfg = get_app_config()
    weather_cfg = cfg.get("weather", {})

    if not weather_cfg.get("enabled", False):
        return {"enabled": False, "timestamps": [], "temperature_c": [], "humidity_pct": []}

    station_id = weather_cfg.get("station_id")
    if not station_id:
        return {
            "enabled": False,
            "timestamps": [],
            "temperature_c": [],
            "humidity_pct": [],
            "error": "Weather station not configured. Set weather.station_id in config.",
        }

    unit = weather_cfg.get("unit", "celsius")
    records = get_weather_for_range(station_id, start, end, unit)

    return {
        "enabled": True,
        "timestamps": [r["timestamp"] for r in records],
        "temperature_c": [r["temperature_c"] for r in records],
        "humidity_pct": [r["humidity_pct"] for r in records],
        "station_id": station_id,
        "unit": unit,
    }
