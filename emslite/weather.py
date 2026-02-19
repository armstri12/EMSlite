"""Weather data fetching and caching via NOAA NCEI Access Data Service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from .database import get_session
from .models import WeatherCache

logger = logging.getLogger(__name__)

NCEI_BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def fetch_weather_from_api(
    station_id: str,
    start_date: str,  # "YYYY-MM-DD"
    end_date: str,  # "YYYY-MM-DD"
    unit: str = "celsius",
) -> list[dict[str, Any]]:
    """Fetch hourly weather data from NOAA NCEI Local Climatological Data."""
    params = {
        "dataset": "local-climatological-data",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "HourlyDryBulbTemperature,HourlyRelativeHumidity",
        "format": "json",
    }
    if unit == "celsius":
        params["units"] = "metric"

    resp = requests.get(NCEI_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        logger.warning("Unexpected NCEI response format: %s", type(data))
        return []

    results = []
    for rec in data:
        ts_str = rec.get("DATE")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        temp_raw = rec.get("HourlyDryBulbTemperature")
        hum_raw = rec.get("HourlyRelativeHumidity")

        temp = _parse_float(temp_raw)
        hum = _parse_float(hum_raw)

        if temp is None and hum is None:
            continue

        results.append({
            "timestamp": ts,
            "temperature_c": temp,
            "humidity_pct": hum,
        })

    return results


def _parse_float(val: Any) -> float | None:
    """Safely parse a value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(str(val).rstrip("s*"))  # NCEI sometimes appends 's' for suspect
    except (ValueError, TypeError):
        return None


def cache_weather_data(
    station_id: str,
    records: list[dict[str, Any]],
) -> int:
    """Upsert weather records into the cache table. Returns count stored."""
    session = get_session()
    try:
        count = 0
        for rec in records:
            existing = (
                session.query(WeatherCache)
                .filter(
                    WeatherCache.station_id == station_id,
                    WeatherCache.timestamp == rec["timestamp"],
                )
                .first()
            )
            if existing:
                existing.temperature_c = rec["temperature_c"]
                existing.humidity_pct = rec["humidity_pct"]
            else:
                session.add(
                    WeatherCache(
                        station_id=station_id,
                        timestamp=rec["timestamp"],
                        temperature_c=rec["temperature_c"],
                        humidity_pct=rec["humidity_pct"],
                    )
                )
            count += 1
        session.commit()
        return count
    finally:
        session.close()


def get_cached_weather(
    station_id: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Return cached weather records for the given range."""
    session = get_session()
    try:
        rows = (
            session.query(WeatherCache)
            .filter(
                WeatherCache.station_id == station_id,
                WeatherCache.timestamp >= start,
                WeatherCache.timestamp <= end,
            )
            .order_by(WeatherCache.timestamp)
            .all()
        )
        return [r.to_dict() for r in rows]
    finally:
        session.close()


def get_weather_for_range(
    station_id: str,
    start_date: str,  # "YYYY-MM-DD"
    end_date: str,  # "YYYY-MM-DD"
    unit: str = "celsius",
) -> list[dict[str, Any]]:
    """Get weather data for a date range. Cache-first with API fallback."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    # Check cache completeness
    cached = get_cached_weather(station_id, start_dt, end_dt)
    expected_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1

    if len(cached) >= expected_hours * 0.8:
        return cached

    # Fetch from NOAA and cache
    try:
        records = fetch_weather_from_api(station_id, start_date, end_date, unit)
        if records:
            cache_weather_data(station_id, records)
            logger.info(
                "Cached %d weather records for station %s (%s to %s)",
                len(records),
                station_id,
                start_date,
                end_date,
            )
    except Exception as e:
        logger.warning("Failed to fetch weather data from NOAA: %s", e)
        if cached:
            return cached
        return []

    return get_cached_weather(station_id, start_dt, end_dt)
