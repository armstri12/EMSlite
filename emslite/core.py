"""Shared helpers extracted from energy_dashboard.py / visualize_meter_data.py."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def amps_to_kw(
    amps: pd.Series,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
) -> pd.Series:
    """Convert amperage readings to kilowatts (three-phase)."""
    return amps * (line_voltage * 3**0.5 * power_factor) / 1000.0


def resolve_columns(
    available: Iterable[str],
    requested: list[str] | None,
    label: str = "",
) -> list[str]:
    """Return the intersection of *requested* columns that exist in *available*."""
    available_set = set(available)
    if requested is None:
        return [c for c in available if c in available_set]
    missing = [c for c in requested if c not in available_set]
    if missing and label:
        print(f"Warning: {label} missing columns skipped: {missing}")
    return [c for c in requested if c in available_set]


def meter_columns(
    columns: Iterable[str],
    exclude: set[str] | None = None,
) -> list[str]:
    """Return column names that represent physical meter/panel readings."""
    always_skip = {"Timestamp", "Total_Amps", "Total_kW"}
    skip = always_skip | (exclude or set())
    return [c for c in columns if c not in skip]


def load_csv(path: str | object) -> pd.DataFrame:
    """Load a wide-format CSV and parse the Timestamp column."""
    from pathlib import Path

    p = Path(path)
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" not in df.columns:
        raise ValueError("Input CSV must include a 'Timestamp' column.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    for col in df.columns:
        if col != "Timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_rolling_window(window: str) -> str:
    return window.replace("H", "h")


def parse_window_to_hours(window: str) -> float:
    normalized = normalize_rolling_window(window).strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-zA-Z]+)", normalized)
    if not match:
        raise ValueError(f"Unsupported rolling window format: {window}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return value
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return value / 60.0
    if unit in {"d", "day", "days"}:
        return value * 24.0
    raise ValueError(f"Unsupported rolling window unit: {window}")
