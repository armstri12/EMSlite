"""Shared helpers extracted from energy_dashboard.py / visualize_meter_data.py."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def amps_to_kw(
    amps: pd.Series,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    calibration_factor: float = 1.0,
) -> pd.Series:
    """Convert amperage readings to kilowatts (three-phase).

    ``power_factor`` turns apparent power (kVA) into real power (kW); leaving it
    at 1.0 returns apparent power and over-reports real energy by ``1 / PF``.
    ``calibration_factor`` is an empirical scalar (default 1.0) used to reconcile
    computed energy against metered/utility-bill kWh — see docs/calculations.md.
    """
    return amps * (line_voltage * 3**0.5 * power_factor * calibration_factor) / 1000.0


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


def excluded_columns(cfg: dict) -> set[str]:
    """Columns that must NOT be summed into the facility total.

    Covers pre-computed group columns (``combo_columns``) and any aggregate /
    main-feed columns (``aggregate_columns``) that already include other panels —
    summing those on top of their branch panels double-counts and over-reports.
    """
    return set(cfg.get("combo_columns", {}).keys()) | set(cfg.get("aggregate_columns", []))


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


def get_device_voltage_map() -> dict[str, float]:
    """Return {device_id: voltage} for devices with a voltage override."""
    from .database import get_session
    from .models import Device

    session = get_session()
    try:
        devices = session.query(Device).filter(Device.voltage.isnot(None)).all()
        return {d.id: d.voltage for d in devices}
    finally:
        session.close()


def get_device_meter_map() -> dict[str, str]:
    """Return {device_id: meter_name} for enabled devices assigned to a meter."""
    from .database import get_session
    from .models import Device

    session = get_session()
    try:
        devices = (
            session.query(Device)
            .filter(Device.enabled.is_(True), Device.meter_name.isnot(None))
            .all()
        )
        return {d.id: d.meter_name for d in devices}
    finally:
        session.close()


def meter_factors(cfg: dict, meter_name: str | None) -> tuple[float, float]:
    """Resolve (power_factor, calibration_factor) for one meter.

    Per-meter values in ``cfg['meter_overrides'][meter_name]`` win over the
    global ``power_factor`` / ``calibration_factor``.
    """
    pf = float(cfg.get("power_factor", 1.0))
    cal = float(cfg.get("calibration_factor", 1.0))
    ov = (cfg.get("meter_overrides", {}) or {}).get(meter_name or "", {})
    if "power_factor" in ov and ov["power_factor"] is not None:
        pf = float(ov["power_factor"])
    if "calibration_factor" in ov and ov["calibration_factor"] is not None:
        cal = float(ov["calibration_factor"])
    return pf, cal


def build_meter_factor_maps(
    cfg: dict, device_meter_map: dict[str, str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Build {device_id: power_factor} and {device_id: calibration_factor} maps.

    Only devices whose meter has an override entry get an entry; callers fall
    back to the global scalar for everything else.
    """
    overrides = cfg.get("meter_overrides", {}) or {}
    pf_map: dict[str, float] = {}
    cal_map: dict[str, float] = {}
    for dev_id, meter in device_meter_map.items():
        ov = overrides.get(meter)
        if not ov:
            continue
        if ov.get("power_factor") is not None:
            pf_map[dev_id] = float(ov["power_factor"])
        if ov.get("calibration_factor") is not None:
            cal_map[dev_id] = float(ov["calibration_factor"])
    return pf_map, cal_map


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
