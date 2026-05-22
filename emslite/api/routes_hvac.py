"""HVAC / weather-correlation analysis API.

Lets the user upload daily temperature observations (avg/min/max) and
correlates them against per-panel energy consumption to surface HVAC
efficiency issues:

  - Energy signature: daily kWh vs. outdoor temperature with a fitted model.
  - Degree-day model: heating/cooling sensitivity (kWh per HDD / CDD).
  - Baseload split: constant load vs. weather-driven HVAC load.
  - Anomaly flags: days whose energy deviates from the temperature-predicted
    baseline, which usually points at HVAC faults or control problems.

Temperatures are stored in Celsius; every endpoint converts to/from a
display unit (``celsius`` or ``fahrenheit``) at the boundary.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core import amps_to_kw, get_device_voltage_map, load_csv, meter_columns
from ..database import get_session
from ..models import DailyTemperature

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hvac"])

MIN_DAYS_FOR_MODEL = 5
MIN_DAYS_FOR_ANOMALY = 8


# ─────────────────────────── helpers ───────────────────────────


def _get_master_path() -> Path:
    from .app import get_app_config, get_project_root

    cfg = get_app_config()
    root = get_project_root()
    data_dir = root / cfg.get("data_dir", "data")
    return data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")


def _get_config() -> dict:
    from .app import get_app_config

    return get_app_config()


def _parse_date(value: str | _date) -> _date:
    if isinstance(value, _date):
        return value
    s = str(value).strip()
    try:
        return _date.fromisoformat(s[:10])
    except ValueError:
        return pd.to_datetime(s).date()


def _clean_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (ValueError, TypeError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 3)


def _c_to_unit(c: float | None, unit: str) -> float | None:
    if c is None:
        return None
    return c * 9.0 / 5.0 + 32.0 if unit == "fahrenheit" else c


def _unit_to_c(v: float | None, unit: str) -> float | None:
    if v is None:
        return None
    return (v - 32.0) * 5.0 / 9.0 if unit == "fahrenheit" else v


def _resolve_unit(unit: str | None) -> str:
    if unit in ("fahrenheit", "celsius"):
        return unit
    cfg_unit = (_get_config().get("weather", {}) or {}).get("unit", "celsius")
    return "fahrenheit" if cfg_unit == "fahrenheit" else "celsius"


def _default_balance_point(unit: str) -> float:
    return 65.0 if unit == "fahrenheit" else 18.3


def _daily_kwh(
    start: str | None, end: str | None, panels: list[str]
) -> tuple[dict[str, float], list[str]]:
    """Return ({date_str: total kWh for selected panels}, resolved_panel_list)."""
    master = _get_master_path()
    if not master.exists():
        return {}, []

    cfg = _get_config()
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    voltage_map = get_device_voltage_map()

    df = load_csv(master)
    if start:
        df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["Timestamp"] <= pd.to_datetime(end, utc=True) + pd.Timedelta(days=1)]

    all_panels = meter_columns(df.columns, exclude=set(cfg.get("combo_columns", {}).keys()))
    resolved = [p for p in panels if p in all_panels] if panels else all_panels[:]
    if df.empty or not resolved:
        return {}, resolved

    df = df.set_index("Timestamp")
    frame = pd.DataFrame(index=df.index)
    for col in resolved:
        if col not in df.columns:
            continue
        v = voltage_map.get(col, line_voltage)
        frame[col] = amps_to_kw(df[col].fillna(0), v, power_factor).fillna(0)
    if frame.empty or not len(frame.columns):
        return {}, resolved

    # Mean kW over each (UTC) day × 24 h ≈ daily kWh.
    daily = frame.resample("1D").mean() * 24.0
    total = daily.sum(axis=1)
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in total.items()}, resolved


def _daily_temps_c(start: str | None, end: str | None) -> dict[str, dict[str, Any]]:
    """Return {date_str: {avg, min, max, source}} in Celsius.

    Uploaded rows win; dates without an upload fall back to the daily mean
    of cached NOAA hourly data so the feature works before any CSV upload.
    """
    out: dict[str, dict[str, Any]] = {}

    session = get_session()
    try:
        q = session.query(DailyTemperature)
        if start:
            q = q.filter(DailyTemperature.entry_date >= _parse_date(start))
        if end:
            q = q.filter(DailyTemperature.entry_date <= _parse_date(end))
        for r in q.all():
            out[r.entry_date.isoformat()] = {
                "avg": r.avg_temp_c,
                "min": r.min_temp_c,
                "max": r.max_temp_c,
                "source": "uploaded",
            }
    finally:
        session.close()

    station = (_get_config().get("weather", {}) or {}).get("station_id")
    if station and start and end:
        from ..weather import get_cached_weather

        try:
            s = datetime.strptime(str(start)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            e = datetime.strptime(str(end)[:10], "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            cached = get_cached_weather(station, s, e)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Weather cache lookup failed: %s", exc)
            cached = []
        buckets: dict[str, list[float]] = {}
        for rec in cached:
            t = rec.get("temperature_c")
            if t is None:
                continue
            day = str(rec.get("timestamp", ""))[:10]
            if day:
                buckets.setdefault(day, []).append(t)
        for day, vals in buckets.items():
            if day not in out and vals:
                out[day] = {
                    "avg": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "source": "noaa",
                }
    return out


# ─────────────────────────── temperature CRUD ──────────────────────────────


class TemperatureUpload(BaseModel):
    csv: str
    unit: str = "fahrenheit"
    replace: bool = False


def _parse_temp_csv(text: str) -> tuple[list[tuple[_date, float, float | None, float | None]], list[str]]:
    """Parse ``date,avg[,min[,max]]`` rows. Header row is auto-detected."""
    rows: list[tuple[_date, float, float | None, float | None]] = []
    errors: list[str] = []
    for ln, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        # A header row has a non-numeric value in the second column.
        if ln == 1:
            try:
                float(parts[1])
            except (ValueError, IndexError):
                continue
        if len(parts) < 2:
            errors.append(f"line {ln}: need at least date,avg_temp")
            continue
        try:
            d = _parse_date(parts[0])
        except Exception:
            errors.append(f"line {ln}: bad date '{parts[0]}'")
            continue

        def _num(i: int) -> float | None:
            if i < len(parts) and parts[i] not in ("", "-", "NA", "N/A"):
                return float(parts[i])
            return None

        try:
            avg = _num(1)
            mn = _num(2)
            mx = _num(3)
        except ValueError:
            errors.append(f"line {ln}: non-numeric temperature")
            continue
        if avg is None:
            errors.append(f"line {ln}: missing avg temperature")
            continue
        rows.append((d, avg, mn, mx))
    return rows, errors


@router.get("/hvac/temperatures")
def list_temperatures(
    start: str | None = Query(None),
    end: str | None = Query(None),
    unit: str | None = Query(None),
) -> dict[str, Any]:
    """List stored daily temperature rows, converted to the display unit."""
    unit = _resolve_unit(unit)
    session = get_session()
    try:
        q = session.query(DailyTemperature)
        if start:
            q = q.filter(DailyTemperature.entry_date >= _parse_date(start))
        if end:
            q = q.filter(DailyTemperature.entry_date <= _parse_date(end))
        rows = q.order_by(DailyTemperature.entry_date).all()
        items = [
            {
                "entry_date": r.entry_date.isoformat(),
                "avg_temp": _clean_float(_c_to_unit(r.avg_temp_c, unit)),
                "min_temp": _clean_float(_c_to_unit(r.min_temp_c, unit)),
                "max_temp": _clean_float(_c_to_unit(r.max_temp_c, unit)),
                "source": r.source,
            }
            for r in rows
        ]
        return {"unit": unit, "count": len(items), "items": items}
    finally:
        session.close()


@router.post("/hvac/temperatures/upload")
def upload_temperatures(body: TemperatureUpload) -> dict[str, Any]:
    """Bulk-upload daily temperatures from CSV text (``date,avg,min,max``)."""
    unit = "fahrenheit" if body.unit == "fahrenheit" else "celsius"
    rows, errors = _parse_temp_csv(body.csv or "")
    if not rows and not errors:
        raise HTTPException(status_code=400, detail="No rows found in CSV input")

    session = get_session()
    try:
        if body.replace:
            session.query(DailyTemperature).delete()

        inserted = updated = 0
        for d, avg, mn, mx in rows:
            existing = (
                session.query(DailyTemperature)
                .filter(DailyTemperature.entry_date == d)
                .one_or_none()
            )
            if existing:
                existing.avg_temp_c = _unit_to_c(avg, unit)
                existing.min_temp_c = _unit_to_c(mn, unit)
                existing.max_temp_c = _unit_to_c(mx, unit)
                existing.source = "upload"
                updated += 1
            else:
                session.add(
                    DailyTemperature(
                        entry_date=d,
                        avg_temp_c=_unit_to_c(avg, unit),
                        min_temp_c=_unit_to_c(mn, unit),
                        max_temp_c=_unit_to_c(mx, unit),
                        source="upload",
                    )
                )
                inserted += 1
        session.commit()
        total = session.query(DailyTemperature).count()
        return {"inserted": inserted, "updated": updated, "errors": errors, "total": total}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/hvac/temperatures")
def clear_temperatures() -> dict[str, int]:
    """Delete all stored daily temperature rows."""
    session = get_session()
    try:
        deleted = session.query(DailyTemperature).delete()
        session.commit()
        return {"deleted": deleted}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────── analysis ──────────────────────────────────


@router.get("/hvac/analysis")
def hvac_analysis(
    start: str | None = Query(None),
    end: str | None = Query(None),
    panels: str | None = Query(None, description="Comma-separated panel ids"),
    unit: str | None = Query(None, description="celsius or fahrenheit"),
    balance_point: float | None = Query(None, description="Balance point in the display unit"),
) -> dict[str, Any]:
    """Correlate daily energy against outdoor temperature for HVAC analysis."""
    unit = _resolve_unit(unit)
    deg = "°F" if unit == "fahrenheit" else "°C"
    panel_list = [p.strip() for p in panels.split(",") if p.strip()] if panels else []

    kwh_map, resolved_panels = _daily_kwh(start, end, panel_list)

    eff_start, eff_end = start, end
    if (not eff_start or not eff_end) and kwh_map:
        keys = sorted(kwh_map)
        eff_start = eff_start or keys[0]
        eff_end = eff_end or keys[-1]

    temps_map = _daily_temps_c(eff_start, eff_end)

    bp = float(balance_point) if balance_point is not None else _default_balance_point(unit)

    # ── Align kWh and temperature on shared dates ──
    common = sorted(set(kwh_map) & set(temps_map))
    days: list[dict[str, Any]] = []
    for d in common:
        kwh = kwh_map.get(d)
        tc = temps_map.get(d, {})
        avg_c = tc.get("avg")
        if kwh is None or avg_c is None:
            continue
        days.append(
            {
                "date": d,
                "kwh": round(float(kwh), 3),
                "avg_temp": _clean_float(_c_to_unit(avg_c, unit)),
                "min_temp": _clean_float(_c_to_unit(tc.get("min"), unit)),
                "max_temp": _clean_float(_c_to_unit(tc.get("max"), unit)),
                "source": tc.get("source", "uploaded"),
            }
        )

    coverage = {
        "kwh_days": len(kwh_map),
        "temp_days": len(temps_map),
        "matched_days": len(days),
        "uploaded_days": sum(1 for d in days if d["source"] == "uploaded"),
        "noaa_days": sum(1 for d in days if d["source"] == "noaa"),
    }

    if len(days) < MIN_DAYS_FOR_MODEL:
        return {
            "ok": False,
            "unit": unit,
            "balance_point": bp,
            "panels": resolved_panels,
            "coverage": coverage,
            "days": days,
            "message": (
                f"Need at least {MIN_DAYS_FOR_MODEL} days with both energy and "
                f"temperature data to build a model (have {len(days)}). "
                "Upload more daily temperatures that overlap your energy data."
            ),
        }

    # ── Degree-day regression: kWh = baseload + kh·HDD + kc·CDD ──
    n = len(days)
    y = np.array([d["kwh"] for d in days], dtype=float)
    temps = np.array([d["avg_temp"] for d in days], dtype=float)
    hdd = np.maximum(0.0, bp - temps)
    cdd = np.maximum(0.0, temps - bp)

    cols = [np.ones(n)]
    col_idx: dict[str, int] = {}
    if (hdd > 0).sum() >= 3 and hdd.std() > 1e-9:
        col_idx["heating"] = len(cols)
        cols.append(hdd)
    if (cdd > 0).sum() >= 3 and cdd.std() > 1e-9:
        col_idx["cooling"] = len(cols)
        cols.append(cdd)

    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef

    baseload = float(coef[0])
    heating_slope = float(coef[col_idx["heating"]]) if "heating" in col_idx else 0.0
    cooling_slope = float(coef[col_idx["cooling"]]) if "cooling" in col_idx else 0.0

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    pearson = (
        float(np.corrcoef(temps, y)[0, 1])
        if temps.std() > 1e-9 and y.std() > 1e-9
        else 0.0
    )

    resid = y - pred
    sigma = float(resid.std(ddof=1)) if n > 2 else float(resid.std())

    # ── Per-day model output + anomaly flags ──
    anomalies: list[dict[str, Any]] = []
    detect_anomalies = n >= MIN_DAYS_FOR_ANOMALY and sigma > 1e-9
    for i, d in enumerate(days):
        r = float(resid[i])
        d["hdd"] = _clean_float(hdd[i])
        d["cdd"] = _clean_float(cdd[i])
        d["predicted_kwh"] = _clean_float(pred[i])
        d["residual"] = _clean_float(r)
        d["anomaly"] = None
        if detect_anomalies and abs(r) > 2.0 * sigma:
            severity = "high" if r > 0 else "low"
            d["anomaly"] = severity
            pct = (r / pred[i] * 100.0) if pred[i] else None
            anomalies.append(
                {
                    "date": d["date"],
                    "kwh": d["kwh"],
                    "predicted_kwh": d["predicted_kwh"],
                    "excess_kwh": _clean_float(r),
                    "pct": _clean_float(pct),
                    "severity": severity,
                    "avg_temp": d["avg_temp"],
                }
            )
    anomalies.sort(key=lambda a: abs(a["excess_kwh"] or 0.0), reverse=True)

    # ── Baseload vs weather-driven energy split ──
    baseload_kwh = max(0.0, baseload) * n
    heating_kwh = max(0.0, heating_slope) * float(hdd.sum())
    cooling_kwh = max(0.0, cooling_slope) * float(cdd.sum())
    weather_kwh = heating_kwh + cooling_kwh
    modeled_total = baseload_kwh + weather_kwh
    actual_total = float(y.sum())
    baseload_pct = (baseload_kwh / modeled_total * 100.0) if modeled_total > 0 else 0.0

    # ── Energy-signature fit curve for the scatter chart ──
    t_lo, t_hi = float(temps.min()), float(temps.max())
    fit_t = np.linspace(t_lo, t_hi, 60)
    fit_kwh = (
        baseload
        + heating_slope * np.maximum(0.0, bp - fit_t)
        + cooling_slope * np.maximum(0.0, fit_t - bp)
    )

    # ── Plain-language insights ──
    insights = _build_insights(
        pearson=pearson,
        r2=r2,
        baseload=baseload,
        baseload_pct=baseload_pct,
        heating_slope=heating_slope,
        cooling_slope=cooling_slope,
        bp=bp,
        deg=deg,
        anomalies=anomalies,
        n_days=n,
    )

    return {
        "ok": True,
        "unit": unit,
        "balance_point": round(bp, 2),
        "panels": resolved_panels,
        "coverage": coverage,
        "days": days,
        "model": {
            "baseload_kwh_per_day": _clean_float(baseload),
            "heating_slope": _clean_float(heating_slope),
            "cooling_slope": _clean_float(cooling_slope),
            "r2": _clean_float(r2),
            "pearson_r": _clean_float(pearson),
        },
        "totals": {
            "n_days": n,
            "actual_kwh": _clean_float(actual_total),
            "total_hdd": _clean_float(float(hdd.sum())),
            "total_cdd": _clean_float(float(cdd.sum())),
        },
        "split": {
            "baseload_kwh": _clean_float(baseload_kwh),
            "heating_kwh": _clean_float(heating_kwh),
            "cooling_kwh": _clean_float(cooling_kwh),
            "weather_kwh": _clean_float(weather_kwh),
            "baseload_pct": _clean_float(baseload_pct),
        },
        "scatter": {
            "fit_temps": [_clean_float(v) for v in fit_t.tolist()],
            "fit_kwh": [_clean_float(v) for v in fit_kwh.tolist()],
        },
        "anomalies": anomalies,
        "insights": insights,
    }


def _build_insights(
    *,
    pearson: float,
    r2: float,
    baseload: float,
    baseload_pct: float,
    heating_slope: float,
    cooling_slope: float,
    bp: float,
    deg: str,
    anomalies: list[dict[str, Any]],
    n_days: int,
) -> list[str]:
    out: list[str] = []

    ar = abs(pearson)
    strength = (
        "very strong" if ar >= 0.8
        else "strong" if ar >= 0.6
        else "moderate" if ar >= 0.4
        else "weak" if ar >= 0.2
        else "negligible"
    )
    direction = "rises" if pearson > 0 else "falls"
    out.append(
        f"Energy use has a {strength} correlation with outdoor temperature "
        f"(r = {pearson:+.2f}) — consumption {direction} as it gets warmer. "
        f"The degree-day model explains {r2 * 100:.0f}% of day-to-day variation."
    )

    if baseload_pct >= 60:
        out.append(
            f"Baseload is {baseload_pct:.0f}% of modeled energy "
            f"({baseload:.1f} kWh/day that does not move with weather). "
            "A high baseload usually means equipment is left running outside "
            "occupied hours — a prime efficiency target."
        )
    else:
        out.append(
            f"Baseload is {baseload_pct:.0f}% of modeled energy "
            f"({baseload:.1f} kWh/day); weather-driven HVAC load is the larger share."
        )

    if cooling_slope > 0:
        out.append(
            f"Cooling sensitivity: {cooling_slope:.1f} kWh per cooling degree-day "
            f"above the {bp:g}{deg} balance point. Track this slope over time — "
            "a rising slope flags degrading chiller/AC efficiency."
        )
    if heating_slope > 0:
        out.append(
            f"Heating sensitivity: {heating_slope:.1f} kWh per heating degree-day "
            f"below the {bp:g}{deg} balance point."
        )
    if cooling_slope < 0 or heating_slope < 0:
        out.append(
            "A negative degree-day slope is physically unusual — the selected "
            "panels are probably not HVAC-dominated, or the balance point needs "
            "adjusting. Try narrowing the panel selection to HVAC equipment."
        )

    if anomalies:
        high = [a for a in anomalies if a["severity"] == "high"]
        out.append(
            f"{len(anomalies)} of {n_days} days deviate from the temperature-"
            f"predicted baseline ({len(high)} used more energy than expected). "
            "Investigate HVAC setpoints, simultaneous heating/cooling, stuck "
            "economizers, or doors left open on those days."
        )
        if high:
            w = high[0]
            out.append(
                f"Largest over-consumption: {w['date']} used "
                f"{w['excess_kwh']:.0f} kWh ({w['pct']:+.0f}%) above prediction."
            )
    else:
        out.append(
            "No major anomaly days detected — daily energy tracks temperature "
            "consistently, which suggests HVAC controls are behaving predictably."
        )

    if r2 < 0.3:
        out.append(
            "Temperature explains little of the daily variation (low R²). "
            "Energy here is driven mainly by production or scheduling rather "
            "than weather — select only HVAC-related panels for a cleaner signal."
        )

    return out
