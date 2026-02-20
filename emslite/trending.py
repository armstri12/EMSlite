"""Power Usage Trending analysis — per-panel trend snapshot and detail."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import amps_to_kw


def _daily_kwh(timestamps: pd.Series, kw: pd.Series) -> pd.Series:
    """Compute daily energy (kWh) via left-point rectangular integration.

    Each sample contributes kw * dt_hours, where dt_hours is the elapsed time
    since the previous sample. This is rectangular (not trapezoidal) integration.
    For uniform cadence the error is negligible; divergence grows with irregular sampling.
    """
    dt_hours = timestamps.diff().dt.total_seconds().fillna(0) / 3600.0
    kwh = kw * dt_hours
    dates = timestamps.dt.date
    return kwh.groupby(dates).sum()


def _split_periods(
    df: pd.DataFrame,
    period_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into recent and prior period of equal length."""
    end = df["Timestamp"].max()
    recent_start = end - pd.Timedelta(days=period_days)
    prior_start = recent_start - pd.Timedelta(days=period_days)

    recent = df[df["Timestamp"] > recent_start]
    prior = df[(df["Timestamp"] > prior_start) & (df["Timestamp"] <= recent_start)]
    return recent, prior


def compute_trending_snapshot(
    df: pd.DataFrame,
    panel_cols: list[str],
    period_days: int = 7,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
    display_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute trend snapshot for all panels.

    Compares the most recent `period_days` against the prior equal period.
    Returns per-panel trend metrics ranked by absolute change.
    """
    names = display_names or {}
    timestamps = df["Timestamp"]

    recent_df, prior_df = _split_periods(df, period_days)

    panels: list[dict[str, Any]] = []
    total_recent_kwh = 0.0
    total_prior_kwh = 0.0
    rising_count = 0
    falling_count = 0
    stable_count = 0

    for col in panel_cols:
        if col not in df.columns:
            continue

        # Full period kW series
        kw_full = amps_to_kw(df[col].fillna(0), line_voltage, power_factor)

        # Recent period
        kw_recent = amps_to_kw(recent_df[col].fillna(0), line_voltage, power_factor)
        recent_kwh_series = _daily_kwh(recent_df["Timestamp"], kw_recent)
        recent_kwh = float(recent_kwh_series.sum())
        recent_avg_kw = float(kw_recent.mean()) if len(kw_recent) > 0 else 0.0
        recent_peak_kw = float(kw_recent.max()) if len(kw_recent) > 0 else 0.0

        # Prior period
        kw_prior = amps_to_kw(prior_df[col].fillna(0), line_voltage, power_factor)
        prior_kwh_series = _daily_kwh(prior_df["Timestamp"], kw_prior)
        prior_kwh = float(prior_kwh_series.sum())
        prior_avg_kw = float(kw_prior.mean()) if len(kw_prior) > 0 else 0.0

        # Trend calculation
        if prior_kwh > 0:
            pct_change = round((recent_kwh - prior_kwh) / prior_kwh * 100, 1)
        elif recent_kwh > 0:
            pct_change = 100.0
        else:
            pct_change = 0.0

        if pct_change > 5:
            direction = "rising"
            rising_count += 1
        elif pct_change < -5:
            direction = "falling"
            falling_count += 1
        else:
            direction = "stable"
            stable_count += 1

        # Daily kWh for sparkline (full dataset, last period_days*2 max)
        daily_full = _daily_kwh(timestamps, kw_full)
        spark_days = daily_full.tail(period_days * 2)
        spark_dates = [str(d) for d in spark_days.index]
        spark_values = [round(float(v), 2) for v in spark_days.values]

        # Cost
        recent_cost = recent_kwh * price_per_kwh
        prior_cost = prior_kwh * price_per_kwh
        cost_change = recent_cost - prior_cost

        total_recent_kwh += recent_kwh
        total_prior_kwh += prior_kwh

        panels.append({
            "panel_id": col,
            "display_name": names.get(col, col),
            "recent_kwh": round(recent_kwh, 2),
            "prior_kwh": round(prior_kwh, 2),
            "recent_avg_kw": round(recent_avg_kw, 4),
            "prior_avg_kw": round(prior_avg_kw, 4),
            "recent_peak_kw": round(recent_peak_kw, 4),
            "pct_change": pct_change,
            "direction": direction,
            "recent_cost": round(recent_cost, 2),
            "prior_cost": round(prior_cost, 2),
            "cost_change": round(cost_change, 2),
            "spark_dates": spark_dates,
            "spark_values": spark_values,
        })

    # Sort by absolute pct_change descending (biggest movers first)
    panels.sort(key=lambda p: abs(p["pct_change"]), reverse=True)

    # Facility-level totals.
    # When prior=0 and recent>0, return 100.0 to match per-panel fallback semantics
    # (new energy appeared; treating as 100% increase rather than undefined).
    if total_prior_kwh > 0:
        total_pct = round((total_recent_kwh - total_prior_kwh) / total_prior_kwh * 100, 1)
    elif total_recent_kwh > 0:
        total_pct = 100.0
    else:
        total_pct = 0.0

    return {
        "panels": panels,
        "summary": {
            "panel_count": len(panels),
            "rising_count": rising_count,
            "falling_count": falling_count,
            "stable_count": stable_count,
            "total_recent_kwh": round(total_recent_kwh, 2),
            "total_prior_kwh": round(total_prior_kwh, 2),
            "total_pct_change": total_pct,
            "total_recent_cost": round(total_recent_kwh * price_per_kwh, 2),
            "total_prior_cost": round(total_prior_kwh * price_per_kwh, 2),
            "period_days": period_days,
        },
        "data_points": len(df),
        "date_range": {
            "start": timestamps.min().isoformat() if not timestamps.empty else None,
            "end": timestamps.max().isoformat() if not timestamps.empty else None,
        },
    }


def compute_trending_detail(
    df: pd.DataFrame,
    panel_col: str,
    period_days: int = 7,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
    panel_display_name: str | None = None,
) -> dict[str, Any]:
    """Full trending detail for a single panel.

    Returns daily energy series, rolling average, hourly profiles for
    recent and prior periods, and the raw time series for charting.
    """
    display_name = panel_display_name or panel_col
    timestamps = df["Timestamp"]
    kw = amps_to_kw(df[panel_col].fillna(0), line_voltage, power_factor)

    recent_df, prior_df = _split_periods(df, period_days)
    kw_recent = amps_to_kw(recent_df[panel_col].fillna(0), line_voltage, power_factor)
    kw_prior = amps_to_kw(prior_df[panel_col].fillna(0), line_voltage, power_factor)

    # --- Daily energy series (full dataset) ---
    daily = _daily_kwh(timestamps, kw)
    daily_dates = [str(d) for d in daily.index]
    daily_values = [round(float(v), 2) for v in daily.values]

    # --- 7-day rolling average of daily kWh ---
    daily_series = pd.Series(daily.values, index=daily.index)
    rolling_7d = daily_series.rolling(window=min(7, len(daily_series)), min_periods=1).mean()
    rolling_dates = [str(d) for d in rolling_7d.index]
    rolling_values = [round(float(v), 2) for v in rolling_7d.values]

    # --- Period comparison metrics ---
    recent_kwh = float(_daily_kwh(recent_df["Timestamp"], kw_recent).sum())
    prior_kwh = float(_daily_kwh(prior_df["Timestamp"], kw_prior).sum())
    recent_avg = float(kw_recent.mean()) if len(kw_recent) > 0 else 0.0
    prior_avg = float(kw_prior.mean()) if len(kw_prior) > 0 else 0.0
    recent_peak = float(kw_recent.max()) if len(kw_recent) > 0 else 0.0
    prior_peak = float(kw_prior.max()) if len(kw_prior) > 0 else 0.0

    pct_change = (
        round((recent_kwh - prior_kwh) / prior_kwh * 100, 1)
        if prior_kwh > 0
        else (100.0 if recent_kwh > 0 else 0.0)
    )

    # --- Hourly profile: recent vs prior ---
    def _hourly_profile(ts, kw_vals):
        if len(ts) == 0:
            return [0.0] * 24
        hours = ts.dt.hour
        df_h = pd.DataFrame({"hour": hours, "kw": kw_vals.values})
        avg = df_h.groupby("hour")["kw"].mean()
        return [round(float(avg.get(h, 0)), 4) for h in range(24)]

    recent_hourly = _hourly_profile(recent_df["Timestamp"], kw_recent)
    prior_hourly = _hourly_profile(prior_df["Timestamp"], kw_prior)

    # --- Time series for chart (last period_days * 3 for context) ---
    lookback = pd.Timedelta(days=period_days * 3)
    cutoff = timestamps.max() - lookback
    chart_mask = timestamps > cutoff
    chart_ts = [t.isoformat() for t in timestamps[chart_mask]]
    chart_kw = [round(float(v), 4) for v in kw[chart_mask]]

    # --- Rolling mean for the chart series ---
    window_hours = max(24, period_days * 2)
    kw_arr = np.array(chart_kw)
    window_pts = max(1, len(chart_ts) // max(1, period_days * 3) * 1)
    # Use a simple rolling window in points (~24h worth)
    if len(chart_kw) > 0:
        ts_dt = pd.to_datetime(chart_ts)
        dt_hours_arr = ts_dt.to_series().diff().dt.total_seconds().fillna(0) / 3600.0
        avg_interval = float(dt_hours_arr.mean()) if len(dt_hours_arr) > 1 else 1.0
        window_n = max(1, int(24 / avg_interval)) if avg_interval > 0 else 24
        roll = pd.Series(kw_arr).rolling(window=window_n, min_periods=1).mean()
        chart_rolling = [round(float(v), 4) for v in roll]
    else:
        chart_rolling = []

    return {
        "panel_id": panel_col,
        "panel_display_name": display_name,
        "period_days": period_days,
        "comparison": {
            "recent_kwh": round(recent_kwh, 2),
            "prior_kwh": round(prior_kwh, 2),
            "pct_change": pct_change,
            "recent_avg_kw": round(recent_avg, 4),
            "prior_avg_kw": round(prior_avg, 4),
            "recent_peak_kw": round(recent_peak, 4),
            "prior_peak_kw": round(prior_peak, 4),
            "recent_cost": round(recent_kwh * price_per_kwh, 2),
            "prior_cost": round(prior_kwh * price_per_kwh, 2),
            "cost_change": round((recent_kwh - prior_kwh) * price_per_kwh, 2),
            "direction": "rising" if pct_change > 5 else ("falling" if pct_change < -5 else "stable"),
        },
        "daily_energy": {
            "dates": daily_dates,
            "values": daily_values,
        },
        "rolling_avg": {
            "dates": rolling_dates,
            "values": rolling_values,
        },
        "hourly_profile": {
            "hours": list(range(24)),
            "recent_avg_kw": recent_hourly,
            "prior_avg_kw": prior_hourly,
        },
        "chart_series": {
            "timestamps": chart_ts,
            "kw": chart_kw,
            "rolling_kw": chart_rolling,
        },
        "data_points": len(df),
        "date_range": {
            "start": timestamps.min().isoformat() if not timestamps.empty else None,
            "end": timestamps.max().isoformat() if not timestamps.empty else None,
        },
    }
