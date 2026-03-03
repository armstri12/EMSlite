"""Power Usage Behavior analysis — phantom draw and narrative generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import amps_to_kw

# Shift definition: 6:00 AM to 2:30 PM Eastern Time
SHIFT_START_HOUR = 6
SHIFT_START_MINUTE = 0
SHIFT_END_HOUR = 14
SHIFT_END_MINUTE = 30
TIMEZONE = "America/New_York"

# Annualization constants
DAYS_PER_YEAR = 365
WORKDAYS_PER_YEAR = 260  # 52 weeks * 5 days

# Reduction scenario percentages
REDUCTION_LEVELS = [0.25, 0.50, 0.75, 1.00]


def classify_shift_hours(timestamps: pd.Series) -> pd.Series:
    """Classify each timestamp as 'shift' or 'off_shift'.

    Shift hours: 6:00 AM - 2:30 PM America/New_York, weekdays only.
    Weekends are entirely off-shift.
    """
    eastern = timestamps.dt.tz_convert(TIMEZONE)

    is_weekday = eastern.dt.weekday < 5  # Mon=0..Fri=4

    time_decimal = eastern.dt.hour + eastern.dt.minute / 60.0
    shift_start = SHIFT_START_HOUR + SHIFT_START_MINUTE / 60.0  # 6.0
    shift_end = SHIFT_END_HOUR + SHIFT_END_MINUTE / 60.0  # 14.5

    is_shift_time = (time_decimal >= shift_start) & (time_decimal < shift_end)
    is_shift = is_weekday & is_shift_time

    return pd.Series(
        np.where(is_shift, "shift", "off_shift"),
        index=timestamps.index,
    )


def compute_hourly_profile(
    timestamps: pd.Series,
    kw: pd.Series,
    shift_class: pd.Series,
) -> dict[str, Any]:
    """Compute average kW for each hour (0-23) in Eastern time.

    Returns overall, shift-only, and off-shift-only averages per hour.
    """
    eastern = timestamps.dt.tz_convert(TIMEZONE)
    hours = eastern.dt.hour

    df = pd.DataFrame({"hour": hours, "kw": kw, "shift": shift_class})

    overall = df.groupby("hour")["kw"].mean()
    shift_only = df[df["shift"] == "shift"].groupby("hour")["kw"].mean()
    off_only = df[df["shift"] == "off_shift"].groupby("hour")["kw"].mean()

    all_hours = list(range(24))
    return {
        "hours": all_hours,
        "avg_kw": [round(float(overall.get(h, 0)), 4) for h in all_hours],
        "shift_avg_kw": [round(float(shift_only.get(h, 0)), 4) for h in all_hours],
        "off_shift_avg_kw": [round(float(off_only.get(h, 0)), 4) for h in all_hours],
    }


def compute_phantom_draw(
    timestamps: pd.Series,
    kw: pd.Series,
) -> dict[str, Any]:
    """Estimate phantom draw from deep off-hour consumption.

    Deep off-hours: 11 PM - 4 AM Eastern, weeknights.
    Phantom draw = 25th percentile of kW during deep off-hours
    (robust to zero-readings / outages).
    """
    eastern = timestamps.dt.tz_convert(TIMEZONE)
    hour = eastern.dt.hour
    weekday = eastern.dt.weekday

    # Deep off-hours: 11 PM (23) through 4 AM (0,1,2,3), weeknights
    is_deep_off = (weekday < 5) & ((hour >= 23) | (hour < 4))

    deep_kw = kw[is_deep_off]

    if deep_kw.empty:
        return {
            "phantom_kw": 0.0,
            "deep_off_hours_avg_kw": 0.0,
            "deep_off_hours_median_kw": 0.0,
            "deep_off_hours_count": 0,
        }

    return {
        "phantom_kw": round(float(deep_kw.quantile(0.25)), 4),
        "deep_off_hours_avg_kw": round(float(deep_kw.mean()), 4),
        "deep_off_hours_median_kw": round(float(deep_kw.median()), 4),
        "deep_off_hours_count": int(is_deep_off.sum()),
    }


def compute_shift_energy_split(
    timestamps: pd.Series,
    kw: pd.Series,
    shift_class: pd.Series,
) -> dict[str, Any]:
    """Split total energy consumption into shift vs off-shift.

    Uses left-point rectangular integration: kwh = kw * dt_hours, where dt_hours
    is the interval to the *next* sample. This is an approximation; for uniform
    cadence the error is negligible, but it will diverge with irregular sampling.
    """
    dt_hours = timestamps.diff().dt.total_seconds().fillna(0) / 3600.0
    kwh = kw * dt_hours

    shift_mask = shift_class == "shift"

    total = float(kwh.sum())
    shift_kwh = float(kwh[shift_mask].sum())
    off_shift_kwh = float(kwh[~shift_mask].sum())
    shift_hours = float(dt_hours[shift_mask].sum())
    off_shift_hours = float(dt_hours[~shift_mask].sum())

    return {
        "total_kwh": round(total, 2),
        "shift_kwh": round(shift_kwh, 2),
        "off_shift_kwh": round(off_shift_kwh, 2),
        "shift_pct": round(shift_kwh / total * 100, 1) if total > 0 else 0,
        "off_shift_pct": round(off_shift_kwh / total * 100, 1) if total > 0 else 0,
        "shift_hours_total": round(shift_hours, 1),
        "off_shift_hours_total": round(off_shift_hours, 1),
    }


def compute_annualized_phantom_cost(
    phantom_kw: float,
    price_per_kwh: float,
    carbon_kg_per_kwh: float,
) -> dict[str, Any]:
    """Annualize phantom draw cost and carbon impact.

    Off-shift hours per year:
      Workdays (260): 15.5 hours off-shift * 260 = 4,030 hours
      Weekends (105 days): 24 * 105 = 2,520 hours
      Total: 6,550 off-shift hours per year
    """
    workday_off_hours = (24 - 8.5) * WORKDAYS_PER_YEAR  # 15.5h * 260 = 4030
    weekend_hours = (DAYS_PER_YEAR - WORKDAYS_PER_YEAR) * 24  # 105 * 24 = 2520
    total_off_shift_hours_year = workday_off_hours + weekend_hours  # 6550

    annual_phantom_kwh = phantom_kw * total_off_shift_hours_year
    annual_cost = annual_phantom_kwh * price_per_kwh
    annual_carbon_kg = annual_phantom_kwh * carbon_kg_per_kwh

    return {
        "off_shift_hours_per_year": round(total_off_shift_hours_year, 0),
        "annual_phantom_kwh": round(annual_phantom_kwh, 2),
        "annual_phantom_cost": round(annual_cost, 2),
        "annual_phantom_carbon_kg": round(annual_carbon_kg, 2),
        "annual_phantom_carbon_tonnes": round(annual_carbon_kg / 1000, 3),
    }


def compute_reduction_scenarios(
    phantom_kw: float,
    price_per_kwh: float,
    carbon_kg_per_kwh: float,
) -> list[dict[str, Any]]:
    """Model savings for different phantom draw reduction levels."""
    annualized = compute_annualized_phantom_cost(
        phantom_kw, price_per_kwh, carbon_kg_per_kwh
    )
    base_kwh = annualized["annual_phantom_kwh"]

    scenarios = []
    for pct in REDUCTION_LEVELS:
        saved_kwh = base_kwh * pct
        saved_carbon = saved_kwh * carbon_kg_per_kwh
        scenarios.append(
            {
                "reduction_pct": int(pct * 100),
                "reduced_phantom_kw": round(phantom_kw * (1 - pct), 4),
                "annual_kwh_saved": round(saved_kwh, 2),
                "annual_cost_saved": round(saved_kwh * price_per_kwh, 2),
                "annual_carbon_kg_saved": round(saved_carbon, 2),
                # Relatable equivalents
                "equivalent_homes_powered_days": round(saved_kwh / 30, 0),
                # avg US home ~30 kWh/day
                "equivalent_trees_year": round(saved_carbon / 22, 1),
                # one tree absorbs ~22 kg CO2/year
                "equivalent_miles_driven": round(saved_carbon / 0.404, 0),
                # avg car emits 0.404 kg CO2/mile
            }
        )

    return scenarios


def generate_narrative(
    panel_name: str,
    phantom: dict[str, Any],
    energy_split: dict[str, Any],
    annualized: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, str]:
    """Generate behavioral narrative text sections."""
    phantom_kw = phantom["phantom_kw"]
    off_pct = energy_split["off_shift_pct"]
    annual_cost = annualized["annual_phantom_cost"]
    annual_carbon_kg = annualized["annual_phantom_carbon_kg"]

    # Headline — adapts to severity
    if off_pct >= 50:
        headline = (
            f"{panel_name} consumes more energy when nobody is working "
            f"than during the entire shift."
        )
    elif off_pct >= 30:
        headline = (
            f"Nearly {off_pct:.0f}% of {panel_name}'s energy is consumed "
            f"outside of shift hours."
        )
    else:
        headline = (
            f"{panel_name} has {off_pct:.0f}% off-shift energy consumption "
            f"that could be reduced."
        )

    # Phantom draw story
    phantom_story = (
        f"Even when the facility is empty, {panel_name} draws an estimated "
        f"{phantom_kw:.2f} kW continuously. This phantom draw runs silently "
        f"through every night, weekend, and holiday \u2014 "
        f"accumulating {annualized['annual_phantom_kwh']:,.0f} kWh per year "
        f"of completely wasted energy."
    )

    # Cost impact
    full_scenario = next((s for s in scenarios if s["reduction_pct"] == 100), None)
    trees = full_scenario["equivalent_trees_year"] if full_scenario else 0

    cost_impact = (
        f"This phantom draw costs ${annual_cost:,.2f} per year and produces "
        f"{annual_carbon_kg:,.0f} kg of CO\u2082 \u2014 equivalent to "
        f"{trees:.0f} trees working for an entire year to offset."
    )

    # Call to action
    half_scenario = next((s for s in scenarios if s["reduction_pct"] == 50), None)
    if half_scenario and full_scenario:
        call_to_action = (
            f"Cutting phantom draw in half would save "
            f"${half_scenario['annual_cost_saved']:,.2f} annually and prevent "
            f"{half_scenario['annual_carbon_kg_saved']:,.0f} kg of CO\u2082. "
            f"Eliminating it entirely saves "
            f"${full_scenario['annual_cost_saved']:,.2f}. "
            f"Simple actions like power strips, smart outlets, or shutdown "
            f"procedures at end of shift can make this happen."
        )
    else:
        call_to_action = (
            "Reducing phantom draw saves money and reduces carbon emissions."
        )

    return {
        "headline": headline,
        "phantom_story": phantom_story,
        "cost_impact": cost_impact,
        "call_to_action": call_to_action,
    }


def compute_phantom_rankings(
    df: pd.DataFrame,
    panel_cols: list[str],
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
    display_names: dict[str, str] | None = None,
    voltage_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Rank all panels by phantom draw waste.

    Computes phantom draw, off-shift energy, and annualized costs for every
    panel, then returns a sorted ranking (worst offenders first).
    """
    names = display_names or {}
    vm = voltage_map or {}
    timestamps = df["Timestamp"]
    dt_hours = timestamps.diff().dt.total_seconds().fillna(0) / 3600.0

    # Classify once — shared across all panels
    shift_class = classify_shift_hours(timestamps)
    shift_mask = shift_class == "shift"

    rankings: list[dict[str, Any]] = []
    for col in panel_cols:
        if col not in df.columns:
            continue
        v = vm.get(col, line_voltage)
        kw = amps_to_kw(df[col].fillna(0), v, power_factor)

        phantom = compute_phantom_draw(timestamps, kw)
        phantom_kw = phantom["phantom_kw"]

        kwh = kw * dt_hours
        total_kwh = float(kwh.sum())
        off_shift_kwh = float(kwh[~shift_mask].sum())
        off_shift_pct = (
            round(off_shift_kwh / total_kwh * 100, 1) if total_kwh > 0 else 0.0
        )

        ann = compute_annualized_phantom_cost(
            phantom_kw, price_per_kwh, carbon_kg_per_kwh
        )

        rankings.append({
            "panel_id": col,
            "display_name": names.get(col, col),
            "phantom_kw": round(phantom_kw, 4),
            "off_shift_pct": off_shift_pct,
            "annual_phantom_kwh": ann["annual_phantom_kwh"],
            "annual_phantom_cost": ann["annual_phantom_cost"],
            "annual_phantom_carbon_kg": ann["annual_phantom_carbon_kg"],
            "total_kwh": round(total_kwh, 2),
            "off_shift_kwh": round(off_shift_kwh, 2),
        })

    rankings.sort(key=lambda r: r["annual_phantom_cost"], reverse=True)

    total_cost = sum(r["annual_phantom_cost"] for r in rankings)
    total_kwh = sum(r["annual_phantom_kwh"] for r in rankings)
    total_carbon = sum(r["annual_phantom_carbon_kg"] for r in rankings)

    return {
        "rankings": rankings,
        "facility_totals": {
            "total_phantom_cost": round(total_cost, 2),
            "total_phantom_kwh": round(total_kwh, 2),
            "total_phantom_carbon_kg": round(total_carbon, 2),
            "panel_count": len(rankings),
        },
        "data_points": len(df),
        "date_range": {
            "start": timestamps.min().isoformat() if not timestamps.empty else None,
            "end": timestamps.max().isoformat() if not timestamps.empty else None,
        },
    }


def analyze_behavior(
    df: pd.DataFrame,
    panel_col: str,
    line_voltage: float = 480.0,
    power_factor: float = 1.0,
    price_per_kwh: float = 0.25,
    carbon_kg_per_kwh: float = 0.4,
    panel_display_name: str | None = None,
    voltage_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full behavior analysis for a single panel.

    Args:
        df: Wide-format DataFrame with Timestamp column and panel amps columns.
        panel_col: The column name of the panel to analyze.
        line_voltage: System voltage for amps-to-kW conversion.
        power_factor: Power factor for conversion.
        price_per_kwh: Electricity cost rate.
        carbon_kg_per_kwh: Carbon intensity factor.
        panel_display_name: Human-friendly name (defaults to panel_col).
        voltage_map: Per-device voltage overrides {device_id: voltage}.

    Returns:
        Complete analysis dict with all sub-results.
    """
    display_name = panel_display_name or panel_col
    vm = voltage_map or {}
    v = vm.get(panel_col, line_voltage)

    timestamps = df["Timestamp"]
    kw = amps_to_kw(df[panel_col].fillna(0), v, power_factor)

    shift_class = classify_shift_hours(timestamps)
    hourly = compute_hourly_profile(timestamps, kw, shift_class)
    phantom = compute_phantom_draw(timestamps, kw)
    energy_split = compute_shift_energy_split(timestamps, kw, shift_class)
    annualized = compute_annualized_phantom_cost(
        phantom["phantom_kw"], price_per_kwh, carbon_kg_per_kwh
    )
    scenarios = compute_reduction_scenarios(
        phantom["phantom_kw"], price_per_kwh, carbon_kg_per_kwh
    )
    narrative = generate_narrative(
        display_name, phantom, energy_split, annualized, scenarios
    )

    return {
        "panel_id": panel_col,
        "panel_display_name": display_name,
        "config": {
            "shift_start": f"{SHIFT_START_HOUR:02d}:{SHIFT_START_MINUTE:02d}",
            "shift_end": f"{SHIFT_END_HOUR:02d}:{SHIFT_END_MINUTE:02d}",
            "timezone": TIMEZONE,
            "price_per_kwh": price_per_kwh,
            "carbon_kg_per_kwh": carbon_kg_per_kwh,
        },
        "hourly_profile": hourly,
        "phantom_draw": phantom,
        "energy_split": energy_split,
        "annualized_phantom": annualized,
        "reduction_scenarios": scenarios,
        "narrative": narrative,
        "data_points": len(df),
        "date_range": {
            "start": timestamps.min().isoformat() if not timestamps.empty else None,
            "end": timestamps.max().isoformat() if not timestamps.empty else None,
        },
    }
