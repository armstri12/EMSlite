"""Weekly energy report — data assembly, narrative, and HTML rendering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .behavior import compute_phantom_rankings
from .core import get_device_voltage_map, load_csv, meter_columns
from .metrics import compute_department_breakdown, compute_kpi, compute_panel_rankings
from .trending import compute_trending_snapshot


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def generate_report_data(period_days: int = 7) -> dict[str, Any]:
    """Assemble all data needed for the weekly report.

    Loads CSV, computes KPIs for this week, prior week, and 4-week average,
    then gathers department breakdown, high runners, trending, and phantom draw.
    """
    from .api.routes_data import (
        _get_all_department_panels,
        _get_config,
        _get_device_display_names,
        _get_master_path,
    )

    master = _get_master_path()
    if not master.exists():
        return {"error": "No data file found"}

    cfg = _get_config()
    df = load_csv(master)
    if df.empty:
        return {"error": "Data file is empty"}

    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    price_per_kwh = float(cfg.get("price_per_kwh", 0.25))
    carbon_kg_per_kwh = float(cfg.get("carbon_kg_per_kwh", 0.4))
    voltage_map = get_device_voltage_map()

    panel_cols = meter_columns(
        df.columns, exclude=set(cfg.get("combo_columns", {}).keys())
    )
    department_panels = _get_all_department_panels()
    display_names = _get_device_display_names()

    # --- Time windows ---
    latest = df["Timestamp"].max()
    week_start = latest - pd.Timedelta(days=period_days)
    prior_week_start = week_start - pd.Timedelta(days=period_days)
    four_week_start = latest - pd.Timedelta(days=period_days * 4)

    df_this = df[df["Timestamp"] > week_start].copy()
    df_prior = df[(df["Timestamp"] > prior_week_start) & (df["Timestamp"] <= week_start)].copy()
    df_4week = df[df["Timestamp"] > four_week_start].copy()

    common = dict(
        line_voltage=line_voltage,
        power_factor=power_factor,
        price_per_kwh=price_per_kwh,
        carbon_kg_per_kwh=carbon_kg_per_kwh,
        panel_cols=panel_cols,
        voltage_map=voltage_map,
    )

    # --- Facility KPIs ---
    kpi_this = compute_kpi(df_this, **common) if len(df_this) > 1 else _zero_kpi()
    kpi_prior = compute_kpi(df_prior, **common) if len(df_prior) > 1 else _zero_kpi()
    kpi_4week = compute_kpi(df_4week, **common) if len(df_4week) > 1 else _zero_kpi()

    # Weekly average from 4-week window
    weeks_in_window = max(1, kpi_4week["date_range_days"] / 7) if kpi_4week["date_range_days"] else 1
    avg_weekly_kwh = kpi_4week["total_kwh"] / weeks_in_window
    avg_weekly_cost = kpi_4week["total_cost"] / weeks_in_window
    avg_weekly_carbon_kg = kpi_4week["total_carbon_kg"] / weeks_in_window

    # --- Week-over-week deltas ---
    def _delta(curr, prev):
        diff = curr - prev
        pct = (diff / prev * 100) if prev else 0.0
        return {"value": round(diff, 2), "pct": round(pct, 1)}

    wow = {
        "kwh": _delta(kpi_this["total_kwh"], kpi_prior["total_kwh"]),
        "cost": _delta(kpi_this["total_cost"], kpi_prior["total_cost"]),
        "carbon_kg": _delta(kpi_this["total_carbon_kg"], kpi_prior["total_carbon_kg"]),
        "peak_kw": _delta(kpi_this["peak_kw"], kpi_prior["peak_kw"]),
        "avg_kw": _delta(kpi_this["avg_kw"], kpi_prior["avg_kw"]),
        "load_factor": _delta(kpi_this["load_factor"], kpi_prior["load_factor"]),
    }

    # --- vs 4-week average ---
    vs_avg = {
        "kwh": _delta(kpi_this["total_kwh"], avg_weekly_kwh),
        "cost": _delta(kpi_this["total_cost"], avg_weekly_cost),
        "carbon_kg": _delta(kpi_this["total_carbon_kg"], avg_weekly_carbon_kg),
    }

    # --- Department breakdown (this week + prior week) ---
    dept_common = dict(
        line_voltage=line_voltage,
        power_factor=power_factor,
        price_per_kwh=price_per_kwh,
        carbon_kg_per_kwh=carbon_kg_per_kwh,
        voltage_map=voltage_map,
    )
    depts_this = compute_department_breakdown(df_this, department_panels, **dept_common) if len(df_this) > 1 else []
    depts_prior = compute_department_breakdown(df_prior, department_panels, **dept_common) if len(df_prior) > 1 else []
    prior_by_name = {d["department"]: d for d in depts_prior}
    for d in depts_this:
        p = prior_by_name.get(d["department"], {})
        prev_kwh = p.get("total_kwh", 0)
        d["wow_kwh_pct"] = round((d["total_kwh"] - prev_kwh) / prev_kwh * 100, 1) if prev_kwh else 0.0
        d["wow_kwh_delta"] = round(d["total_kwh"] - prev_kwh, 2)

    # --- High runners (top 10 panels) ---
    rankings_common = dict(
        line_voltage=line_voltage,
        power_factor=power_factor,
        panel_cols=panel_cols,
        top_n=10,
        voltage_map=voltage_map,
    )
    high_runners = compute_panel_rankings(df_this, **rankings_common) if len(df_this) > 1 else []
    # Enrich with display names and department
    panel_to_dept = {}
    for dept_name, panels in department_panels.items():
        for p in panels:
            panel_to_dept[p] = dept_name
    for r in high_runners:
        r["display_name"] = display_names.get(r["panel_id"], r["panel_id"])
        r["department"] = panel_to_dept.get(r["panel_id"], "Unassigned")

    # --- Trending snapshot (significant changes) ---
    trending = compute_trending_snapshot(
        df, panel_cols, period_days=period_days,
        line_voltage=line_voltage, power_factor=power_factor,
        price_per_kwh=price_per_kwh, carbon_kg_per_kwh=carbon_kg_per_kwh,
        display_names=display_names, voltage_map=voltage_map,
    ) if len(df) > 1 else {"panels": [], "summary": {}}

    significant = [p for p in trending.get("panels", []) if abs(p.get("pct_change", 0)) > 10]
    increases = [p for p in significant if p.get("direction") == "rising"][:5]
    decreases = [p for p in significant if p.get("direction") == "falling"][:5]

    # --- Phantom draw ---
    phantom = compute_phantom_rankings(
        df_this, panel_cols,
        line_voltage=line_voltage, power_factor=power_factor,
        price_per_kwh=price_per_kwh, carbon_kg_per_kwh=carbon_kg_per_kwh,
        display_names=display_names, voltage_map=voltage_map,
    ) if len(df_this) > 1 else {"rankings": [], "facility_totals": {}}

    top_phantom = phantom.get("rankings", [])[:5]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {
            "start": week_start.isoformat(),
            "end": latest.isoformat(),
            "days": period_days,
        },
        "facility_kpi": kpi_this,
        "prior_kpi": kpi_prior,
        "four_week_avg": {
            "weekly_kwh": round(avg_weekly_kwh, 2),
            "weekly_cost": round(avg_weekly_cost, 2),
            "weekly_carbon_kg": round(avg_weekly_carbon_kg, 2),
        },
        "wow": wow,
        "vs_avg": vs_avg,
        "departments": depts_this,
        "high_runners": high_runners,
        "increases": increases,
        "decreases": decreases,
        "trending_summary": trending.get("summary", {}),
        "phantom": {
            "facility_totals": phantom.get("facility_totals", {}),
            "top_offenders": top_phantom,
        },
    }


def _zero_kpi() -> dict[str, Any]:
    return {
        "total_kwh": 0, "total_cost": 0, "total_carbon_kg": 0,
        "total_carbon_tonnes": 0, "peak_kw": 0, "avg_kw": 0,
        "load_factor": 0, "device_count": 0, "latest_timestamp": None,
        "date_range_days": 0,
    }


# ---------------------------------------------------------------------------
# YTD Cost Allocation Report — data assembly
# ---------------------------------------------------------------------------

_MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def generate_ytd_report_data(year: int | None = None) -> dict[str, Any]:
    """Assemble data for the YTD cost allocation report.

    Computes facility and per-department metrics for the year to date,
    including a month-by-month breakdown and optional prior-year comparison.
    """
    from .api.routes_data import (
        _get_all_department_panels,
        _get_config,
        _get_master_path,
    )

    master = _get_master_path()
    if not master.exists():
        return {"error": "No data file found"}

    cfg = _get_config()
    df = load_csv(master)
    if df.empty:
        return {"error": "Data file is empty"}

    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    price_per_kwh = float(cfg.get("price_per_kwh", 0.25))
    carbon_kg_per_kwh = float(cfg.get("carbon_kg_per_kwh", 0.4))
    voltage_map = get_device_voltage_map()

    panel_cols = meter_columns(
        df.columns, exclude=set(cfg.get("combo_columns", {}).keys())
    )
    department_panels = _get_all_department_panels()

    # Filter panel_cols to only enabled devices (meter_columns returns all
    # CSV columns regardless of Device.enabled; department_panels is already
    # filtered to enabled devices by _get_all_department_panels).
    enabled_panels: set[str] = set()
    for panels in department_panels.values():
        enabled_panels.update(panels)
    panel_cols = [p for p in panel_cols if p in enabled_panels]

    # --- Determine YTD window ---
    latest = df["Timestamp"].max()
    report_year = year or latest.year
    ytd_start = pd.Timestamp(year=report_year, month=1, day=1, tz="UTC")
    # For current year, end at latest data; for past years, end at Dec 31
    if report_year < latest.year:
        ytd_end = pd.Timestamp(year=report_year, month=12, day=31, hour=23,
                               minute=59, second=59, tz="UTC")
    else:
        ytd_end = latest

    df_ytd = df[(df["Timestamp"] >= ytd_start) & (df["Timestamp"] <= ytd_end)].copy()
    if len(df_ytd) < 2:
        return {"error": f"Insufficient data for year {report_year}"}

    common = dict(
        line_voltage=line_voltage,
        power_factor=power_factor,
        price_per_kwh=price_per_kwh,
        carbon_kg_per_kwh=carbon_kg_per_kwh,
        voltage_map=voltage_map,
    )
    common_kpi = dict(**common, panel_cols=panel_cols)

    # --- Facility YTD KPIs ---
    facility_kpi = compute_kpi(df_ytd, **common_kpi)

    # --- Department YTD breakdown ---
    departments = compute_department_breakdown(df_ytd, department_panels, **common)
    total_cost = facility_kpi["total_cost"]
    total_kwh = facility_kpi["total_kwh"]
    for d in departments:
        d["pct_of_total_cost"] = round(
            d["total_cost"] / total_cost * 100, 1
        ) if total_cost > 0 else 0.0
        d["pct_of_total_kwh"] = round(
            d["total_kwh"] / total_kwh * 100, 1
        ) if total_kwh > 0 else 0.0

    # --- Monthly breakdown ---
    last_month = ytd_end.month
    monthly_breakdown = []
    for m in range(1, last_month + 1):
        m_start = pd.Timestamp(year=report_year, month=m, day=1, tz="UTC")
        if m < 12:
            m_end = pd.Timestamp(year=report_year, month=m + 1, day=1, tz="UTC")
        else:
            m_end = pd.Timestamp(year=report_year + 1, month=1, day=1, tz="UTC")
        df_month = df_ytd[
            (df_ytd["Timestamp"] >= m_start) & (df_ytd["Timestamp"] < m_end)
        ].copy()

        if len(df_month) < 2:
            monthly_breakdown.append({
                "month": _MONTH_ABBR[m],
                "month_num": m,
                "facility": {"kwh": 0, "cost": 0, "peak_kw": 0},
                "departments": [],
            })
            continue

        m_kpi = compute_kpi(df_month, **common_kpi)
        m_depts = compute_department_breakdown(df_month, department_panels, **common)
        m_total = m_kpi["total_cost"]
        for d in m_depts:
            d["pct"] = round(
                d["total_cost"] / m_total * 100, 1
            ) if m_total > 0 else 0.0

        monthly_breakdown.append({
            "month": _MONTH_ABBR[m],
            "month_num": m,
            "facility": {
                "kwh": m_kpi["total_kwh"],
                "cost": m_kpi["total_cost"],
                "peak_kw": m_kpi["peak_kw"],
            },
            "departments": m_depts,
        })

    # --- Prior year comparison (if data exists) ---
    prior_year_data = None
    prior_year = report_year - 1
    py_start = pd.Timestamp(year=prior_year, month=1, day=1, tz="UTC")
    # Same calendar window in prior year
    py_end_month = ytd_end.month
    py_end_day = min(ytd_end.day, 28)  # safe for Feb
    py_end = pd.Timestamp(
        year=prior_year, month=py_end_month, day=py_end_day,
        hour=23, minute=59, second=59, tz="UTC",
    )
    df_py = df[(df["Timestamp"] >= py_start) & (df["Timestamp"] <= py_end)].copy()
    if len(df_py) > 1:
        py_kpi = compute_kpi(df_py, **common_kpi)
        py_depts = compute_department_breakdown(df_py, department_panels, **common)
        py_by_name = {d["department"]: d for d in py_depts}
        for d in departments:
            prev = py_by_name.get(d["department"], {})
            prev_cost = prev.get("total_cost", 0)
            d["yoy_cost_delta"] = round(d["total_cost"] - prev_cost, 2)
            d["yoy_cost_pct"] = round(
                (d["total_cost"] - prev_cost) / prev_cost * 100, 1
            ) if prev_cost > 0 else 0.0
        prior_year_data = {
            "year": prior_year,
            "facility_kpi": py_kpi,
            "departments": py_depts,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {
            "start": ytd_start.isoformat(),
            "end": ytd_end.isoformat(),
            "year": report_year,
        },
        "facility_kpi": facility_kpi,
        "departments": departments,
        "monthly_breakdown": monthly_breakdown,
        "prior_year": prior_year_data,
        "config": {
            "price_per_kwh": price_per_kwh,
            "line_voltage": line_voltage,
            "power_factor": power_factor,
            "carbon_kg_per_kwh": carbon_kg_per_kwh,
        },
    }


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_executive_narrative(data: dict) -> str:
    """Build a 3-4 sentence executive summary from report data."""
    kpi = data["facility_kpi"]
    wow = data["wow"]
    vs_avg = data["vs_avg"]

    kwh = kpi["total_kwh"]
    cost = kpi["total_cost"]
    carbon_kg = kpi["total_carbon_kg"]
    wow_kwh_pct = wow["kwh"]["pct"]
    wow_cost_delta = wow["cost"]["value"]
    wow_carbon_delta = wow["carbon_kg"]["value"]
    avg_kwh_pct = vs_avg["kwh"]["pct"]

    # Sentence 1: direction + magnitude
    if wow_kwh_pct > 1:
        s1 = (f"Facility energy consumption increased {abs(wow_kwh_pct):.1f}% this week "
              f"to {kwh:,.0f} kWh, costing ${cost:,.2f}.")
    elif wow_kwh_pct < -1:
        s1 = (f"Facility energy consumption decreased {abs(wow_kwh_pct):.1f}% this week "
              f"to {kwh:,.0f} kWh, saving ${abs(wow_cost_delta):,.2f} versus last week.")
    else:
        s1 = (f"Facility energy consumption held steady this week "
              f"at {kwh:,.0f} kWh (${cost:,.2f}).")

    # Sentence 2: carbon
    if wow_carbon_delta > 0:
        s2 = (f"CO\u2082 emissions rose to {carbon_kg:,.0f} kg "
              f"({abs(wow_carbon_delta):,.0f} kg more than last week).")
    elif wow_carbon_delta < 0:
        s2 = (f"CO\u2082 emissions fell to {carbon_kg:,.0f} kg, "
              f"a reduction of {abs(wow_carbon_delta):,.0f} kg from last week.")
    else:
        s2 = f"CO\u2082 emissions were {carbon_kg:,.0f} kg, unchanged from last week."

    # Sentence 3: vs 4-week average
    if avg_kwh_pct > 5:
        s3 = (f"Compared to the 4-week rolling average, usage is "
              f"{abs(avg_kwh_pct):.1f}% above baseline — worth monitoring.")
    elif avg_kwh_pct < -5:
        s3 = (f"Usage is {abs(avg_kwh_pct):.1f}% below the 4-week rolling average, "
              f"reflecting an improving trend.")
    else:
        s3 = "Usage is in line with the 4-week rolling average."

    # Sentence 4: notable callout
    phantom_totals = data.get("phantom", {}).get("facility_totals", {})
    phantom_cost = phantom_totals.get("total_phantom_cost", 0)
    increases = data.get("increases", [])
    if increases:
        top = increases[0]
        s4 = (f"Notable: {top.get('display_name', top.get('panel_id', 'Unknown'))} "
              f"saw a {top.get('pct_change', 0):+.1f}% change in consumption this week.")
    elif phantom_cost > 100:
        s4 = (f"Facility-wide phantom draw represents an estimated "
              f"${phantom_cost:,.0f}/year in wasted energy.")
    else:
        s4 = ""

    parts = [s1, s2, s3]
    if s4:
        parts.append(s4)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_report_html(data: dict) -> str:
    """Render the weekly report as self-contained HTML for Outlook."""
    if "error" in data:
        return _render_error_html(data["error"])

    narrative = _build_executive_narrative(data)
    period = data["period"]
    kpi = data["facility_kpi"]
    wow = data["wow"]
    vs_avg = data["vs_avg"]
    four_week = data["four_week_avg"]

    start_dt = pd.Timestamp(period["start"])
    end_dt = pd.Timestamp(period["end"])
    date_range = f"{start_dt.strftime('%b %d')} \u2013 {end_dt.strftime('%b %d, %Y')}"

    # Arrow helper
    def arrow(val, invert=False):
        """Return colored arrow span. invert=True means lower is better."""
        if val > 0:
            color = "#e74c3c" if not invert else "#27ae60"
            return f'<span style="color:{color}">&#9650; +{val:.1f}%</span>'
        elif val < 0:
            color = "#27ae60" if not invert else "#e74c3c"
            return f'<span style="color:{color}">&#9660; {val:.1f}%</span>'
        return '<span style="color:#7f8c8d">\u2014 0%</span>'

    def fmt_kwh(v):
        if v >= 10000:
            return f"{v/1000:,.1f} MWh"
        return f"{v:,.0f} kWh"

    def fmt_dollars(v):
        return f"${v:,.2f}"

    def fmt_carbon(v):
        if v >= 1000:
            return f"{v/1000:,.2f} tCO\u2082"
        return f"{v:,.0f} kg CO\u2082"

    # --- Build sections ---
    kpi_cards = _render_kpi_cards(kpi, wow, vs_avg, four_week, arrow, fmt_kwh, fmt_dollars, fmt_carbon)
    dept_table = _render_department_table(data["departments"])
    runners_table = _render_high_runners_table(data["high_runners"])
    changes_html = _render_significant_changes(data["increases"], data["decreases"])
    carbon_section = _render_carbon_section(data, arrow)
    phantom_section = _render_phantom_section(data["phantom"])

    html = f"""\
<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Weekly Energy Report — {date_range}</title>
</head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:Calibri,Arial,sans-serif;color:#2c3e50;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7;">
<tr><td align="center" style="padding:24px 16px;">

<!-- Main container -->
<table role="presentation" width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;border:1px solid #e0e0e0;">

<!-- Header -->
<tr>
<td style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:28px 32px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td>
      <div style="font-size:24px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">Weekly Energy Report</div>
      <div style="font-size:14px;color:#8bd435;margin-top:6px;font-weight:600;">{date_range}</div>
    </td>
    <td align="right" style="vertical-align:top;">
      <div style="font-size:11px;color:#a0a8b4;">EMSlite</div>
    </td>
  </tr>
  </table>
</td>
</tr>

<!-- Executive narrative -->
<tr>
<td style="padding:24px 32px 16px;">
  <div style="background:#f0f7e6;border-left:4px solid #8bd435;padding:16px 20px;border-radius:4px;">
    <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Executive Summary</div>
    <div style="font-size:14px;color:#34495e;line-height:1.6;">{narrative}</div>
  </div>
</td>
</tr>

<!-- KPI Scorecards -->
<tr>
<td style="padding:8px 32px 16px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Facility Performance</div>
  {kpi_cards}
</td>
</tr>

<!-- Department Breakdown -->
<tr>
<td style="padding:16px 32px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Department Breakdown</div>
  {dept_table}
</td>
</tr>

<!-- High Runners -->
<tr>
<td style="padding:16px 32px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Top 10 Energy Consumers</div>
  {runners_table}
</td>
</tr>

<!-- Significant Changes -->
<tr>
<td style="padding:16px 32px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Significant Changes (Week-over-Week)</div>
  {changes_html}
</td>
</tr>

<!-- Carbon & Sustainability -->
<tr>
<td style="padding:16px 32px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Carbon &amp; Sustainability</div>
  {carbon_section}
</td>
</tr>

<!-- Phantom Draw Insights -->
<tr>
<td style="padding:16px 32px;">
  <div style="font-size:13px;font-weight:700;color:#2c3e50;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;">Phantom Draw Insights</div>
  {phantom_section}
</td>
</tr>

<!-- Footer -->
<tr>
<td style="background:#f8f9fa;padding:20px 32px;border-top:1px solid #e0e0e0;">
  <div style="font-size:11px;color:#95a5a6;text-align:center;">
    Generated by EMSlite &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    &middot; Report covers {date_range}
  </div>
</td>
</tr>

</table>
<!-- /Main container -->

</td></tr>
</table>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_kpi_cards(kpi, wow, vs_avg, four_week, arrow, fmt_kwh, fmt_dollars, fmt_carbon):
    """Render 2x3 KPI scorecard grid."""
    cards = [
        ("Total Energy", fmt_kwh(kpi["total_kwh"]), arrow(wow["kwh"]["pct"]),
         f"4-wk avg: {fmt_kwh(four_week['weekly_kwh'])}"),
        ("Total Cost", fmt_dollars(kpi["total_cost"]), arrow(wow["cost"]["pct"]),
         f"4-wk avg: {fmt_dollars(four_week['weekly_cost'])}"),
        ("CO\u2082 Emissions", fmt_carbon(kpi["total_carbon_kg"]),
         arrow(wow["carbon_kg"]["pct"], invert=True),
         f"4-wk avg: {fmt_carbon(four_week['weekly_carbon_kg'])}"),
        ("Peak Demand", f"{kpi['peak_kw']:,.1f} kW", arrow(wow["peak_kw"]["pct"]), ""),
        ("Average Load", f"{kpi['avg_kw']:,.1f} kW", arrow(wow["avg_kw"]["pct"]), ""),
        ("Load Factor", f"{kpi['load_factor']:.1f}%", arrow(wow["load_factor"]["pct"]), ""),
    ]

    rows_html = ""
    for i in range(0, len(cards), 3):
        cells = ""
        for label, value, trend, sub in cards[i:i+3]:
            cells += f"""\
<td width="33%" style="padding:8px;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background:#f8f9fa;border-radius:6px;border:1px solid #eee;">
  <tr><td style="padding:12px 14px;">
    <div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.3px;">{label}</div>
    <div style="font-size:20px;font-weight:700;color:#2c3e50;margin:4px 0;">{value}</div>
    <div style="font-size:12px;">{trend}</div>
    <div style="font-size:11px;color:#95a5a6;margin-top:2px;">{sub}</div>
  </td></tr>
  </table>
</td>"""
        rows_html += f"<tr>{cells}</tr>"

    return f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{rows_html}</table>'


def _render_department_table(departments):
    """Render department breakdown table."""
    if not departments:
        return '<div style="font-size:13px;color:#95a5a6;">No departments configured.</div>'

    header = """\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="border-collapse:collapse;font-size:13px;">
<tr style="background:#f8f9fa;">
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;">Department</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">kWh</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Cost</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">CO&#8322; (kg)</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Peak kW</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">WoW Change</td>
</tr>"""

    rows = ""
    for i, d in enumerate(departments):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fa"
        pct = d.get("wow_kwh_pct", 0)
        if pct > 1:
            change_html = f'<span style="color:#e74c3c;">&#9650; +{pct:.1f}%</span>'
        elif pct < -1:
            change_html = f'<span style="color:#27ae60;">&#9660; {pct:.1f}%</span>'
        else:
            change_html = '<span style="color:#7f8c8d;">\u2014</span>'

        rows += f"""\
<tr style="background:{bg};">
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;">{d['department']}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{d['total_kwh']:,.0f}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">${d['total_cost']:,.2f}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{d['total_carbon_kg']:,.0f}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{d['peak_kw']:,.1f}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{change_html}</td>
</tr>"""

    return header + rows + "</table>"


def _render_high_runners_table(runners):
    """Render top energy consumers table."""
    if not runners:
        return '<div style="font-size:13px;color:#95a5a6;">No data available.</div>'

    header = """\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="border-collapse:collapse;font-size:13px;">
<tr style="background:#f8f9fa;">
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;width:30px;">#</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;">Device</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;">Department</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">kWh</td>
  <td style="padding:8px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Peak kW</td>
</tr>"""

    rows = ""
    for i, r in enumerate(runners):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fa"
        name = r.get("display_name", r["panel_id"])
        dept = r.get("department", "")
        rows += f"""\
<tr style="background:{bg};">
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;font-weight:700;color:#3498db;">{i+1}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;">{name}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;color:#7f8c8d;">{dept}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{r['total_kwh']:,.0f}</td>
  <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{r['peak_kw']:,.1f}</td>
</tr>"""

    return header + rows + "</table>"


def _render_significant_changes(increases, decreases):
    """Render significant changes section with increases and decreases."""
    if not increases and not decreases:
        return '<div style="font-size:13px;color:#95a5a6;">No significant changes (&gt;10%) this week.</div>'

    html = ""

    if increases:
        html += '<div style="font-size:12px;font-weight:700;color:#e74c3c;margin-bottom:8px;">Biggest Increases</div>'
        html += _render_changes_table(increases, "#e74c3c")

    if decreases:
        if increases:
            html += '<div style="height:16px;"></div>'
        html += '<div style="font-size:12px;font-weight:700;color:#27ae60;margin-bottom:8px;">Biggest Decreases</div>'
        html += _render_changes_table(decreases, "#27ae60")

    return html


def _render_changes_table(panels, color):
    """Render a changes sub-table."""
    header = f"""\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="border-collapse:collapse;font-size:13px;border-left:3px solid {color};">
<tr style="background:#f8f9fa;">
  <td style="padding:6px 12px;font-weight:700;border-bottom:1px solid #e0e0e0;">Device</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:1px solid #e0e0e0;text-align:right;">This Week</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:1px solid #e0e0e0;text-align:right;">Last Week</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:1px solid #e0e0e0;text-align:right;">Change</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:1px solid #e0e0e0;text-align:right;">Cost Impact</td>
</tr>"""

    rows = ""
    for p in panels:
        name = p.get("display_name", p.get("panel_id", ""))
        pct = p.get("pct_change", 0)
        rows += f"""\
<tr>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;">{name}</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{p.get('recent_kwh', 0):,.0f} kWh</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{p.get('prior_kwh', 0):,.0f} kWh</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;color:{color};font-weight:700;">{pct:+.1f}%</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">${p.get('cost_change', 0):+,.2f}</td>
</tr>"""

    return header + rows + "</table>"


def _render_carbon_section(data, arrow):
    """Render carbon & sustainability section."""
    kpi = data["facility_kpi"]
    prior = data["prior_kpi"]
    four_week = data["four_week_avg"]
    wow = data["wow"]

    carbon_this = kpi["total_carbon_kg"]
    carbon_prior = prior["total_carbon_kg"]
    carbon_delta = wow["carbon_kg"]["value"]
    carbon_4wk = four_week["weekly_carbon_kg"]

    # Equivalents: 1 tree absorbs ~22 kg CO2/year, avg home uses ~900 kWh/month
    trees_equivalent = abs(carbon_delta) / 22.0 * 52 if carbon_delta != 0 else 0

    equiv_text = ""
    if carbon_delta < -1:
        equiv_text = (f"This weekly reduction of {abs(carbon_delta):,.0f} kg CO\u2082 is equivalent to "
                      f"planting {trees_equivalent:,.0f} trees for a year.")
    elif carbon_delta > 1:
        equiv_text = (f"This weekly increase of {carbon_delta:,.0f} kg CO\u2082 would require "
                      f"{trees_equivalent:,.0f} additional trees for a year to offset.")

    # Department carbon contributions
    dept_carbon = ""
    departments = data.get("departments", [])
    if departments:
        dept_rows = ""
        total_carbon = sum(d["total_carbon_kg"] for d in departments)
        for d in departments:
            share = (d["total_carbon_kg"] / total_carbon * 100) if total_carbon else 0
            bar_width = max(1, int(share * 2.5))
            dept_rows += f"""\
<tr>
  <td style="padding:4px 12px;font-size:12px;">{d['department']}</td>
  <td style="padding:4px 12px;font-size:12px;text-align:right;">{d['total_carbon_kg']:,.0f} kg</td>
  <td style="padding:4px 12px;font-size:12px;text-align:right;">{share:.1f}%</td>
  <td style="padding:4px 6px;width:120px;">
    <div style="background:#3498db;height:10px;border-radius:3px;width:{bar_width}%;"></div>
  </td>
</tr>"""
        dept_carbon = f"""\
<div style="margin-top:12px;font-size:12px;font-weight:600;color:#2c3e50;">Department Carbon Contributions</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:6px;">
{dept_rows}
</table>"""

    return f"""\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">
<tr>
  <td width="33%" style="padding:8px;">
    <div style="background:#f0f7e6;border-radius:6px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;">This Week</div>
      <div style="font-size:18px;font-weight:700;color:#2c3e50;">{carbon_this:,.0f} kg</div>
      <div style="font-size:11px;color:#95a5a6;">{kpi['total_carbon_tonnes']:.2f} tonnes</div>
    </div>
  </td>
  <td width="33%" style="padding:8px;">
    <div style="background:#f8f9fa;border-radius:6px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;">Prior Week</div>
      <div style="font-size:18px;font-weight:700;color:#2c3e50;">{carbon_prior:,.0f} kg</div>
      <div style="font-size:12px;">{arrow(wow['carbon_kg']['pct'], invert=True)}</div>
    </div>
  </td>
  <td width="33%" style="padding:8px;">
    <div style="background:#f8f9fa;border-radius:6px;padding:14px;text-align:center;">
      <div style="font-size:11px;color:#7f8c8d;text-transform:uppercase;">4-Week Avg</div>
      <div style="font-size:18px;font-weight:700;color:#2c3e50;">{carbon_4wk:,.0f} kg</div>
    </div>
  </td>
</tr>
</table>
{f'<div style="font-size:13px;color:#34495e;margin-top:8px;padding:0 8px;">{equiv_text}</div>' if equiv_text else ''}
{dept_carbon}"""


def _render_phantom_section(phantom_data):
    """Render phantom draw insights section."""
    totals = phantom_data.get("facility_totals", {})
    offenders = phantom_data.get("top_offenders", [])

    total_cost = totals.get("total_phantom_cost", 0)
    total_kwh = totals.get("total_phantom_kwh", 0)
    total_carbon = totals.get("total_phantom_carbon_kg", 0)

    if not offenders and total_cost == 0:
        return '<div style="font-size:13px;color:#95a5a6;">Insufficient data for phantom draw analysis.</div>'

    # Facility totals banner
    banner = f"""\
<div style="background:#fff3e0;border-left:4px solid #f39c12;padding:14px 18px;border-radius:4px;margin-bottom:12px;">
  <div style="font-size:14px;color:#2c3e50;">
    After-hours phantom draw across the facility is estimated at
    <strong>${total_cost:,.0f}/year</strong> ({total_kwh:,.0f} kWh),
    producing <strong>{total_carbon:,.0f} kg CO&#8322;</strong> annually.
  </div>
</div>"""

    if not offenders:
        return banner

    # Top offenders table
    header = """\
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
       style="border-collapse:collapse;font-size:13px;">
<tr style="background:#f8f9fa;">
  <td style="padding:6px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;">Device</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Phantom kW</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Off-Shift %</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Annual Cost</td>
  <td style="padding:6px 12px;font-weight:700;border-bottom:2px solid #e0e0e0;text-align:right;">Annual CO&#8322;</td>
</tr>"""

    rows = ""
    top_savings = 0
    for o in offenders:
        name = o.get("display_name", o.get("panel_id", ""))
        top_savings += o.get("annual_phantom_cost", 0)
        rows += f"""\
<tr>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;">{name}</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{o.get('phantom_kw', 0):.2f}</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{o.get('off_shift_pct', 0):.0f}%</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;color:#e67e22;font-weight:700;">${o.get('annual_phantom_cost', 0):,.0f}</td>
  <td style="padding:6px 12px;border-bottom:1px solid #f0f0f0;text-align:right;">{o.get('annual_phantom_carbon_kg', 0):,.0f} kg</td>
</tr>"""

    cta = ""
    if top_savings > 0:
        cta = f"""\
<div style="font-size:13px;color:#34495e;margin-top:10px;padding:10px 12px;background:#f0f7e6;border-radius:4px;">
  Addressing phantom draw in the top {len(offenders)} offenders could save up to
  <strong>${top_savings:,.0f}/year</strong>.
  Simple actions: smart power strips, automated shutdown schedules, equipment timers.
</div>"""

    return banner + header + rows + "</table>" + cta


def _render_error_html(message: str) -> str:
    """Render a simple error page."""
    return f"""\
<!DOCTYPE html>
<html><head><title>Report Error</title></head>
<body style="font-family:Calibri,Arial,sans-serif;padding:40px;text-align:center;">
<h2 style="color:#e74c3c;">Report Generation Error</h2>
<p style="color:#7f8c8d;">{message}</p>
</body></html>"""


# ---------------------------------------------------------------------------
# YTD Cost Allocation Report — HTML rendering (slide-ready)
# ---------------------------------------------------------------------------

# Shared color palette for department chart bars
_DEPT_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#d35400", "#16a085",
]


def render_ytd_report_html(data: dict) -> str:
    """Render the YTD cost allocation report as slide-ready self-contained HTML."""
    if "error" in data:
        return _render_error_html(data["error"])

    period = data["period"]
    start_dt = pd.Timestamp(period["start"])
    end_dt = pd.Timestamp(period["end"])
    year = period["year"]

    title_slide = _render_ytd_title_slide(data, start_dt, end_dt, year)
    allocation_table = _render_ytd_allocation_table(
        data["departments"], data["facility_kpi"], data["config"],
    )
    share_bar = _render_ytd_share_bar(data["departments"])
    monthly_matrix = _render_ytd_monthly_matrix(
        data["monthly_breakdown"], data["departments"],
    )
    monthly_facility = _render_ytd_monthly_facility(data["monthly_breakdown"])
    yoy_section = _render_ytd_yoy_comparison(
        data["departments"], data["prior_year"],
    ) if data.get("prior_year") else ""
    methodology = _render_ytd_methodology(data["config"], period)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>YTD Cost Allocation Report — {year}</title>
<style>
  @media print {{
    .slide {{ page-break-inside: avoid; page-break-after: always; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:#e8eaed;font-family:Calibri,Arial,Helvetica,sans-serif;color:#2c3e50;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#e8eaed;">
<tr><td align="center" style="padding:24px 16px;">

<!-- Report container -->
<table role="presentation" width="960" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.1);">

{title_slide}
{allocation_table}
{share_bar}
{monthly_matrix}
{monthly_facility}
{yoy_section}
{methodology}

</table>
<!-- /Report container -->

</td></tr>
</table>
</body>
</html>"""
    return html


def _render_ytd_title_slide(data, start_dt, end_dt, year):
    """Slide 1: Title + facility KPI summary."""
    kpi = data["facility_kpi"]
    dept_count = len([d for d in data["departments"] if d["total_kwh"] > 0])
    date_range = f"January 1 \u2013 {end_dt.strftime('%B %d, %Y')}"

    def _fmt_kwh(v):
        if v >= 10_000:
            return f"{v / 1000:,.1f} MWh"
        return f"{v:,.0f} kWh"

    return f"""\
<!-- Slide 1: Title -->
<tr>
<td class="slide" style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:48px 56px;">
  <div style="font-size:32px;font-weight:700;color:#ffffff;letter-spacing:-0.5px;">
    Energy Cost Allocation Report
  </div>
  <div style="font-size:18px;color:#8bd435;margin-top:8px;font-weight:600;">
    Year to Date {year} &mdash; {date_range}
  </div>
  <div style="margin-top:32px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td width="33%" style="padding:0 8px 0 0;">
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:24px 20px;text-align:center;border:1px solid rgba(255,255,255,0.12);">
          <div style="font-size:12px;color:#a0a8b4;text-transform:uppercase;letter-spacing:1px;">Total Energy</div>
          <div style="font-size:36px;font-weight:700;color:#ffffff;margin:8px 0 4px;">{_fmt_kwh(kpi["total_kwh"])}</div>
        </div>
      </td>
      <td width="33%" style="padding:0 4px;">
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:24px 20px;text-align:center;border:1px solid rgba(255,255,255,0.12);">
          <div style="font-size:12px;color:#a0a8b4;text-transform:uppercase;letter-spacing:1px;">Total Cost</div>
          <div style="font-size:36px;font-weight:700;color:#8bd435;margin:8px 0 4px;">${kpi["total_cost"]:,.2f}</div>
        </div>
      </td>
      <td width="33%" style="padding:0 0 0 8px;">
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:24px 20px;text-align:center;border:1px solid rgba(255,255,255,0.12);">
          <div style="font-size:12px;color:#a0a8b4;text-transform:uppercase;letter-spacing:1px;">Departments</div>
          <div style="font-size:36px;font-weight:700;color:#ffffff;margin:8px 0 4px;">{dept_count}</div>
        </div>
      </td>
    </tr>
    </table>
  </div>
</td>
</tr>"""


def _render_ytd_allocation_table(departments, facility_kpi, config):
    """Slide 2: Department cost allocation table — the centerpiece."""
    if not departments:
        return ""

    price = config["price_per_kwh"]

    def _fmt_kwh(v):
        if v >= 10_000:
            return f"{v / 1000:,.1f} MWh"
        return f"{v:,.0f} kWh"

    header = """\
<!-- Slide 2: Cost Allocation Table -->
<tr>
<td class="slide" style="padding:40px 56px;">
  <div style="font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">Department Cost Allocation</div>
  <div style="font-size:13px;color:#7f8c8d;margin-bottom:20px;">Year-to-date energy cost by department, sorted by total cost</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;font-size:15px;">
  <tr style="background:#1a1a2e;">
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;border-radius:4px 0 0 0;">Department</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:right;">Energy</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:right;">Cost</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:center;">Share of Total</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:right;border-radius:0 4px 0 0;">Peak kW</td>
  </tr>"""

    rows = ""
    max_pct = max((d.get("pct_of_total_cost", 0) for d in departments), default=1)
    for i, d in enumerate(departments):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fb"
        pct = d.get("pct_of_total_cost", 0)
        bar_w = max(2, int(pct / max(max_pct, 1) * 100)) if pct > 0 else 0
        color = _DEPT_COLORS[i % len(_DEPT_COLORS)]
        rows += f"""\
  <tr style="background:{bg};">
    <td style="padding:10px 16px;border-bottom:1px solid #eee;font-weight:600;">{d['department']}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">{_fmt_kwh(d['total_kwh'])}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;font-weight:700;">${d['total_cost']:,.2f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:center;">
      <div style="display:inline-block;width:60%;text-align:left;">
        <div style="background:{color};height:14px;border-radius:3px;width:{bar_w}%;display:inline-block;vertical-align:middle;"></div>
        <span style="font-size:13px;color:#555;margin-left:6px;vertical-align:middle;">{pct:.1f}%</span>
      </div>
    </td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">{d['peak_kw']:,.1f}</td>
  </tr>"""

    # Totals row
    total_kwh = facility_kpi["total_kwh"]
    total_cost = facility_kpi["total_cost"]
    total_peak = facility_kpi["peak_kw"]
    rows += f"""\
  <tr style="background:#1a1a2e;">
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;border-radius:0 0 0 4px;">TOTAL</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:right;">{_fmt_kwh(total_kwh)}</td>
    <td style="padding:12px 16px;font-weight:700;color:#8bd435;text-align:right;">${total_cost:,.2f}</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:center;">100%</td>
    <td style="padding:12px 16px;font-weight:700;color:#ffffff;text-align:right;border-radius:0 0 4px 0;">{total_peak:,.1f}</td>
  </tr>"""

    footer = f"""\
  </table>
  <div style="font-size:11px;color:#95a5a6;margin-top:12px;">
    Rate: ${price:.4f}/kWh &middot; Costs calculated from metered current readings
  </div>
</td>
</tr>"""

    return header + rows + footer


def _render_ytd_share_bar(departments):
    """Slide 3: Horizontal stacked bar showing department share."""
    active = [d for d in departments if d.get("pct_of_total_cost", 0) > 0]
    if not active:
        return ""

    # Build stacked bar segments
    segments = ""
    for i, d in enumerate(active):
        pct = d["pct_of_total_cost"]
        color = _DEPT_COLORS[i % len(_DEPT_COLORS)]
        # Only show label if segment is wide enough
        label = f"{pct:.0f}%" if pct >= 5 else ""
        segments += (
            f'<div style="display:inline-block;width:{pct}%;background:{color};'
            f'height:48px;line-height:48px;text-align:center;color:#fff;font-size:13px;'
            f'font-weight:600;overflow:hidden;vertical-align:top;">{label}</div>'
        )

    # Legend
    legend_items = ""
    for i, d in enumerate(active):
        color = _DEPT_COLORS[i % len(_DEPT_COLORS)]
        legend_items += (
            f'<span style="display:inline-block;margin:4px 16px 4px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
            f'border-radius:2px;vertical-align:middle;margin-right:6px;"></span>'
            f'<span style="font-size:13px;color:#2c3e50;vertical-align:middle;">'
            f'{d["department"]} ({d["pct_of_total_cost"]:.1f}%)</span></span>'
        )

    return f"""\
<!-- Slide 3: Department Share -->
<tr>
<td class="slide" style="padding:40px 56px;">
  <div style="font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">Cost Share by Department</div>
  <div style="font-size:13px;color:#7f8c8d;margin-bottom:20px;">Proportional share of total facility energy cost</div>
  <div style="background:#e0e0e0;border-radius:6px;overflow:hidden;font-size:0;line-height:0;">
    {segments}
  </div>
  <div style="margin-top:16px;line-height:2;">
    {legend_items}
  </div>
</td>
</tr>"""


def _render_ytd_monthly_matrix(monthly_breakdown, departments):
    """Slide 4: Monthly cost matrix — departments as rows, months as columns."""
    if not monthly_breakdown or not departments:
        return ""

    active_depts = [d["department"] for d in departments if d["total_kwh"] > 0]
    if not active_depts:
        return ""

    months = monthly_breakdown

    # Header row with month abbreviations
    month_headers = ""
    for m in months:
        month_headers += (
            f'<td style="padding:8px 6px;font-weight:700;color:#ffffff;'
            f'text-align:right;font-size:13px;">{m["month"]}</td>'
        )

    # Build lookup: {dept_name: {month_num: cost}}
    dept_monthly: dict[str, dict[int, float]] = {}
    all_costs: list[float] = []
    for m in months:
        for d in m["departments"]:
            dept_monthly.setdefault(d["department"], {})[m["month_num"]] = d["total_cost"]
            if d["total_cost"] > 0:
                all_costs.append(d["total_cost"])

    max_cost = max(all_costs) if all_costs else 1

    # Data rows
    data_rows = ""
    for i, dept_name in enumerate(active_depts):
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fb"
        cells = ""
        ytd_total = 0.0
        for m in months:
            cost = dept_monthly.get(dept_name, {}).get(m["month_num"], 0)
            ytd_total += cost
            # Heat-map intensity
            intensity = min(cost / max_cost, 1.0) if max_cost > 0 else 0
            heat_bg = f"rgba(52, 152, 219, {intensity * 0.25:.2f})" if cost > 0 else "transparent"
            cells += (
                f'<td style="padding:8px 6px;text-align:right;font-size:13px;'
                f'border-bottom:1px solid #eee;background:{heat_bg};">'
                f'${cost:,.0f}</td>'
            )
        # YTD total column
        cells += (
            f'<td style="padding:8px 10px;text-align:right;font-size:13px;'
            f'font-weight:700;border-bottom:1px solid #eee;background:#f0f4f8;">'
            f'${ytd_total:,.0f}</td>'
        )
        data_rows += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:8px 12px;font-weight:600;font-size:13px;'
            f'border-bottom:1px solid #eee;white-space:nowrap;">{dept_name}</td>'
            f'{cells}</tr>'
        )

    # Facility total row
    facility_cells = ""
    running = 0.0
    for m in months:
        c = m["facility"]["cost"]
        running += c
        facility_cells += (
            f'<td style="padding:8px 6px;text-align:right;font-size:13px;'
            f'font-weight:700;color:#ffffff;">${c:,.0f}</td>'
        )
    facility_cells += (
        f'<td style="padding:8px 10px;text-align:right;font-size:14px;'
        f'font-weight:700;color:#8bd435;">${running:,.0f}</td>'
    )

    return f"""\
<!-- Slide 4: Monthly Cost Matrix -->
<tr>
<td class="slide" style="padding:40px 56px;">
  <div style="font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">Monthly Cost by Department</div>
  <div style="font-size:13px;color:#7f8c8d;margin-bottom:20px;">Department charges per month with heat-map shading</div>
  <div style="overflow-x:auto;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;min-width:700px;">
  <tr style="background:#1a1a2e;">
    <td style="padding:8px 12px;font-weight:700;color:#ffffff;font-size:13px;">Department</td>
    {month_headers}
    <td style="padding:8px 10px;font-weight:700;color:#8bd435;text-align:right;font-size:13px;">YTD Total</td>
  </tr>
  {data_rows}
  <tr style="background:#1a1a2e;">
    <td style="padding:8px 12px;font-weight:700;color:#ffffff;font-size:13px;">FACILITY</td>
    {facility_cells}
  </tr>
  </table>
  </div>
</td>
</tr>"""


def _render_ytd_monthly_facility(monthly_breakdown):
    """Slide 5: Month-by-month facility totals with running cumulative."""
    if not monthly_breakdown:
        return ""

    def _fmt_kwh(v):
        if v >= 10_000:
            return f"{v / 1000:,.1f} MWh"
        return f"{v:,.0f} kWh"

    rows = ""
    cum_kwh = 0.0
    cum_cost = 0.0
    for m in monthly_breakdown:
        fac = m["facility"]
        cum_kwh += fac["kwh"]
        cum_cost += fac["cost"]
        rows += f"""\
  <tr>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;font-weight:600;">{m["month"]}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">{_fmt_kwh(fac["kwh"])}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">${fac["cost"]:,.2f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">{fac["peak_kw"]:,.1f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;color:#7f8c8d;">{_fmt_kwh(cum_kwh)}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;font-weight:700;color:#1a1a2e;">${cum_cost:,.2f}</td>
  </tr>"""

    return f"""\
<!-- Slide 5: Monthly Facility Trend -->
<tr>
<td class="slide" style="padding:40px 56px;">
  <div style="font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">Monthly Facility Summary</div>
  <div style="font-size:13px;color:#7f8c8d;margin-bottom:20px;">Month-by-month facility totals with cumulative year-to-date</div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;font-size:15px;">
  <tr style="background:#1a1a2e;">
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;border-radius:4px 0 0 0;">Month</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Energy</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Cost</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Peak kW</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Cumul. Energy</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;border-radius:0 4px 0 0;">Cumul. Cost</td>
  </tr>
  {rows}
  </table>
</td>
</tr>"""


def _render_ytd_yoy_comparison(departments, prior_year):
    """Slide 6: Year-over-year comparison table."""
    if not prior_year:
        return ""

    py_year = prior_year["year"]
    cy_year = py_year + 1
    py_by_name = {d["department"]: d for d in prior_year["departments"]}

    def _arrow(val):
        if val > 1:
            return f'<span style="color:#e74c3c;">&#9650; +{val:.1f}%</span>'
        elif val < -1:
            return f'<span style="color:#27ae60;">&#9660; {val:.1f}%</span>'
        return '<span style="color:#7f8c8d;">&mdash; 0%</span>'

    rows = ""
    for i, d in enumerate(departments):
        if d["total_kwh"] == 0:
            continue
        bg = "#ffffff" if i % 2 == 0 else "#f8f9fb"
        prev = py_by_name.get(d["department"], {})
        prev_cost = prev.get("total_cost", 0)
        delta = d.get("yoy_cost_delta", d["total_cost"] - prev_cost)
        pct = d.get("yoy_cost_pct", 0)
        rows += f"""\
  <tr style="background:{bg};">
    <td style="padding:10px 16px;border-bottom:1px solid #eee;font-weight:600;">{d["department"]}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;font-weight:700;">${d["total_cost"]:,.2f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">${prev_cost:,.2f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;font-weight:700;">${delta:+,.2f}</td>
    <td style="padding:10px 16px;border-bottom:1px solid #eee;text-align:right;">{_arrow(pct)}</td>
  </tr>"""

    # Facility totals
    cy_total = sum(d["total_cost"] for d in departments)
    py_total = sum(d["total_cost"] for d in prior_year["departments"])
    fac_delta = cy_total - py_total
    fac_pct = (fac_delta / py_total * 100) if py_total > 0 else 0

    return f"""\
<!-- Slide 6: Year-over-Year -->
<tr>
<td class="slide" style="padding:40px 56px;">
  <div style="font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;">Year-over-Year Comparison</div>
  <div style="font-size:13px;color:#7f8c8d;margin-bottom:20px;">
    {cy_year} YTD vs same period {py_year} &mdash; facility costs are
    <strong>{_arrow(fac_pct)}</strong> (${fac_delta:+,.2f})
  </div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="border-collapse:collapse;font-size:15px;">
  <tr style="background:#1a1a2e;">
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;">Department</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">{cy_year} Cost</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">{py_year} Cost</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Delta ($)</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">Change</td>
  </tr>
  {rows}
  <tr style="background:#1a1a2e;">
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;">TOTAL</td>
    <td style="padding:10px 16px;font-weight:700;color:#8bd435;text-align:right;">${cy_total:,.2f}</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">${py_total:,.2f}</td>
    <td style="padding:10px 16px;font-weight:700;color:#ffffff;text-align:right;">${fac_delta:+,.2f}</td>
    <td style="padding:10px 16px;font-weight:700;text-align:right;">{_arrow(fac_pct)}</td>
  </tr>
  </table>
</td>
</tr>"""


def _render_ytd_methodology(config, period):
    """Slide 7: Methodology notes and footnotes."""
    start_dt = pd.Timestamp(period["start"])
    end_dt = pd.Timestamp(period["end"])
    date_range = f"{start_dt.strftime('%B %d, %Y')} \u2013 {end_dt.strftime('%B %d, %Y')}"

    return f"""\
<!-- Slide 7: Methodology -->
<tr>
<td style="padding:32px 56px;background:#f8f9fa;border-top:2px solid #e0e0e0;">
  <div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-bottom:12px;">Methodology &amp; Notes</div>
  <table role="presentation" cellpadding="0" cellspacing="0" style="font-size:13px;color:#555;">
  <tr>
    <td style="padding:4px 24px 4px 0;font-weight:600;color:#2c3e50;">Electricity Rate</td>
    <td style="padding:4px 0;">${config["price_per_kwh"]:.4f} / kWh</td>
  </tr>
  <tr>
    <td style="padding:4px 24px 4px 0;font-weight:600;color:#2c3e50;">Line Voltage</td>
    <td style="padding:4px 0;">{config["line_voltage"]:.0f} V</td>
  </tr>
  <tr>
    <td style="padding:4px 24px 4px 0;font-weight:600;color:#2c3e50;">Power Factor</td>
    <td style="padding:4px 0;">{config["power_factor"]:.2f}</td>
  </tr>
  <tr>
    <td style="padding:4px 24px 4px 0;font-weight:600;color:#2c3e50;">Carbon Factor</td>
    <td style="padding:4px 0;">{config["carbon_kg_per_kwh"]:.2f} kg CO&#8322; / kWh</td>
  </tr>
  <tr>
    <td style="padding:4px 24px 4px 0;font-weight:600;color:#2c3e50;">Data Period</td>
    <td style="padding:4px 0;">{date_range}</td>
  </tr>
  </table>
  <div style="font-size:11px;color:#95a5a6;margin-top:16px;">
    Costs are calculated from metered current readings, not utility bills.
    Department allocation is based on panel assignments at time of report generation.
  </div>
  <div style="font-size:11px;color:#95a5a6;margin-top:8px;">
    Generated by EMSlite &middot; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  </div>
</td>
</tr>"""
