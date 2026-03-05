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
