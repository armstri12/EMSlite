#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
product_test_dashboard.py

Generate a standalone HTML dashboard that cross-references product test data
with EMS panel energy usage to compute per-test energy cost.

Expected spreadsheet columns (CSV or XLSX):
  - Product SN        : serial number of the pump / product
  - Test Start Date   : date-time the test began
  - Total Test Time   : test duration in minutes
  - Pass/Fail         : result string ("Pass" or "Fail")

The dashboard highlights products tested more than once and calculates
energy cost for each test using two configurable panels from the EMS data.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Shared config helpers (mirrors visualize_meter_data.py)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "input_file": "RawPanelUsageHistory_UPDATED.csv",
    "output_dir": "visualizations",
    "line_voltage": 480.0,
    "power_factor": 1.0,
    "price_per_kwh": 0.25,
    "total_amps_sources": None,
    "utility_meters": [],
    "combo_columns": {},
    "rolling_window": "1h",
    "dashboard_logo_path": "",
    "visualizations": {},
    "product_test": {
        "panels": [],
        "output": "product_test_dashboard.html",
    },
}

CONFIG: dict = deepcopy(DEFAULT_CONFIG)


def merge_config(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def strip_json_noise(raw_text: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", raw_text, flags=re.DOTALL)
    no_line = re.sub(r"//.*?$", "", no_block, flags=re.MULTILINE)
    no_trailing = re.sub(r",(\s*[}\]])", r"\1", no_line)
    return no_trailing


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        sanitized = strip_json_noise(raw_text)
        loaded = json.loads(sanitized)
    return merge_config(DEFAULT_CONFIG, loaded)


def amps_to_kw(amps: pd.Series, config: dict) -> pd.Series:
    return amps * (config["line_voltage"] * 3 ** 0.5 * config["power_factor"]) / 1000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_energy_data(path: Path) -> pd.DataFrame:
    """Load the wide-format EMS CSV (Timestamp + meter columns in amps)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" not in df.columns:
        raise ValueError("EMS CSV must include a 'Timestamp' column.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    for col in df.columns:
        if col != "Timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


SPREADSHEET_COLUMN_MAP = {
    "product sn": "Product SN",
    "product_sn": "Product SN",
    "productsn": "Product SN",
    "sn": "Product SN",
    "serial": "Product SN",
    "serial number": "Product SN",
    "serial_number": "Product SN",
    "test start date": "Test Start Date",
    "test_start_date": "Test Start Date",
    "teststartdate": "Test Start Date",
    "start date": "Test Start Date",
    "start_date": "Test Start Date",
    "total test time": "Total Test Time",
    "total_test_time": "Total Test Time",
    "totaltesttime": "Total Test Time",
    "test time": "Total Test Time",
    "test_time": "Total Test Time",
    "duration": "Total Test Time",
    "pass/fail": "Pass/Fail",
    "pass_fail": "Pass/Fail",
    "passfail": "Pass/Fail",
    "result": "Pass/Fail",
}

REQUIRED_COLUMNS = {"Product SN", "Test Start Date", "Total Test Time", "Pass/Fail"}


def load_test_data(path: Path) -> pd.DataFrame:
    """Load product test spreadsheet (CSV or XLSX)."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    # Normalise column names
    rename_map = {}
    for col in df.columns:
        normalised = col.lower().strip()
        if normalised in SPREADSHEET_COLUMN_MAP:
            rename_map[col] = SPREADSHEET_COLUMN_MAP[normalised]
    df = df.rename(columns=rename_map)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Test spreadsheet is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["Test Start Date"] = pd.to_datetime(df["Test Start Date"], errors="coerce", utc=True)
    df["Total Test Time"] = pd.to_numeric(df["Total Test Time"], errors="coerce").fillna(0)
    df["Product SN"] = df["Product SN"].astype(str).str.strip()
    df["Pass/Fail"] = df["Pass/Fail"].astype(str).str.strip()
    df = df.dropna(subset=["Test Start Date"]).sort_values("Test Start Date").reset_index(drop=True)
    df["Test End Date"] = df["Test Start Date"] + pd.to_timedelta(df["Total Test Time"], unit="m")
    return df


# ---------------------------------------------------------------------------
# Energy cost computation
# ---------------------------------------------------------------------------

def compute_test_energy(
    test_row: pd.Series,
    energy_df: pd.DataFrame,
    panel_cols: list[str],
    config: dict,
) -> dict:
    """Compute kWh and cost for a single test using the configured panels."""
    start = test_row["Test Start Date"]
    end = test_row["Test End Date"]

    mask = (energy_df["Timestamp"] >= start) & (energy_df["Timestamp"] <= end)
    window = energy_df.loc[mask].sort_values("Timestamp")

    if window.empty or len(window) < 2:
        # Fallback: estimate from average power across full dataset
        duration_hours = test_row["Total Test Time"] / 60.0
        if panel_cols:
            avg_amps = energy_df[panel_cols].fillna(0).sum(axis=1).mean()
            avg_kw = float(amps_to_kw(pd.Series([avg_amps]), config).iloc[0])
        else:
            avg_kw = 0.0
        kwh = avg_kw * duration_hours
        cost = kwh * config["price_per_kwh"]
        return {"kw_avg": avg_kw, "kwh": kwh, "cost": cost, "method": "estimated"}

    # Sum the two panel columns (in amps), convert to kW, integrate over time
    if panel_cols:
        total_amps = window[panel_cols].fillna(0).sum(axis=1)
        total_kw = amps_to_kw(total_amps, config)
    else:
        total_kw = pd.Series([0.0] * len(window))

    # Time-weighted integration (trapezoid-style)
    deltas = window["Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    kwh = float((total_kw * deltas).sum())
    avg_kw = float(total_kw.mean())
    cost = kwh * config["price_per_kwh"]
    return {"kw_avg": avg_kw, "kwh": kwh, "cost": cost, "method": "measured"}


def enrich_test_data(
    test_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Add energy metrics to each test row."""
    pt_config = config.get("product_test", {})
    panel_cols = pt_config.get("panels", [])

    # Validate panels exist
    available = [c for c in energy_df.columns if c != "Timestamp"]
    valid_panels = [p for p in panel_cols if p in available]
    if len(valid_panels) != len(panel_cols):
        missing = set(panel_cols) - set(valid_panels)
        print(f"Warning: product_test panels not found in EMS data: {missing}")
    panel_cols = valid_panels

    results = []
    for _, row in test_df.iterrows():
        energy = compute_test_energy(row, energy_df, panel_cols, config)
        results.append(energy)

    energy_results = pd.DataFrame(results)
    test_df = test_df.copy()
    test_df["Avg kW"] = energy_results["kw_avg"].values
    test_df["Energy (kWh)"] = energy_results["kwh"].values
    test_df["Energy Cost ($)"] = energy_results["cost"].values
    test_df["Calc Method"] = energy_results["method"].values

    # Flag products tested more than once
    sn_counts = test_df["Product SN"].value_counts()
    multi_test_sns = set(sn_counts[sn_counts > 1].index)
    test_df["Retest"] = test_df["Product SN"].isin(multi_test_sns)
    test_df["Test Count"] = test_df["Product SN"].map(sn_counts)

    return test_df


# ---------------------------------------------------------------------------
# Dashboard HTML generation
# ---------------------------------------------------------------------------

def build_product_test_dashboard(
    test_df: pd.DataFrame,
    config: dict,
    output_path: Path,
) -> Path:
    """Generate a standalone HTML product test dashboard."""
    price_per_kwh = float(config["price_per_kwh"])
    logo_path = config.get("dashboard_logo_path", "")
    pt_config = config.get("product_test", {})
    panel_names = pt_config.get("panels", [])

    # Prepare data for JS
    records = []
    for _, row in test_df.iterrows():
        records.append({
            "sn": str(row["Product SN"]),
            "start": row["Test Start Date"].isoformat() if pd.notna(row["Test Start Date"]) else "",
            "end": row["Test End Date"].isoformat() if pd.notna(row["Test End Date"]) else "",
            "duration_min": float(row["Total Test Time"]),
            "result": str(row["Pass/Fail"]),
            "avg_kw": round(float(row["Avg kW"]), 4),
            "kwh": round(float(row["Energy (kWh)"]), 4),
            "cost": round(float(row["Energy Cost ($)"]), 4),
            "method": str(row["Calc Method"]),
            "retest": bool(row["Retest"]),
            "test_count": int(row["Test Count"]),
        })

    # Summary statistics
    total_tests = len(records)
    total_pass = sum(1 for r in records if r["result"].lower() == "pass")
    total_fail = total_tests - total_pass
    total_kwh = sum(r["kwh"] for r in records)
    total_cost = sum(r["cost"] for r in records)
    retest_count = sum(1 for r in records if r["retest"])
    unique_sns = len(set(r["sn"] for r in records))
    multi_test_sns = len(set(r["sn"] for r in records if r["retest"]))

    summary = {
        "total_tests": total_tests,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "pass_rate": round(total_pass / total_tests * 100, 1) if total_tests else 0,
        "total_kwh": round(total_kwh, 2),
        "total_cost": round(total_cost, 2),
        "retest_count": retest_count,
        "unique_sns": unique_sns,
        "multi_test_sns": multi_test_sns,
        "price_per_kwh": price_per_kwh,
        "panel_names": panel_names,
    }

    data_payload = json.dumps({"records": records, "summary": summary})

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Product Test Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
      :root {{
        --bg: #f8f9fa;
        --card: #ffffff;
        --ink: #2d363a;
        --ink-strong: #1a1f22;
        --muted: #6c757d;
        --accent: #c4262e;
        --accent-hover: #a61f26;
        --accent-soft: rgba(196, 38, 46, 0.08);
        --outline: rgba(45, 54, 58, 0.08);
        --border: rgba(45, 54, 58, 0.12);
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
        --pass-bg: #d4edda;
        --pass-text: #155724;
        --fail-bg: #f8d7da;
        --fail-text: #721c24;
        --retest-bg: #fff3cd;
        --retest-border: #ffc107;
      }}
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: var(--bg);
        color: var(--ink);
        line-height: 1.6;
        padding-top: 100px;
      }}
      .header-wrapper {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: var(--card);
        border-bottom: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        z-index: 1000;
      }}
      .top-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 20px 48px;
        max-width: 1920px;
        margin: 0 auto;
      }}
      .brand {{
        display: flex;
        align-items: center;
        gap: 20px;
      }}
      .logo-wrapper {{
        width: 56px;
        height: 56px;
        border-radius: 12px;
        background: var(--accent-soft);
        border: 2px solid var(--outline);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        flex-shrink: 0;
      }}
      .logo-wrapper img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
      }}
      .logo-placeholder {{
        font-size: 10px;
        font-weight: 600;
        color: var(--accent);
        text-align: center;
        padding: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }}
      .brand-text {{
        display: flex;
        flex-direction: column;
      }}
      .title {{
        font-size: 28px;
        font-weight: 700;
        color: var(--ink-strong);
        letter-spacing: -0.5px;
      }}
      .subtitle {{
        margin-top: 2px;
        color: var(--muted);
        font-size: 13px;
        font-weight: 500;
      }}
      .nav-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .nav-pill {{
        padding: 8px 18px;
        border-radius: 8px;
        background: transparent;
        color: var(--ink);
        font-size: 14px;
        font-weight: 600;
        text-decoration: none;
        border: 1px solid transparent;
        transition: all 0.2s ease;
      }}
      .nav-pill:hover {{
        background: var(--accent-soft);
        border-color: var(--outline);
        color: var(--accent);
      }}
      .container {{
        max-width: 1920px;
        margin: 0 auto;
        padding: 0 48px;
      }}
      .section {{
        margin-bottom: 48px;
      }}
      .section-header {{
        margin-bottom: 24px;
      }}
      .section-title {{
        font-size: 20px;
        font-weight: 700;
        color: var(--ink-strong);
        letter-spacing: -0.3px;
        margin-bottom: 4px;
      }}
      .section-description {{
        font-size: 14px;
        color: var(--muted);
      }}
      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-bottom: 24px;
      }}
      .stat-card {{
        background: var(--card);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
        transition: all 0.2s ease;
      }}
      .stat-card:hover {{
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
      }}
      .stat-card.accent {{
        border: 2px solid var(--accent);
        background: linear-gradient(135deg, var(--card), var(--accent-soft));
      }}
      .stat-card.warning {{
        border: 2px solid var(--retest-border);
        background: linear-gradient(135deg, var(--card), var(--retest-bg));
      }}
      .stat-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--muted);
      }}
      .stat-value {{
        font-size: 32px;
        font-weight: 700;
        margin-top: 12px;
        color: var(--ink-strong);
        line-height: 1;
      }}
      .stat-hint {{
        margin-top: 8px;
        font-size: 12px;
        color: var(--muted);
        font-weight: 500;
      }}
      .charts-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 24px;
        margin-bottom: 24px;
      }}
      .chart-card {{
        background: var(--card);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
        transition: all 0.2s ease;
      }}
      .chart-card:hover {{
        box-shadow: var(--shadow-lg);
      }}
      .chart-card.full-width {{
        grid-column: 1 / -1;
      }}
      .chart-title {{
        font-size: 16px;
        font-weight: 700;
        color: var(--ink-strong);
        margin-bottom: 16px;
        letter-spacing: -0.2px;
      }}
      .chart {{
        min-height: 380px;
      }}

      /* --- Table --- */
      .table-wrapper {{
        overflow-x: auto;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md);
        background: var(--card);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }}
      thead th {{
        background: var(--ink);
        color: #ffffff;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        white-space: nowrap;
        position: sticky;
        top: 0;
        cursor: pointer;
        user-select: none;
      }}
      thead th:hover {{
        background: var(--ink-strong);
      }}
      thead th .sort-arrow {{
        margin-left: 4px;
        opacity: 0.5;
      }}
      thead th.sorted .sort-arrow {{
        opacity: 1;
      }}
      tbody tr {{
        border-bottom: 1px solid var(--border);
        transition: background 0.15s ease;
      }}
      tbody tr:hover {{
        background: var(--accent-soft);
      }}
      tbody tr.retest-row {{
        background: var(--retest-bg);
        border-left: 4px solid var(--retest-border);
      }}
      tbody tr.retest-row:hover {{
        background: #ffecb5;
      }}
      tbody td {{
        padding: 12px 16px;
        white-space: nowrap;
      }}
      .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }}
      .badge-pass {{
        background: var(--pass-bg);
        color: var(--pass-text);
      }}
      .badge-fail {{
        background: var(--fail-bg);
        color: var(--fail-text);
      }}
      .badge-retest {{
        background: var(--retest-bg);
        color: #856404;
        border: 1px solid var(--retest-border);
      }}
      .badge-estimated {{
        background: #e2e3e5;
        color: #383d41;
        font-size: 10px;
      }}

      /* --- Filters --- */
      .filters {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
        margin-bottom: 20px;
      }}
      .filters label {{
        font-size: 11px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }}
      .filters input, .filters select {{
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        color: var(--ink);
        background: var(--card);
        transition: all 0.2s ease;
      }}
      .filters input:focus, .filters select:focus {{
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-soft);
      }}
      .filters button {{
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        color: #ffffff;
        background: var(--accent);
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
      }}
      .filters button:hover {{
        background: var(--accent-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
      }}
      .filters button.secondary {{
        background: var(--ink);
        color: #ffffff;
      }}
      .filters button.secondary:hover {{
        background: var(--ink-strong);
      }}
      .section-anchor {{
        scroll-margin-top: 120px;
      }}
      .footer {{
        padding: 32px 48px;
        color: var(--muted);
        font-size: 12px;
        text-align: center;
        border-top: 1px solid var(--border);
        background: var(--card);
        margin-top: 48px;
      }}
      @media (max-width: 1200px) {{
        .charts-grid {{
          grid-template-columns: 1fr;
        }}
      }}
      @media (max-width: 768px) {{
        body {{
          padding-top: 120px;
        }}
        .container {{
          padding: 0 24px;
        }}
        .top-bar {{
          flex-direction: column;
          align-items: flex-start;
          padding: 16px 24px;
        }}
        .stats-grid {{
          grid-template-columns: 1fr;
        }}
        .footer {{
          padding: 24px;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="header-wrapper">
      <div class="top-bar">
        <div class="brand">
          <div class="logo-wrapper">
            <img id="logo-image" src="{logo_path}" alt="Logo" style="display: none;" />
            <div class="logo-placeholder" id="logo-placeholder">Logo</div>
          </div>
          <div class="brand-text">
            <h1 class="title">Product Test Dashboard</h1>
            <div class="subtitle">Test energy cost analysis &middot; ${price_per_kwh:.2f}/kWh &middot; Panels: {', '.join(panel_names) if panel_names else 'None configured'}</div>
          </div>
        </div>
        <div class="nav-pills">
          <a class="nav-pill" href="#kpi-section">Summary</a>
          <a class="nav-pill" href="#charts-section">Analytics</a>
          <a class="nav-pill" href="#table-section">Test Log</a>
        </div>
      </div>
    </div>

    <div class="container">
      <!-- KPI Section -->
      <div class="section section-anchor" id="kpi-section">
        <div class="section-header">
          <h2 class="section-title">Test Summary</h2>
          <div class="section-description">Overview of product test results and energy costs</div>
        </div>
        <div class="stats-grid" id="kpi-grid"></div>
      </div>

      <!-- Charts Section -->
      <div class="section section-anchor" id="charts-section">
        <div class="section-header">
          <h2 class="section-title">Test Analytics</h2>
          <div class="section-description">Visual analysis of test results, energy usage, and retests</div>
        </div>
        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-title">Energy Cost per Test</div>
            <div id="cost-per-test-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Pass / Fail Distribution</div>
            <div id="pass-fail-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Energy Usage Over Time</div>
            <div id="energy-timeline-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Retest Frequency by Product SN</div>
            <div id="retest-chart" class="chart"></div>
          </div>
          <div class="chart-card full-width">
            <div class="chart-title">Test Duration vs Energy Cost</div>
            <div id="duration-vs-cost-chart" class="chart"></div>
          </div>
        </div>
      </div>

      <!-- Table Section -->
      <div class="section section-anchor" id="table-section">
        <div class="section-header">
          <h2 class="section-title">Test Log</h2>
          <div class="section-description">Complete test records &mdash; rows highlighted in yellow indicate products tested more than once</div>
        </div>
        <div class="filters">
          <div>
            <label for="filter-sn">Search SN</label>
            <input type="text" id="filter-sn" placeholder="Filter by SN..." />
          </div>
          <div>
            <label for="filter-result">Result</label>
            <select id="filter-result">
              <option value="all">All</option>
              <option value="pass">Pass</option>
              <option value="fail">Fail</option>
            </select>
          </div>
          <div>
            <label for="filter-retest">Retest</label>
            <select id="filter-retest">
              <option value="all">All</option>
              <option value="yes">Retested Only</option>
              <option value="no">Single Test Only</option>
            </select>
          </div>
          <button id="reset-table-filters" class="secondary">Reset</button>
        </div>
        <div class="table-wrapper">
          <table id="test-table">
            <thead>
              <tr>
                <th data-col="sn">Product SN <span class="sort-arrow">&#9650;</span></th>
                <th data-col="start">Test Start <span class="sort-arrow">&#9650;</span></th>
                <th data-col="duration_min">Duration (min) <span class="sort-arrow">&#9650;</span></th>
                <th data-col="result">Result <span class="sort-arrow">&#9650;</span></th>
                <th data-col="avg_kw">Avg kW <span class="sort-arrow">&#9650;</span></th>
                <th data-col="kwh">Energy (kWh) <span class="sort-arrow">&#9650;</span></th>
                <th data-col="cost">Cost ($) <span class="sort-arrow">&#9650;</span></th>
                <th data-col="test_count">Tests <span class="sort-arrow">&#9650;</span></th>
                <th data-col="method">Method <span class="sort-arrow">&#9650;</span></th>
              </tr>
            </thead>
            <tbody id="test-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="footer">
      Product Test Dashboard &middot; Energy Management System
    </div>

    <script>
      const DATA = {data_payload};
      const records = DATA.records;
      const summary = DATA.summary;
      const logoPath = {json.dumps(logo_path)};

      const theme = {{
        ink: "#2d363a",
        inkStrong: "#1a1f22",
        muted: "#6c757d",
        card: "#ffffff",
        grid: "rgba(0, 0, 0, 0.08)",
        accent: "#c4262e",
        accentDark: "#2d363a",
        series: ["#c4262e", "#2d363a", "#495057", "#6c757d", "#adb5bd", "#dee2e6"]
      }};

      /* ---- Logo ---- */
      (function() {{
        const img = document.getElementById("logo-image");
        const ph = document.getElementById("logo-placeholder");
        if (logoPath) {{
          img.src = logoPath;
          img.style.display = "block";
          ph.style.display = "none";
        }}
      }})();

      /* ---- KPIs ---- */
      function renderKPIs() {{
        const grid = document.getElementById("kpi-grid");
        const cards = [
          {{ label: "Total Tests", value: summary.total_tests, hint: `${{summary.unique_sns}} unique products` }},
          {{ label: "Pass Rate", value: `${{summary.pass_rate}}%`, hint: `${{summary.total_pass}} pass / ${{summary.total_fail}} fail`, cls: "accent" }},
          {{ label: "Total Energy", value: `${{summary.total_kwh.toFixed(2)}} kWh`, hint: "All tests combined" }},
          {{ label: "Total Cost", value: `$${{summary.total_cost.toFixed(2)}}`, hint: `At $${{summary.price_per_kwh.toFixed(2)}}/kWh`, cls: "accent" }},
          {{ label: "Retested Products", value: summary.multi_test_sns, hint: `${{summary.retest_count}} total retest records`, cls: "warning" }},
        ];
        grid.innerHTML = cards.map(c =>
          `<div class="stat-card ${{c.cls || ''}}">
            <div class="stat-label">${{c.label}}</div>
            <div class="stat-value">${{c.value}}</div>
            <div class="stat-hint">${{c.hint}}</div>
          </div>`
        ).join("");
      }}

      /* ---- Charts ---- */
      const plotConfig = {{ displaylogo: false, responsive: true }};
      const layoutBase = {{
        margin: {{ t: 16, l: 60, r: 24, b: 50 }},
        paper_bgcolor: theme.card,
        plot_bgcolor: theme.card,
        font: {{ family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: theme.ink, size: 12 }},
        colorway: theme.series,
        hovermode: "closest",
        hoverlabel: {{ bgcolor: theme.inkStrong, font: {{ color: "#ffffff" }} }},
      }};

      function renderCharts() {{
        // Cost per test (bar)
        const sorted = [...records].sort((a, b) => b.cost - a.cost);
        const barColors = sorted.map(r => r.retest ? "#ffc107" : (r.result.toLowerCase() === "pass" ? "#28a745" : "#c4262e"));
        Plotly.newPlot("cost-per-test-chart", [{{
          x: sorted.map(r => r.sn),
          y: sorted.map(r => r.cost),
          type: "bar",
          marker: {{ color: barColors }},
          text: sorted.map(r => `$${{r.cost.toFixed(2)}}`),
          textposition: "outside",
          hovertemplate: "SN: %{{x}}<br>Cost: $%{{y:.2f}}<br>Result: %{{customdata}}<extra></extra>",
          customdata: sorted.map(r => r.result),
        }}], {{
          ...layoutBase,
          xaxis: {{ title: {{ text: "Product SN", font: {{ size: 12, weight: 600 }} }}, tickangle: -45, gridcolor: theme.grid }},
          yaxis: {{ title: {{ text: "Cost ($)", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid }},
          bargap: 0.2,
        }}, plotConfig);

        // Pass/fail pie
        Plotly.newPlot("pass-fail-chart", [{{
          values: [summary.total_pass, summary.total_fail],
          labels: ["Pass", "Fail"],
          type: "pie",
          marker: {{ colors: ["#28a745", "#c4262e"] }},
          textinfo: "label+percent+value",
          hole: 0.45,
        }}], {{
          ...layoutBase,
          margin: {{ t: 16, l: 24, r: 24, b: 24 }},
          showlegend: true,
          legend: {{ orientation: "h", y: -0.1 }},
        }}, plotConfig);

        // Energy timeline (scatter)
        const passRecords = records.filter(r => r.result.toLowerCase() === "pass");
        const failRecords = records.filter(r => r.result.toLowerCase() !== "pass");
        const traces = [];
        if (passRecords.length) {{
          traces.push({{
            x: passRecords.map(r => r.start),
            y: passRecords.map(r => r.kwh),
            mode: "markers",
            name: "Pass",
            marker: {{ color: "#28a745", size: 10, line: {{ color: "#ffffff", width: 1.5 }},
              symbol: passRecords.map(r => r.retest ? "diamond" : "circle") }},
            text: passRecords.map(r => `SN: ${{r.sn}}<br>kWh: ${{r.kwh.toFixed(3)}}<br>Cost: $${{r.cost.toFixed(2)}}${{r.retest ? "<br>RETEST" : ""}}`),
            hoverinfo: "text",
          }});
        }}
        if (failRecords.length) {{
          traces.push({{
            x: failRecords.map(r => r.start),
            y: failRecords.map(r => r.kwh),
            mode: "markers",
            name: "Fail",
            marker: {{ color: "#c4262e", size: 10, line: {{ color: "#ffffff", width: 1.5 }},
              symbol: failRecords.map(r => r.retest ? "diamond" : "circle") }},
            text: failRecords.map(r => `SN: ${{r.sn}}<br>kWh: ${{r.kwh.toFixed(3)}}<br>Cost: $${{r.cost.toFixed(2)}}${{r.retest ? "<br>RETEST" : ""}}`),
            hoverinfo: "text",
          }});
        }}
        Plotly.newPlot("energy-timeline-chart", traces, {{
          ...layoutBase,
          xaxis: {{ title: {{ text: "Test Date", font: {{ size: 12, weight: 600 }} }}, type: "date", gridcolor: theme.grid }},
          yaxis: {{ title: {{ text: "Energy (kWh)", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid }},
          legend: {{ orientation: "h", y: -0.15 }},
        }}, plotConfig);

        // Retest frequency (only SNs tested > 1)
        const retestSNs = {{}};
        records.forEach(r => {{ if (r.retest) retestSNs[r.sn] = r.test_count; }});
        const retestEntries = Object.entries(retestSNs).sort((a, b) => b[1] - a[1]);
        if (retestEntries.length) {{
          Plotly.newPlot("retest-chart", [{{
            x: retestEntries.map(e => e[0]),
            y: retestEntries.map(e => e[1]),
            type: "bar",
            marker: {{ color: "#ffc107", line: {{ color: "#856404", width: 1 }} }},
            text: retestEntries.map(e => e[1].toString()),
            textposition: "outside",
          }}], {{
            ...layoutBase,
            xaxis: {{ title: {{ text: "Product SN", font: {{ size: 12, weight: 600 }} }}, tickangle: -45, gridcolor: theme.grid }},
            yaxis: {{ title: {{ text: "Number of Tests", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", dtick: 1, gridcolor: theme.grid }},
            bargap: 0.3,
          }}, plotConfig);
        }} else {{
          document.getElementById("retest-chart").innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-size:14px;">No retested products found</div>';
        }}

        // Duration vs Cost (scatter)
        const dvColors = records.map(r => r.result.toLowerCase() === "pass" ? "#28a745" : "#c4262e");
        const dvSymbols = records.map(r => r.retest ? "diamond" : "circle");
        const dvSizes = records.map(r => r.retest ? 14 : 10);
        Plotly.newPlot("duration-vs-cost-chart", [{{
          x: records.map(r => r.duration_min),
          y: records.map(r => r.cost),
          mode: "markers",
          marker: {{ color: dvColors, size: dvSizes, symbol: dvSymbols, line: {{ color: "#ffffff", width: 1.5 }} }},
          text: records.map(r => `SN: ${{r.sn}}<br>Duration: ${{r.duration_min}} min<br>Cost: $${{r.cost.toFixed(2)}}<br>Result: ${{r.result}}${{r.retest ? "<br>RETEST" : ""}}`),
          hoverinfo: "text",
        }}], {{
          ...layoutBase,
          xaxis: {{ title: {{ text: "Test Duration (min)", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid }},
          yaxis: {{ title: {{ text: "Energy Cost ($)", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid }},
        }}, plotConfig);
      }}

      /* ---- Table ---- */
      let sortCol = "start";
      let sortAsc = true;

      function renderTable() {{
        const snFilter = document.getElementById("filter-sn").value.toLowerCase();
        const resultFilter = document.getElementById("filter-result").value;
        const retestFilter = document.getElementById("filter-retest").value;

        let filtered = records.filter(r => {{
          if (snFilter && !r.sn.toLowerCase().includes(snFilter)) return false;
          if (resultFilter !== "all" && r.result.toLowerCase() !== resultFilter) return false;
          if (retestFilter === "yes" && !r.retest) return false;
          if (retestFilter === "no" && r.retest) return false;
          return true;
        }});

        // Sort
        filtered.sort((a, b) => {{
          let va = a[sortCol];
          let vb = b[sortCol];
          if (typeof va === "string") va = va.toLowerCase();
          if (typeof vb === "string") vb = vb.toLowerCase();
          if (va < vb) return sortAsc ? -1 : 1;
          if (va > vb) return sortAsc ? 1 : -1;
          return 0;
        }});

        const tbody = document.getElementById("test-table-body");
        tbody.innerHTML = filtered.map(r => {{
          const rowClass = r.retest ? 'class="retest-row"' : '';
          const resultBadge = r.result.toLowerCase() === "pass"
            ? '<span class="badge badge-pass">Pass</span>'
            : '<span class="badge badge-fail">Fail</span>';
          const retestBadge = r.retest ? ' <span class="badge badge-retest">Retest</span>' : '';
          const methodBadge = r.method === "estimated" ? ' <span class="badge badge-estimated">Est</span>' : '';
          const startFormatted = r.start ? new Date(r.start).toLocaleString() : "N/A";
          return `<tr ${{rowClass}}>
            <td>${{r.sn}}${{retestBadge}}</td>
            <td>${{startFormatted}}</td>
            <td>${{r.duration_min.toFixed(1)}}</td>
            <td>${{resultBadge}}</td>
            <td>${{r.avg_kw.toFixed(3)}}</td>
            <td>${{r.kwh.toFixed(4)}}</td>
            <td>$${{r.cost.toFixed(4)}}</td>
            <td>${{r.test_count}}</td>
            <td>${{r.method}}${{methodBadge}}</td>
          </tr>`;
        }}).join("");

        // Update sort arrows
        document.querySelectorAll("#test-table thead th").forEach(th => {{
          const col = th.dataset.col;
          th.classList.toggle("sorted", col === sortCol);
          const arrow = th.querySelector(".sort-arrow");
          if (arrow) arrow.textContent = (col === sortCol && !sortAsc) ? "\\u25BC" : "\\u25B2";
        }});
      }}

      // Sort on header click
      document.querySelectorAll("#test-table thead th").forEach(th => {{
        th.addEventListener("click", () => {{
          const col = th.dataset.col;
          if (!col) return;
          if (sortCol === col) {{
            sortAsc = !sortAsc;
          }} else {{
            sortCol = col;
            sortAsc = true;
          }}
          renderTable();
        }});
      }});

      // Filter events
      document.getElementById("filter-sn").addEventListener("input", renderTable);
      document.getElementById("filter-result").addEventListener("change", renderTable);
      document.getElementById("filter-retest").addEventListener("change", renderTable);
      document.getElementById("reset-table-filters").addEventListener("click", () => {{
        document.getElementById("filter-sn").value = "";
        document.getElementById("filter-result").value = "all";
        document.getElementById("filter-retest").value = "all";
        renderTable();
      }});

      /* ---- Init ---- */
      renderKPIs();
      renderCharts();
      renderTable();
    </script>
  </body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a product test dashboard with energy cost analysis."
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to the product test spreadsheet (CSV or XLSX).",
    )
    parser.add_argument(
        "--config",
        default="visualization_config.json",
        help="Path to the visualization config JSON file.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the EMS energy CSV (overrides config input_file).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the dashboard HTML (overrides config output_dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CONFIG
    CONFIG = load_config(Path(args.config))

    input_path = Path(args.input or CONFIG["input_file"])
    output_dir = Path(args.output_dir or CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EMS energy data from: {input_path}")
    energy_df = load_energy_data(input_path)

    print(f"Loading product test data from: {args.test_data}")
    test_df = load_test_data(Path(args.test_data))
    print(f"  Found {len(test_df)} test records")

    pt_config = CONFIG.get("product_test", {})
    panel_names = pt_config.get("panels", [])
    print(f"  Energy panels for cost calculation: {panel_names}")

    test_df = enrich_test_data(test_df, energy_df, CONFIG)

    retest_count = test_df["Retest"].sum()
    print(f"  Retested product records: {retest_count}")

    output_filename = pt_config.get("output", "product_test_dashboard.html")
    output_path = output_dir / output_filename
    result = build_product_test_dashboard(test_df, CONFIG, output_path)
    print(f"Dashboard written to: {result}")


if __name__ == "__main__":
    main()
