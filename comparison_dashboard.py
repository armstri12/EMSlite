#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparison_dashboard.py

Generate a comparison dashboard for analyzing energy savings between two time periods.

Expected input format:
- Timestamp column named "Timestamp".
- One or more numeric meter columns (amps).

Config:
- Defaults loaded from visualization_config.json (override with --config).
- Config loader accepts JSON with optional // or /* */ comments and trailing commas.

Outputs:
- comparison_dashboard.html (period comparison analysis)
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Iterable
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

TOTAL_AMPS_COLUMN_NAME = "Total_Amps"
TOTAL_KW_COLUMN_NAME = "Total_kW"
PLOT_THEME = {
    "font_family": "Calibri, Segoe UI, Helvetica Neue, Arial, sans-serif",
    "ink": "#2d363a",
    "card": "#ffffff",
    "grid": "rgba(45, 54, 58, 0.12)",
    "colorway": ["#c4262e", "#2d363a", "#000000", "#6b7280", "#cbd0cc"],
}

DEFAULT_CONFIG = {
    "input_file": "RawPanelUsageHistory_UPDATED.csv",
    "output_dir": "visualizations",
    "line_voltage": 480.0,
    "power_factor": 1.0,
    "price_per_kwh": 0.25,
    "total_amps_sources": None,
    "utility_meters": [],
    "combo_columns": {
        "Production_kW": [],
        "Facilities_kW": [],
        "Engineering_kW": [],
    },
    "rolling_window": "1h",
    "dashboard_logo_path": "",
    "visualizations": {
        "comparison_dashboard": {
            "enabled": True,
            "output": "comparison_dashboard.html",
        },
    },
}

CONFIG: dict[str, object] = deepcopy(DEFAULT_CONFIG)


def amps_to_kw(amps: pd.Series) -> pd.Series:
    return amps * (CONFIG["line_voltage"] * 3 ** 0.5 * CONFIG["power_factor"]) / 1000.0


def resolve_columns(available: Iterable[str], requested: list[str] | None, label: str) -> list[str]:
    available_set = set(available)
    if requested is None:
        return [c for c in available if c in available_set]
    missing = [c for c in requested if c not in available_set]
    if missing:
        print(f"Warning: {label} missing columns skipped: {missing}")
    return [c for c in requested if c in available_set]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison dashboard for two time periods.")
    parser.add_argument(
        "--config",
        default="visualization_config.json",
        help="Path to the visualization config JSON file.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the input CSV file (wide format with Timestamp column).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write HTML plots (default: visualizations).",
    )
    parser.add_argument(
        "--period1-start",
        default=None,
        help="Start date for period 1 (YYYY-MM-DD). If not provided, will use earliest available data.",
    )
    parser.add_argument(
        "--period1-end",
        default=None,
        help="End date for period 1 (YYYY-MM-DD). Required if period1-start is provided.",
    )
    parser.add_argument(
        "--period2-start",
        default=None,
        help="Start date for period 2 (YYYY-MM-DD). Required if period1-start is provided.",
    )
    parser.add_argument(
        "--period2-end",
        default=None,
        help="End date for period 2 (YYYY-MM-DD). Required if period1-start is provided.",
    )
    return parser.parse_args()


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
        raise FileNotFoundError(
            f"Config file not found: {path}. Create it or pass --config to specify one."
        )
    raw_text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        sanitized = strip_json_noise(raw_text)
        loaded = json.loads(sanitized)
    return merge_config(DEFAULT_CONFIG, loaded)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" not in df.columns:
        raise ValueError("Input CSV must include a 'Timestamp' column.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    for col in df.columns:
        if col != "Timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def meter_columns(columns: Iterable[str]) -> list[str]:
    computed_names = {
        TOTAL_AMPS_COLUMN_NAME,
        TOTAL_KW_COLUMN_NAME,
        *CONFIG["combo_columns"].keys(),
    }
    return [col for col in columns if col != "Timestamp" and col not in computed_names]


def add_usage_columns(df: pd.DataFrame, meters: list[str]) -> pd.DataFrame:
    df = df.copy()
    total_sources = resolve_columns(meters, CONFIG["total_amps_sources"], "total_amps_sources")
    if total_sources:
        total_amps = df[total_sources].fillna(0).sum(axis=1)
        df[TOTAL_AMPS_COLUMN_NAME] = total_amps
        df[TOTAL_KW_COLUMN_NAME] = amps_to_kw(total_amps)

    for group_name, group_columns in CONFIG["combo_columns"].items():
        resolved = resolve_columns(meters, group_columns, f"combo_columns[{group_name}]")
        if not resolved:
            continue
        group_amps = df[resolved].fillna(0).sum(axis=1)
        df[group_name] = amps_to_kw(group_amps)

    return df


def compute_energy_metrics(df: pd.DataFrame, kw_column: str = TOTAL_KW_COLUMN_NAME) -> dict[str, float]:
    if kw_column not in df.columns:
        return {}
    ordered = df.sort_values("Timestamp").dropna(subset=["Timestamp"])
    deltas = ordered["Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    total_kwh = (ordered[kw_column] * deltas).sum()
    return {
        "total_kwh": float(total_kwh),
        "average_kw": float(ordered[kw_column].mean()),
        "peak_kw": float(ordered[kw_column].max()),
    }


def filter_by_period(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Filter dataframe by date range."""
    if start_date is None and end_date is None:
        return df

    filtered = df.copy()
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        filtered = filtered[filtered["Timestamp"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True) + timedelta(days=1) - timedelta(seconds=1)
        filtered = filtered[filtered["Timestamp"] <= end_dt]

    return filtered


def auto_select_periods(df: pd.DataFrame) -> tuple[tuple[str, str], tuple[str, str]]:
    """Auto-select two comparable periods from the data."""
    timestamps = df["Timestamp"].sort_values()
    min_date = timestamps.min()
    max_date = timestamps.max()

    total_days = (max_date - min_date).days

    # If we have at least 14 days, compare two 7-day periods
    if total_days >= 14:
        period2_end = max_date.strftime("%Y-%m-%d")
        period2_start = (max_date - timedelta(days=6)).strftime("%Y-%m-%d")
        period1_end = (max_date - timedelta(days=7)).strftime("%Y-%m-%d")
        period1_start = (max_date - timedelta(days=13)).strftime("%Y-%m-%d")
    # Otherwise split the data in half
    else:
        mid_point = min_date + timedelta(days=total_days // 2)
        period1_start = min_date.strftime("%Y-%m-%d")
        period1_end = mid_point.strftime("%Y-%m-%d")
        period2_start = (mid_point + timedelta(days=1)).strftime("%Y-%m-%d")
        period2_end = max_date.strftime("%Y-%m-%d")

    return (period1_start, period1_end), (period2_start, period2_end)


def build_comparison_dashboard(
    df: pd.DataFrame,
    period1: pd.DataFrame,
    period2: pd.DataFrame,
    period1_dates: tuple[str, str],
    period2_dates: tuple[str, str],
    output_dir: Path
) -> Path:
    """Build the comparison dashboard HTML."""

    # Compute metrics for both periods
    metrics1 = compute_energy_metrics(period1)
    metrics2 = compute_energy_metrics(period2)

    # Calculate savings
    energy_savings = metrics1["total_kwh"] - metrics2["total_kwh"]
    energy_savings_pct = (energy_savings / metrics1["total_kwh"] * 100) if metrics1["total_kwh"] > 0 else 0

    cost1 = metrics1["total_kwh"] * CONFIG["price_per_kwh"]
    cost2 = metrics2["total_kwh"] * CONFIG["price_per_kwh"]
    cost_savings = cost1 - cost2
    cost_savings_pct = (cost_savings / cost1 * 100) if cost1 > 0 else 0

    avg_load_change = metrics2["average_kw"] - metrics1["average_kw"]
    avg_load_change_pct = (avg_load_change / metrics1["average_kw"] * 100) if metrics1["average_kw"] > 0 else 0

    peak_load_reduction = metrics1["peak_kw"] - metrics2["peak_kw"]
    peak_load_reduction_pct = (peak_load_reduction / metrics1["peak_kw"] * 100) if metrics1["peak_kw"] > 0 else 0

    # Load factor = (Average Load / Peak Load) * 100
    load_factor1 = (metrics1["average_kw"] / metrics1["peak_kw"] * 100) if metrics1["peak_kw"] > 0 else 0
    load_factor2 = (metrics2["average_kw"] / metrics2["peak_kw"] * 100) if metrics2["peak_kw"] > 0 else 0
    load_factor_improvement = load_factor2 - load_factor1

    # Calculate daily averages
    period1_days = (pd.to_datetime(period1_dates[1]) - pd.to_datetime(period1_dates[0])).days + 1
    period2_days = (pd.to_datetime(period2_dates[1]) - pd.to_datetime(period2_dates[0])).days + 1

    daily_energy1 = metrics1["total_kwh"] / period1_days if period1_days > 0 else 0
    daily_energy2 = metrics2["total_kwh"] / period2_days if period2_days > 0 else 0
    daily_cost1 = cost1 / period1_days if period1_days > 0 else 0
    daily_cost2 = cost2 / period2_days if period2_days > 0 else 0

    # Prepare time series data
    p1_ordered = period1.sort_values("Timestamp")
    p2_ordered = period2.sort_values("Timestamp")

    p1_timestamps = p1_ordered["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    p2_timestamps = p2_ordered["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    p1_kw = p1_ordered[TOTAL_KW_COLUMN_NAME].fillna(0).tolist()
    p2_kw = p2_ordered[TOTAL_KW_COLUMN_NAME].fillna(0).tolist()

    price_per_kwh = float(CONFIG["price_per_kwh"])
    logo_path = CONFIG.get("dashboard_logo_path") or ""

    data_payload = {
        "period1": {
            "dates": period1_dates,
            "timestamps": p1_timestamps,
            "kw": p1_kw,
            "total_kwh": metrics1["total_kwh"],
            "average_kw": metrics1["average_kw"],
            "peak_kw": metrics1["peak_kw"],
            "total_cost": cost1,
            "load_factor": load_factor1,
            "daily_energy": daily_energy1,
            "daily_cost": daily_cost1,
        },
        "period2": {
            "dates": period2_dates,
            "timestamps": p2_timestamps,
            "kw": p2_kw,
            "total_kwh": metrics2["total_kwh"],
            "average_kw": metrics2["average_kw"],
            "peak_kw": metrics2["peak_kw"],
            "total_cost": cost2,
            "load_factor": load_factor2,
            "daily_energy": daily_energy2,
            "daily_cost": daily_cost2,
        },
        "savings": {
            "energy_kwh": energy_savings,
            "energy_pct": energy_savings_pct,
            "cost": cost_savings,
            "cost_pct": cost_savings_pct,
            "avg_load_change": avg_load_change,
            "avg_load_change_pct": avg_load_change_pct,
            "peak_load_reduction": peak_load_reduction,
            "peak_load_reduction_pct": peak_load_reduction_pct,
            "load_factor_improvement": load_factor_improvement,
        },
        "price_per_kwh": price_per_kwh,
    }

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Energy Comparison Dashboard</title>
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
            --success: #10b981;
            --success-soft: rgba(16, 185, 129, 0.08);
            --warning: #f59e0b;
            --warning-soft: rgba(245, 158, 11, 0.08);
            --outline: rgba(45, 54, 58, 0.08);
            --border: rgba(45, 54, 58, 0.12);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
            --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
          }}
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--ink);
            line-height: 1.6;
            padding-top: 140px;
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
          .header-nav {{
            padding: 0 48px 16px;
            max-width: 1920px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
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
          .period-selector {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
          }}
          .period-selector label {{
            font-size: 11px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }}
          .period-selector input {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
            color: var(--ink);
            background: var(--card);
            transition: all 0.2s ease;
          }}
          .period-selector input:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
          }}
          .period-selector button {{
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
          .period-selector button:hover {{
            background: var(--accent-hover);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
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
          .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 24px;
          }}
          .comparison-card {{
            background: var(--card);
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border);
            transition: all 0.2s ease;
          }}
          .comparison-card:hover {{
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
          }}
          .comparison-card.savings {{
            border: 2px solid var(--success);
            background: linear-gradient(135deg, var(--card), var(--success-soft));
          }}
          .comparison-card.increase {{
            border: 2px solid var(--warning);
            background: linear-gradient(135deg, var(--card), var(--warning-soft));
          }}
          .metric-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--muted);
            margin-bottom: 12px;
          }}
          .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
          }}
          .metric-period {{
            font-size: 11px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
          }}
          .metric-value {{
            font-size: 20px;
            font-weight: 700;
            color: var(--ink-strong);
          }}
          .metric-delta {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
          }}
          .delta-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--muted);
          }}
          .delta-value {{
            font-size: 18px;
            font-weight: 700;
          }}
          .delta-value.positive {{
            color: var(--success);
          }}
          .delta-value.negative {{
            color: var(--warning);
          }}
          .savings-summary {{
            background: linear-gradient(135deg, var(--success), #059669);
            color: #ffffff;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: var(--shadow-lg);
          }}
          .savings-title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 20px;
          }}
          .savings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
          }}
          .savings-item {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 16px;
          }}
          .savings-item-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            opacity: 0.9;
            margin-bottom: 8px;
          }}
          .savings-item-value {{
            font-size: 24px;
            font-weight: 700;
            line-height: 1;
          }}
          .chart-card {{
            background: var(--card);
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border);
            transition: all 0.2s ease;
            margin-bottom: 24px;
          }}
          .chart-card:hover {{
            box-shadow: var(--shadow-lg);
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
            .comparison-grid {{
              grid-template-columns: repeat(2, 1fr);
            }}
          }}
          @media (max-width: 768px) {{
            body {{
              padding-top: 180px;
            }}
            .container {{
              padding: 0 24px;
            }}
            .top-bar {{
              flex-direction: column;
              align-items: flex-start;
              padding: 16px 24px;
            }}
            .header-nav {{
              padding: 0 24px 12px;
              flex-direction: column;
              align-items: flex-start;
            }}
            .comparison-grid {{
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
                <h1 class="title">Energy Comparison Dashboard</h1>
                <div class="subtitle">Period-over-period energy savings analysis</div>
              </div>
            </div>
          </div>
          <div class="header-nav">
            <div class="nav-pills">
              <a class="nav-pill" href="#savings-section">Savings Summary</a>
              <a class="nav-pill" href="#comparison-section">Metric Comparison</a>
              <a class="nav-pill" href="#charts-section">Visual Analysis</a>
            </div>
            <div class="period-selector">
              <div>
                <label for="p1-start">Period 1 Start</label>
                <input type="date" id="p1-start" />
              </div>
              <div>
                <label for="p1-end">Period 1 End</label>
                <input type="date" id="p1-end" />
              </div>
              <div>
                <label for="p2-start">Period 2 Start</label>
                <input type="date" id="p2-start" />
              </div>
              <div>
                <label for="p2-end">Period 2 End</label>
                <input type="date" id="p2-end" />
              </div>
              <button id="refresh-btn">Refresh</button>
            </div>
          </div>
        </div>

        <div class="container">
          <div class="section" id="savings-section">
            <div class="savings-summary" id="savings-summary">
              <div class="savings-title">Energy Savings Impact</div>
              <div class="savings-grid">
                <div class="savings-item">
                  <div class="savings-item-label">Energy Savings</div>
                  <div class="savings-item-value" id="energy-savings">0.00 kWh</div>
                </div>
                <div class="savings-item">
                  <div class="savings-item-label">Cost Savings</div>
                  <div class="savings-item-value" id="cost-savings">$0.00</div>
                </div>
                <div class="savings-item">
                  <div class="savings-item-label">Peak Reduction</div>
                  <div class="savings-item-value" id="peak-reduction">0.00 kW</div>
                </div>
                <div class="savings-item">
                  <div class="savings-item-label">Load Factor Improvement</div>
                  <div class="savings-item-value" id="load-factor-improvement">0.0%</div>
                </div>
              </div>
            </div>
          </div>

          <div class="section" id="comparison-section">
            <div class="section-header">
              <h2 class="section-title">Metric Comparison</h2>
              <div class="section-description">Side-by-side comparison of key energy metrics</div>
            </div>
            <div class="comparison-grid">
              <div class="comparison-card" id="energy-card">
                <div class="metric-label">Total Energy Consumption</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-energy">0.00 kWh</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-energy">0.00 kWh</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="energy-delta">0.00 kWh (0.0%)</span>
                </div>
              </div>

              <div class="comparison-card" id="cost-card">
                <div class="metric-label">Total Energy Cost</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-cost">$0.00</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-cost">$0.00</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="cost-delta">$0.00 (0.0%)</span>
                </div>
              </div>

              <div class="comparison-card" id="avg-load-card">
                <div class="metric-label">Average Load</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-avg">0.00 kW</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-avg">0.00 kW</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="avg-delta">0.00 kW (0.0%)</span>
                </div>
              </div>

              <div class="comparison-card" id="peak-load-card">
                <div class="metric-label">Peak Load</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-peak">0.00 kW</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-peak">0.00 kW</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="peak-delta">0.00 kW (0.0%)</span>
                </div>
              </div>

              <div class="comparison-card" id="load-factor-card">
                <div class="metric-label">Load Factor</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-lf">0.0%</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-lf">0.0%</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="lf-delta">0.0 pts</span>
                </div>
              </div>

              <div class="comparison-card" id="daily-energy-card">
                <div class="metric-label">Daily Energy (Average)</div>
                <div class="metric-row">
                  <span class="metric-period">Period 1</span>
                  <span class="metric-value" id="p1-daily">0.00 kWh/day</span>
                </div>
                <div class="metric-row">
                  <span class="metric-period">Period 2</span>
                  <span class="metric-value" id="p2-daily">0.00 kWh/day</span>
                </div>
                <div class="metric-delta">
                  <span class="delta-label">Change</span>
                  <span class="delta-value" id="daily-delta">0.00 kWh/day</span>
                </div>
              </div>
            </div>
          </div>

          <div class="section" id="charts-section">
            <div class="section-header">
              <h2 class="section-title">Visual Analysis</h2>
              <div class="section-description">Detailed comparative visualizations</div>
            </div>

            <div class="chart-card">
              <div class="chart-title">Load Profile Comparison</div>
              <div id="load-comparison-chart" class="chart"></div>
            </div>

            <div class="chart-card">
              <div class="chart-title">Hourly Average Comparison</div>
              <div id="hourly-comparison-chart" class="chart"></div>
            </div>

            <div class="chart-card">
              <div class="chart-title">Weekday Average Comparison</div>
              <div id="weekday-comparison-chart" class="chart"></div>
            </div>
          </div>
        </div>

        <div class="footer">
          Energy Management System &middot; Period Comparison Analysis
        </div>
        <script>
          const comparisonData = {json.dumps(data_payload)};
          const logoPath = {json.dumps(logo_path)};
          const theme = {{
            ink: "#2d363a",
            inkStrong: "#1a1f22",
            muted: "#6c757d",
            card: "#ffffff",
            grid: "rgba(0, 0, 0, 0.08)",
            accent: "#c4262e",
            accentDark: "#2d363a",
            success: "#10b981",
            series: ["#c4262e", "#2d363a"]
          }};

          function applyLogo(src) {{
            const logoImage = document.getElementById("logo-image");
            const logoPlaceholder = document.getElementById("logo-placeholder");
            if (!src) {{
              logoImage.style.display = "none";
              logoPlaceholder.style.display = "block";
              return;
            }}
            logoImage.src = src;
            logoImage.style.display = "block";
            logoPlaceholder.style.display = "none";
          }}

          function formatNumber(num, decimals = 2) {{
            return num.toFixed(decimals);
          }}

          function updateSavingsSummary() {{
            const savings = comparisonData.savings;

            // Determine if we have savings or increase
            const hasSavings = savings.energy_kwh > 0;
            const summaryEl = document.getElementById("savings-summary");

            if (hasSavings) {{
              summaryEl.style.background = "linear-gradient(135deg, var(--success), #059669)";
              document.getElementById("energy-savings").textContent =
                `${{formatNumber(savings.energy_kwh)}} kWh (${{formatNumber(savings.energy_pct)}}%)`;
              document.getElementById("cost-savings").textContent =
                `$${{formatNumber(savings.cost)}} (${{formatNumber(savings.cost_pct)}}%)`;
            }} else {{
              summaryEl.style.background = "linear-gradient(135deg, var(--warning), #d97706)";
              document.getElementById("energy-savings").textContent =
                `${{formatNumber(-savings.energy_kwh)}} kWh increase`;
              document.getElementById("cost-savings").textContent =
                `$${{formatNumber(-savings.cost)}} increase`;
            }}

            document.getElementById("peak-reduction").textContent =
              `${{formatNumber(savings.peak_load_reduction)}} kW (${{formatNumber(savings.peak_load_reduction_pct)}}%)`;
            document.getElementById("load-factor-improvement").textContent =
              `${{formatNumber(savings.load_factor_improvement, 1)}}%`;
          }}

          function updateMetricComparison() {{
            const p1 = comparisonData.period1;
            const p2 = comparisonData.period2;
            const savings = comparisonData.savings;

            // Energy
            document.getElementById("p1-energy").textContent = `${{formatNumber(p1.total_kwh)}} kWh`;
            document.getElementById("p2-energy").textContent = `${{formatNumber(p2.total_kwh)}} kWh`;
            const energyDelta = document.getElementById("energy-delta");
            energyDelta.textContent = `${{formatNumber(-savings.energy_kwh)}} kWh (${{formatNumber(-savings.energy_pct)}}%)`;
            energyDelta.className = savings.energy_kwh > 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("energy-card").className = savings.energy_kwh > 0 ? "comparison-card savings" : "comparison-card increase";

            // Cost
            document.getElementById("p1-cost").textContent = `$${{formatNumber(p1.total_cost)}}`;
            document.getElementById("p2-cost").textContent = `$${{formatNumber(p2.total_cost)}}`;
            const costDelta = document.getElementById("cost-delta");
            costDelta.textContent = `$${{formatNumber(-savings.cost)}} (${{formatNumber(-savings.cost_pct)}}%)`;
            costDelta.className = savings.cost > 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("cost-card").className = savings.cost > 0 ? "comparison-card savings" : "comparison-card increase";

            // Average Load
            document.getElementById("p1-avg").textContent = `${{formatNumber(p1.average_kw)}} kW`;
            document.getElementById("p2-avg").textContent = `${{formatNumber(p2.average_kw)}} kW`;
            const avgDelta = document.getElementById("avg-delta");
            avgDelta.textContent = `${{formatNumber(savings.avg_load_change)}} kW (${{formatNumber(savings.avg_load_change_pct)}}%)`;
            avgDelta.className = savings.avg_load_change < 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("avg-load-card").className = savings.avg_load_change < 0 ? "comparison-card savings" : "comparison-card increase";

            // Peak Load
            document.getElementById("p1-peak").textContent = `${{formatNumber(p1.peak_kw)}} kW`;
            document.getElementById("p2-peak").textContent = `${{formatNumber(p2.peak_kw)}} kW`;
            const peakDelta = document.getElementById("peak-delta");
            peakDelta.textContent = `${{formatNumber(-savings.peak_load_reduction)}} kW (${{formatNumber(-savings.peak_load_reduction_pct)}}%)`;
            peakDelta.className = savings.peak_load_reduction > 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("peak-load-card").className = savings.peak_load_reduction > 0 ? "comparison-card savings" : "comparison-card increase";

            // Load Factor
            document.getElementById("p1-lf").textContent = `${{formatNumber(p1.load_factor, 1)}}%`;
            document.getElementById("p2-lf").textContent = `${{formatNumber(p2.load_factor, 1)}}%`;
            const lfDelta = document.getElementById("lf-delta");
            lfDelta.textContent = `${{formatNumber(savings.load_factor_improvement, 1)}} pts`;
            lfDelta.className = savings.load_factor_improvement > 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("load-factor-card").className = savings.load_factor_improvement > 0 ? "comparison-card savings" : "comparison-card increase";

            // Daily Energy
            document.getElementById("p1-daily").textContent = `${{formatNumber(p1.daily_energy)}} kWh/day`;
            document.getElementById("p2-daily").textContent = `${{formatNumber(p2.daily_energy)}} kWh/day`;
            const dailyDelta = document.getElementById("daily-delta");
            const dailyChange = p2.daily_energy - p1.daily_energy;
            dailyDelta.textContent = `${{formatNumber(dailyChange)}} kWh/day`;
            dailyDelta.className = dailyChange < 0 ? "delta-value positive" : "delta-value negative";
            document.getElementById("daily-energy-card").className = dailyChange < 0 ? "comparison-card savings" : "comparison-card increase";
          }}

          function renderCharts() {{
            const p1 = comparisonData.period1;
            const p2 = comparisonData.period2;

            const layoutBase = {{
              margin: {{ t: 16, l: 60, r: 24, b: 50 }},
              paper_bgcolor: theme.card,
              plot_bgcolor: theme.card,
              font: {{ family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: theme.ink, size: 12 }},
              hovermode: "x unified",
              hoverlabel: {{ bgcolor: theme.inkStrong, font: {{ color: "#ffffff" }} }}
            }};

            // Load Profile Comparison
            Plotly.newPlot("load-comparison-chart", [
              {{
                x: p1.timestamps,
                y: p1.kw,
                mode: "lines",
                name: `Period 1 (${{p1.dates[0]}} to ${{p1.dates[1]}})`,
                line: {{ color: theme.accent, width: 2.5, shape: "spline" }}
              }},
              {{
                x: p2.timestamps,
                y: p2.kw,
                mode: "lines",
                name: `Period 2 (${{p2.dates[0]}} to ${{p2.dates[1]}})`,
                line: {{ color: theme.accentDark, width: 2.5, shape: "spline" }}
              }}
            ], {{
              ...layoutBase,
              xaxis: {{ title: {{ text: "Time", font: {{ size: 12, weight: 600 }} }}, type: "date", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              yaxis: {{ title: {{ text: "kW", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              legend: {{ orientation: "h", y: -0.15 }}
            }}, {{ displaylogo: false, responsive: true }});

            // Hourly Average Comparison
            const p1Hourly = computeHourlyAverage(p1.timestamps, p1.kw);
            const p2Hourly = computeHourlyAverage(p2.timestamps, p2.kw);

            Plotly.newPlot("hourly-comparison-chart", [
              {{
                x: p1Hourly.hours,
                y: p1Hourly.averages,
                mode: "lines+markers",
                name: `Period 1`,
                line: {{ color: theme.accent, width: 2.5, shape: "spline" }},
                marker: {{ size: 8, color: theme.accent }}
              }},
              {{
                x: p2Hourly.hours,
                y: p2Hourly.averages,
                mode: "lines+markers",
                name: `Period 2`,
                line: {{ color: theme.accentDark, width: 2.5, shape: "spline" }},
                marker: {{ size: 8, color: theme.accentDark }}
              }}
            ], {{
              ...layoutBase,
              xaxis: {{ title: {{ text: "Hour of Day", font: {{ size: 12, weight: 600 }} }}, dtick: 2, gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              yaxis: {{ title: {{ text: "Average kW", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              legend: {{ orientation: "h", y: -0.15 }}
            }}, {{ displaylogo: false, responsive: true }});

            // Weekday Average Comparison
            const p1Weekday = computeWeekdayAverage(p1.timestamps, p1.kw);
            const p2Weekday = computeWeekdayAverage(p2.timestamps, p2.kw);

            Plotly.newPlot("weekday-comparison-chart", [
              {{
                x: p1Weekday.weekdays,
                y: p1Weekday.averages,
                type: "bar",
                name: "Period 1",
                marker: {{ color: theme.accent }}
              }},
              {{
                x: p2Weekday.weekdays,
                y: p2Weekday.averages,
                type: "bar",
                name: "Period 2",
                marker: {{ color: theme.accentDark }}
              }}
            ], {{
              ...layoutBase,
              xaxis: {{ title: {{ text: "Day of Week", font: {{ size: 12, weight: 600 }} }}, gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              yaxis: {{ title: {{ text: "Average kW", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
              legend: {{ orientation: "h", y: -0.15 }},
              barmode: "group"
            }}, {{ displaylogo: false, responsive: true }});
          }}

          function computeHourlyAverage(timestamps, kw) {{
            const sums = Array(24).fill(0);
            const counts = Array(24).fill(0);
            timestamps.forEach((ts, idx) => {{
              const hour = new Date(ts).getUTCHours();
              sums[hour] += kw[idx] ?? 0;
              counts[hour] += 1;
            }});
            const averages = sums.map((s, i) => (counts[i] ? s / counts[i] : 0));
            return {{ hours: Array.from({{ length: 24 }}, (_, i) => i), averages }};
          }}

          function computeWeekdayAverage(timestamps, kw) {{
            const sums = Array(7).fill(0);
            const counts = Array(7).fill(0);
            timestamps.forEach((ts, idx) => {{
              const weekday = new Date(ts).getUTCDay();
              sums[weekday] += kw[idx] ?? 0;
              counts[weekday] += 1;
            }});
            const averages = sums.map((s, i) => (counts[i] ? s / counts[i] : 0));
            return {{
              weekdays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
              averages
            }};
          }}

          function initPeriodInputs() {{
            document.getElementById("p1-start").value = comparisonData.period1.dates[0];
            document.getElementById("p1-end").value = comparisonData.period1.dates[1];
            document.getElementById("p2-start").value = comparisonData.period2.dates[0];
            document.getElementById("p2-end").value = comparisonData.period2.dates[1];
          }}

          document.getElementById("refresh-btn").addEventListener("click", () => {{
            const p1Start = document.getElementById("p1-start").value;
            const p1End = document.getElementById("p1-end").value;
            const p2Start = document.getElementById("p2-start").value;
            const p2End = document.getElementById("p2-end").value;

            const url = new URL(window.location.href);
            url.searchParams.set("regenerate", "1");
            url.searchParams.set("p1_start", p1Start);
            url.searchParams.set("p1_end", p1End);
            url.searchParams.set("p2_start", p2Start);
            url.searchParams.set("p2_end", p2End);

            alert("To regenerate with new periods, run the comparison_dashboard.py script with:\\n\\n" +
              `--period1-start ${{p1Start}} --period1-end ${{p1End}} --period2-start ${{p2Start}} --period2-end ${{p2End}}`);
          }});

          initPeriodInputs();
          updateSavingsSummary();
          updateMetricComparison();
          renderCharts();
          applyLogo(logoPath);
        </script>
      </body>
    </html>
    """

    output_path = output_dir / CONFIG["visualizations"]["comparison_dashboard"]["output"]
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    global CONFIG
    CONFIG = load_config(Path(args.config))
    input_path = Path(args.input or CONFIG["input_file"])
    output_dir = Path(args.output_dir or CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    meters = meter_columns(df.columns)
    if not meters:
        raise ValueError("No meter columns found in the input file.")

    df = add_usage_columns(df, meters)

    if TOTAL_KW_COLUMN_NAME not in df.columns:
        raise ValueError(f"Missing {TOTAL_KW_COLUMN_NAME} column. Cannot generate comparison dashboard.")

    # Determine periods
    if args.period1_start and args.period1_end and args.period2_start and args.period2_end:
        period1_dates = (args.period1_start, args.period1_end)
        period2_dates = (args.period2_start, args.period2_end)
        print(f"Using custom periods:")
        print(f"  Period 1: {period1_dates[0]} to {period1_dates[1]}")
        print(f"  Period 2: {period2_dates[0]} to {period2_dates[1]}")
    else:
        period1_dates, period2_dates = auto_select_periods(df)
        print(f"Auto-selected periods:")
        print(f"  Period 1: {period1_dates[0]} to {period1_dates[1]}")
        print(f"  Period 2: {period2_dates[0]} to {period2_dates[1]}")

    # Filter data for each period
    period1_df = filter_by_period(df, period1_dates[0], period1_dates[1])
    period2_df = filter_by_period(df, period2_dates[0], period2_dates[1])

    if period1_df.empty or period2_df.empty:
        raise ValueError("One or both periods have no data. Please check your date ranges.")

    print(f"Period 1 data points: {len(period1_df)}")
    print(f"Period 2 data points: {len(period2_df)}")

    # Build comparison dashboard
    output_path = build_comparison_dashboard(
        df, period1_df, period2_df, period1_dates, period2_dates, output_dir
    )

    print(f"Generated comparison dashboard: {output_path}")


if __name__ == "__main__":
    main()
