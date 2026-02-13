#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparison_dashboard.py

Generate a comparison dashboard for analyzing energy savings between two time periods.
Now with interactive panel selection and period adjustment.

Expected input format:
- Timestamp column named "Timestamp".
- One or more numeric meter columns (amps).

Config:
- Defaults loaded from visualization_config.json (override with --config).
- Config loader accepts JSON with optional // or /* */ comments and trailing commas.

Outputs:
- comparison_dashboard.html (interactive period comparison analysis)
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

TOTAL_AMPS_COLUMN_NAME = "Total_Amps"
TOTAL_KW_COLUMN_NAME = "Total_kW"

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
        help="Start date for period 1 (YYYY-MM-DD). If not provided, will auto-select.",
    )
    parser.add_argument(
        "--period1-end",
        default=None,
        help="End date for period 1 (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--period2-start",
        default=None,
        help="Start date for period 2 (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--period2-end",
        default=None,
        help="End date for period 2 (YYYY-MM-DD).",
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
    period1_dates: tuple[str, str],
    period2_dates: tuple[str, str],
    output_dir: Path
) -> Path:
    """Build the comparison dashboard HTML with full dataset for client-side filtering."""

    # Prepare full dataset with panel series for client-side filtering
    ordered = df.sort_values("Timestamp")
    timestamps = ordered["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    panel_columns = meter_columns(ordered.columns)
    panel_series = {
        name: amps_to_kw(ordered[name].fillna(0)).fillna(0).tolist() for name in panel_columns
    }

    # Get min/max dates for the date inputs
    min_date = ordered["Timestamp"].min().strftime("%Y-%m-%d")
    max_date = ordered["Timestamp"].max().strftime("%Y-%m-%d")

    price_per_kwh = float(CONFIG["price_per_kwh"])
    logo_path = CONFIG.get("dashboard_logo_path") or ""

    data_payload = {
        "timestamps": timestamps,
        "panel_series": panel_series,
        "panel_names": panel_columns,
        "period1_dates": period1_dates,
        "period2_dates": period2_dates,
        "min_date": min_date,
        "max_date": max_date,
        "price_per_kwh": price_per_kwh,
    }

    # Generate HTML with embedded JavaScript for client-side filtering
    html_content = generate_dashboard_html(data_payload, logo_path)

    output_path = output_dir / CONFIG["visualizations"]["comparison_dashboard"]["output"]
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def generate_dashboard_html(data: dict, logo_path: str) -> str:
    """Generate the complete HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Energy Comparison Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
  <style>
    :root {{
      --bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      --bg-light: #f0f4f8;
      --card: #ffffff;
      --ink: #2d3748;
      --ink-strong: #1a202c;
      --muted: #718096;
      --accent: #c4262e;
      --accent-hover: #a61f26;
      --accent-soft: rgba(196, 38, 46, 0.08);
      --success: #48bb78;
      --success-dark: #38a169;
      --success-soft: rgba(72, 187, 120, 0.1);
      --warning: #ed8936;
      --warning-dark: #dd6b20;
      --warning-soft: rgba(237, 137, 54, 0.1);
      --outline: rgba(45, 54, 58, 0.08);
      --border: rgba(45, 54, 58, 0.08);
      --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
      --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
      --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.12);
      --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.15);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-light);
      color: var(--ink);
      line-height: 1.6;
      padding-top: 280px;
    }}
    .header-wrapper {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
      backdrop-filter: blur(20px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
      z-index: 1000;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
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
      width: 60px;
      height: 60px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.15);
      border: 2px solid rgba(255, 255, 255, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      flex-shrink: 0;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}
    .logo-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .logo-placeholder {{
      font-size: 11px;
      font-weight: 800;
      color: #ffffff;
      text-align: center;
      padding: 8px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}
    .brand-text {{
      display: flex;
      flex-direction: column;
    }}
    .title {{
      font-size: 32px;
      font-weight: 900;
      color: #ffffff;
      letter-spacing: -0.8px;
      text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    .subtitle {{
      margin-top: 4px;
      color: rgba(255, 255, 255, 0.9);
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.3px;
    }}
    .controls {{
      padding: 20px 48px 24px;
      max-width: 1920px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    .control-row label {{
      font-size: 12px;
      font-weight: 700;
      color: rgba(255, 255, 255, 0.95);
      text-transform: uppercase;
      letter-spacing: 1px;
      min-width: 80px;
    }}
    .control-row input {{
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 13px;
      color: #ffffff;
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }}
    .control-row input::placeholder {{
      color: rgba(255, 255, 255, 0.6);
    }}
    .control-row input:focus {{
      outline: none;
      border-color: rgba(255, 255, 255, 0.6);
      background: rgba(255, 255, 255, 0.2);
      box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
    }}
    .control-row button {{
      border: none;
      border-radius: 10px;
      padding: 10px 20px;
      font-size: 14px;
      font-weight: 700;
      color: #667eea;
      background: #ffffff;
      cursor: pointer;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }}
    .control-row button:hover {{
      background: #f7fafc;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
      transform: translateY(-2px);
    }}
    .panel-filter-wrapper {{
      position: relative;
    }}
    .panel-filter-btn {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 10px;
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
      background: rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(10px);
      cursor: pointer;
      transition: all 0.3s ease;
      white-space: nowrap;
    }}
    .panel-filter-btn:hover {{
      border-color: rgba(255, 255, 255, 0.5);
      background: rgba(255, 255, 255, 0.2);
    }}
    .panel-filter-btn .count-badge {{
      background: #ffffff;
      color: #667eea;
      border-radius: 12px;
      padding: 2px 10px;
      font-size: 12px;
      font-weight: 800;
    }}
    .panel-filter-dropdown {{
      display: none; position: absolute; top: calc(100% + 6px); left: 0;
      min-width: 280px; max-height: 360px; background: var(--card);
      border: 1px solid var(--border); border-radius: 10px;
      box-shadow: var(--shadow-lg); z-index: 2000; flex-direction: column;
    }}
    .panel-filter-dropdown.open {{ display: flex; }}
    .panel-filter-actions {{
      display: flex; gap: 8px; padding: 10px 14px; border-bottom: 1px solid var(--border);
    }}
    .panel-filter-actions button {{
      border: none; background: none; font-size: 12px; font-weight: 600;
      color: var(--accent); cursor: pointer; padding: 2px 4px;
    }}
    .panel-filter-actions button:hover {{ text-decoration: underline; }}
    .panel-filter-list {{ overflow-y: auto; padding: 6px 0; }}
    .panel-filter-item {{
      display: flex; align-items: center; gap: 10px; padding: 7px 14px;
      cursor: pointer; font-size: 13px; transition: background 0.15s ease;
    }}
    .panel-filter-item:hover {{ background: var(--accent-soft); }}
    .panel-filter-item input[type="checkbox"] {{
      width: 15px; height: 15px; cursor: pointer; accent-color: var(--accent);
    }}
    .container {{
      max-width: 1920px;
      margin: 0 auto;
      padding: 32px 48px;
    }}
    .section {{
      margin-bottom: 60px;
    }}
    .section-header {{
      margin-bottom: 32px;
      position: relative;
      padding-left: 20px;
    }}
    .section-header::before {{
      content: '';
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: linear-gradient(180deg, #667eea, #764ba2);
      border-radius: 2px;
    }}
    .section-title {{
      font-size: 28px;
      font-weight: 900;
      color: var(--ink-strong);
      letter-spacing: -0.8px;
      margin-bottom: 8px;
    }}
    .section-description {{
      font-size: 15px;
      color: var(--muted);
      font-weight: 500;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
      margin-bottom: 32px;
    }}
    .comparison-card {{
      background: var(--card);
      border-radius: 16px;
      padding: 28px;
      box-shadow: var(--shadow-md);
      border: 2px solid transparent;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
    }}
    .comparison-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      opacity: 0;
      transition: opacity 0.3s ease;
    }}
    .comparison-card:hover {{
      box-shadow: var(--shadow-xl);
      transform: translateY(-4px);
      border-color: var(--accent);
    }}
    .comparison-card:hover::before {{
      opacity: 1;
    }}
    .comparison-card.savings {{
      border-color: var(--success);
      background: linear-gradient(135deg, var(--card) 0%, var(--success-soft) 100%);
    }}
    .comparison-card.savings::before {{
      background: linear-gradient(90deg, var(--success), var(--success-dark), var(--success));
      opacity: 1;
    }}
    .comparison-card.increase {{
      border-color: var(--warning);
      background: linear-gradient(135deg, var(--card) 0%, var(--warning-soft) 100%);
    }}
    .comparison-card.increase::before {{
      background: linear-gradient(90deg, var(--warning), var(--warning-dark), var(--warning));
      opacity: 1;
    }}
    .metric-label {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--muted);
      margin-bottom: 16px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .metric-period {{
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 800;
      color: var(--ink-strong);
    }}
    .metric-delta {{
      margin-top: 16px;
      padding-top: 16px;
      border-top: 2px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .delta-label {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--muted);
    }}
    .delta-value {{
      font-size: 22px;
      font-weight: 900;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .delta-value.positive {{
      color: var(--success);
    }}
    .delta-value.positive::before {{
      content: '↓';
      font-size: 28px;
      line-height: 1;
    }}
    .delta-value.negative {{
      color: var(--warning);
    }}
    .delta-value.negative::before {{
      content: '↑';
      font-size: 28px;
      line-height: 1;
    }}
    .savings-summary {{
      background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
      color: #ffffff;
      border-radius: 20px;
      padding: 48px;
      margin-bottom: 40px;
      box-shadow: var(--shadow-xl);
      position: relative;
      overflow: hidden;
    }}
    .savings-summary::before {{
      content: '';
      position: absolute;
      top: -50%;
      right: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
      animation: pulse 15s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: translate(0, 0) scale(1); }}
      50% {{ transform: translate(-20px, -20px) scale(1.1); }}
    }}
    .savings-title {{
      font-size: 32px;
      font-weight: 800;
      margin-bottom: 24px;
      position: relative;
      z-index: 1;
      text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .savings-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 24px;
      position: relative;
      z-index: 1;
    }}
    .savings-item {{
      background: rgba(255, 255, 255, 0.15);
      border-radius: 16px;
      padding: 24px;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      transition: all 0.3s ease;
    }}
    .savings-item:hover {{
      background: rgba(255, 255, 255, 0.2);
      transform: translateY(-4px);
      box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }}
    .savings-item-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      opacity: 0.95;
      margin-bottom: 12px;
    }}
    .savings-item-value {{
      font-size: 36px;
      font-weight: 900;
      line-height: 1;
      text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .chart-card {{
      background: var(--card);
      border-radius: 16px;
      padding: 32px;
      box-shadow: var(--shadow-md);
      border: 1px solid var(--border);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      margin-bottom: 32px;
      position: relative;
      overflow: hidden;
    }}
    .chart-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--accent), #764ba2, var(--accent));
    }}
    .chart-card:hover {{
      box-shadow: var(--shadow-xl);
      transform: translateY(-2px);
    }}
    .chart-title {{
      font-size: 18px;
      font-weight: 800;
      color: var(--ink-strong);
      margin-bottom: 20px;
      letter-spacing: -0.3px;
    }}
    .chart {{
      min-height: 400px;
    }}
    .footer {{
      padding: 40px 48px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      text-align: center;
      border-top: 2px solid var(--border);
      background: var(--card);
      margin-top: 60px;
    }}
    @media (max-width: 1200px) {{
      .comparison-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}
    @media (max-width: 768px) {{
      body {{
        padding-top: 340px;
      }}
      .container {{
        padding: 24px 24px;
      }}
      .top-bar {{
        flex-direction: column;
        align-items: flex-start;
        padding: 16px 24px;
      }}
      .controls {{
        padding: 16px 24px 20px;
      }}
      .comparison-grid {{
        grid-template-columns: 1fr;
      }}
      .savings-grid {{
        grid-template-columns: 1fr;
      }}
      .footer {{
        padding: 32px 24px;
      }}
      .section-header {{
        padding-left: 16px;
      }}
      .section-header::before {{
        width: 3px;
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
          <div class="subtitle">Interactive period-over-period analysis</div>
        </div>
      </div>
    </div>
    <div class="controls">
      <div class="control-row">
        <div class="panel-filter-wrapper" id="panel-filter-wrapper">
          <button class="panel-filter-btn" id="panel-filter-btn">
            Panels <span class="count-badge" id="panel-count-badge">All</span>
          </button>
          <div class="panel-filter-dropdown" id="panel-filter-dropdown">
            <div class="panel-filter-actions">
              <button id="panel-select-all">Select All</button>
              <button id="panel-select-none">Clear</button>
            </div>
            <div class="panel-filter-list" id="panel-filter-list"></div>
          </div>
        </div>
      </div>
      <div class="control-row">
        <label>Period 1:</label>
        <input type="date" id="p1-start" />
        <span style="color: var(--muted);">to</span>
        <input type="date" id="p1-end" />
      </div>
      <div class="control-row">
        <label>Period 2:</label>
        <input type="date" id="p2-start" />
        <span style="color: var(--muted);">to</span>
        <input type="date" id="p2-end" />
      </div>
      <div class="control-row">
        <button id="apply-btn">Apply Filters</button>
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
    Energy Management System &middot; Interactive Period Comparison
  </div>

  <script>
    const dashboardData = {json.dumps(data)};
    const logoPath = {json.dumps(logo_path)};
    const theme = {{
      ink: "#2d363a", inkStrong: "#1a1f22", muted: "#6c757d", card: "#ffffff",
      grid: "rgba(0, 0, 0, 0.08)", accent: "#c4262e", accentDark: "#2d363a",
      success: "#10b981", series: ["#c4262e", "#2d363a"]
    }};

    // State
    let selectedPanels = new Set(dashboardData.panel_names || []);

    // Initialize
    function init() {{
      applyLogo(logoPath);
      initPanelFilter();
      initPeriodInputs();
      renderDashboard();
    }}

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

    function initPeriodInputs() {{
      document.getElementById("p1-start").value = dashboardData.period1_dates[0];
      document.getElementById("p1-end").value = dashboardData.period1_dates[1];
      document.getElementById("p2-start").value = dashboardData.period2_dates[0];
      document.getElementById("p2-end").value = dashboardData.period2_dates[1];

      document.getElementById("p1-start").min = dashboardData.min_date;
      document.getElementById("p1-start").max = dashboardData.max_date;
      document.getElementById("p1-end").min = dashboardData.min_date;
      document.getElementById("p1-end").max = dashboardData.max_date;
      document.getElementById("p2-start").min = dashboardData.min_date;
      document.getElementById("p2-start").max = dashboardData.max_date;
      document.getElementById("p2-end").min = dashboardData.min_date;
      document.getElementById("p2-end").max = dashboardData.max_date;
    }}

    function initPanelFilter() {{
      const allPanels = dashboardData.panel_names || [];
      if (!allPanels.length) {{
        document.getElementById("panel-filter-wrapper").style.display = "none";
        return;
      }}

      const list = document.getElementById("panel-filter-list");
      allPanels.forEach((panel) => {{
        const item = document.createElement("label");
        item.className = "panel-filter-item";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = panel;
        cb.checked = true;
        cb.addEventListener("change", () => {{
          if (cb.checked) {{
            selectedPanels.add(panel);
          }} else {{
            selectedPanels.delete(panel);
          }}
          syncPanelUI();
        }});
        const span = document.createElement("span");
        span.textContent = panel;
        item.appendChild(cb);
        item.appendChild(span);
        list.appendChild(item);
      }});

      document.getElementById("panel-filter-btn").addEventListener("click", (e) => {{
        e.stopPropagation();
        document.getElementById("panel-filter-dropdown").classList.toggle("open");
      }});
      document.addEventListener("click", (e) => {{
        const wrapper = document.getElementById("panel-filter-wrapper");
        if (!wrapper.contains(e.target)) {{
          document.getElementById("panel-filter-dropdown").classList.remove("open");
        }}
      }});

      document.getElementById("panel-select-all").addEventListener("click", () => {{
        selectedPanels = new Set(allPanels);
        syncPanelCheckboxes();
        syncPanelUI();
      }});
      document.getElementById("panel-select-none").addEventListener("click", () => {{
        selectedPanels.clear();
        syncPanelCheckboxes();
        syncPanelUI();
      }});

      syncPanelUI();
    }}

    function syncPanelCheckboxes() {{
      const cbs = document.querySelectorAll("#panel-filter-list input[type=checkbox]");
      cbs.forEach((cb) => {{
        cb.checked = selectedPanels.has(cb.value);
      }});
    }}

    function syncPanelUI() {{
      const allPanels = dashboardData.panel_names || [];
      const badge = document.getElementById("panel-count-badge");
      if (selectedPanels.size === 0 || selectedPanels.size === allPanels.length) {{
        badge.textContent = selectedPanels.size === 0 ? "None" : "All";
      }} else {{
        badge.textContent = selectedPanels.size;
      }}
    }}

    function filterDataByPanelsAndPeriods() {{
      const p1Start = new Date(document.getElementById("p1-start").value + "T00:00:00Z");
      const p1End = new Date(document.getElementById("p1-end").value + "T23:59:59Z");
      const p2Start = new Date(document.getElementById("p2-start").value + "T00:00:00Z");
      const p2End = new Date(document.getElementById("p2-end").value + "T23:59:59Z");

      const allPanels = dashboardData.panel_names || [];
      const activePanels = selectedPanels.size ? Array.from(selectedPanels) : allPanels;

      const period1 = {{ timestamps: [], kw: [] }};
      const period2 = {{ timestamps: [], kw: [] }};

      dashboardData.timestamps.forEach((ts, idx) => {{
        const date = new Date(ts);

        // Calculate total kW from selected panels
        let totalKw = 0;
        activePanels.forEach((panel) => {{
          totalKw += (dashboardData.panel_series[panel] || [])[idx] || 0;
        }});

        if (date >= p1Start && date <= p1End) {{
          period1.timestamps.push(ts);
          period1.kw.push(totalKw);
        }}
        if (date >= p2Start && date <= p2End) {{
          period2.timestamps.push(ts);
          period2.kw.push(totalKw);
        }}
      }});

      return {{ period1, period2 }};
    }}

    function computeMetrics(timestamps, kw) {{
      let totalKwh = 0, peakKw = 0, sumKw = 0;
      for (let i = 0; i < timestamps.length; i++) {{
        const k = kw[i] ?? 0;
        sumKw += k;
        if (k > peakKw) peakKw = k;
        if (i === 0) continue;
        const prev = new Date(timestamps[i - 1]).getTime();
        const curr = new Date(timestamps[i]).getTime();
        const hours = Math.max(0, (curr - prev) / 3600000);
        totalKwh += k * hours;
      }}
      const avgKw = timestamps.length ? sumKw / timestamps.length : 0;
      return {{ totalKwh, avgKw, peakKw }};
    }}

    function renderDashboard() {{
      const {{ period1, period2 }} = filterDataByPanelsAndPeriods();

      const m1 = computeMetrics(period1.timestamps, period1.kw);
      const m2 = computeMetrics(period2.timestamps, period2.kw);

      const cost1 = m1.totalKwh * dashboardData.price_per_kwh;
      const cost2 = m2.totalKwh * dashboardData.price_per_kwh;

      const energySavings = m1.totalKwh - m2.totalKwh;
      const energySavingsPct = m1.totalKwh > 0 ? (energySavings / m1.totalKwh * 100) : 0;
      const costSavings = cost1 - cost2;
      const costSavingsPct = cost1 > 0 ? (costSavings / cost1 * 100) : 0;

      const avgLoadChange = m2.avgKw - m1.avgKw;
      const avgLoadChangePct = m1.avgKw > 0 ? (avgLoadChange / m1.avgKw * 100) : 0;

      const peakReduction = m1.peakKw - m2.peakKw;
      const peakReductionPct = m1.peakKw > 0 ? (peakReduction / m1.peakKw * 100) : 0;

      const loadFactor1 = m1.peakKw > 0 ? (m1.avgKw / m1.peakKw * 100) : 0;
      const loadFactor2 = m2.peakKw > 0 ? (m2.avgKw / m2.peakKw * 100) : 0;
      const loadFactorImprovement = loadFactor2 - loadFactor1;

      const p1Days = period1.timestamps.length > 0 ?
        (new Date(period1.timestamps[period1.timestamps.length - 1]) - new Date(period1.timestamps[0])) / 86400000 + 1 : 1;
      const p2Days = period2.timestamps.length > 0 ?
        (new Date(period2.timestamps[period2.timestamps.length - 1]) - new Date(period2.timestamps[0])) / 86400000 + 1 : 1;

      const dailyEnergy1 = m1.totalKwh / p1Days;
      const dailyEnergy2 = m2.totalKwh / p2Days;

      updateSavingsSummary(energySavings, energySavingsPct, costSavings, costSavingsPct,
        peakReduction, peakReductionPct, loadFactorImprovement);
      updateMetricCards(m1, m2, cost1, cost2, loadFactor1, loadFactor2, dailyEnergy1, dailyEnergy2,
        energySavings, energySavingsPct, costSavings, costSavingsPct, avgLoadChange, avgLoadChangePct,
        peakReduction, peakReductionPct, loadFactorImprovement);
      renderCharts(period1, period2);
    }}

    function updateSavingsSummary(energySavings, energySavingsPct, costSavings, costSavingsPct,
      peakReduction, peakReductionPct, loadFactorImprovement) {{
      const hasSavings = energySavings > 0;
      const summaryEl = document.getElementById("savings-summary");

      if (hasSavings) {{
        summaryEl.style.background = "linear-gradient(135deg, var(--success), #059669)";
        document.getElementById("energy-savings").textContent =
          `${{energySavings.toFixed(2)}} kWh (${{energySavingsPct.toFixed(1)}}%)`;
        document.getElementById("cost-savings").textContent =
          `$${{costSavings.toFixed(2)}} (${{costSavingsPct.toFixed(1)}}%)`;
      }} else {{
        summaryEl.style.background = "linear-gradient(135deg, var(--warning), #d97706)";
        document.getElementById("energy-savings").textContent =
          `${{(-energySavings).toFixed(2)}} kWh increase`;
        document.getElementById("cost-savings").textContent =
          `$${{(-costSavings).toFixed(2)}} increase`;
      }}

      document.getElementById("peak-reduction").textContent =
        `${{peakReduction.toFixed(2)}} kW (${{peakReductionPct.toFixed(1)}}%)`;
      document.getElementById("load-factor-improvement").textContent =
        `${{loadFactorImprovement.toFixed(1)}}%`;
    }}

    function updateMetricCards(m1, m2, cost1, cost2, lf1, lf2, dailyE1, dailyE2,
      eSavings, eSavingsPct, cSavings, cSavingsPct, avgChange, avgChangePct,
      peakRed, peakRedPct, lfImprove) {{

      document.getElementById("p1-energy").textContent = `${{m1.totalKwh.toFixed(2)}} kWh`;
      document.getElementById("p2-energy").textContent = `${{m2.totalKwh.toFixed(2)}} kWh`;
      const eDelta = document.getElementById("energy-delta");
      eDelta.textContent = `${{(-eSavings).toFixed(2)}} kWh (${{(-eSavingsPct).toFixed(1)}}%)`;
      eDelta.className = eSavings > 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("energy-card").className =
        eSavings > 0 ? "comparison-card savings" : "comparison-card increase";

      document.getElementById("p1-cost").textContent = `$${{cost1.toFixed(2)}}`;
      document.getElementById("p2-cost").textContent = `$${{cost2.toFixed(2)}}`;
      const cDelta = document.getElementById("cost-delta");
      cDelta.textContent = `$${{(-cSavings).toFixed(2)}} (${{(-cSavingsPct).toFixed(1)}}%)`;
      cDelta.className = cSavings > 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("cost-card").className =
        cSavings > 0 ? "comparison-card savings" : "comparison-card increase";

      document.getElementById("p1-avg").textContent = `${{m1.avgKw.toFixed(2)}} kW`;
      document.getElementById("p2-avg").textContent = `${{m2.avgKw.toFixed(2)}} kW`;
      const avgDelta = document.getElementById("avg-delta");
      avgDelta.textContent = `${{avgChange.toFixed(2)}} kW (${{avgChangePct.toFixed(1)}}%)`;
      avgDelta.className = avgChange < 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("avg-load-card").className =
        avgChange < 0 ? "comparison-card savings" : "comparison-card increase";

      document.getElementById("p1-peak").textContent = `${{m1.peakKw.toFixed(2)}} kW`;
      document.getElementById("p2-peak").textContent = `${{m2.peakKw.toFixed(2)}} kW`;
      const peakDelta = document.getElementById("peak-delta");
      peakDelta.textContent = `${{(-peakRed).toFixed(2)}} kW (${{(-peakRedPct).toFixed(1)}}%)`;
      peakDelta.className = peakRed > 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("peak-load-card").className =
        peakRed > 0 ? "comparison-card savings" : "comparison-card increase";

      document.getElementById("p1-lf").textContent = `${{lf1.toFixed(1)}}%`;
      document.getElementById("p2-lf").textContent = `${{lf2.toFixed(1)}}%`;
      const lfDelta = document.getElementById("lf-delta");
      lfDelta.textContent = `${{lfImprove.toFixed(1)}} pts`;
      lfDelta.className = lfImprove > 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("load-factor-card").className =
        lfImprove > 0 ? "comparison-card savings" : "comparison-card increase";

      document.getElementById("p1-daily").textContent = `${{dailyE1.toFixed(2)}} kWh/day`;
      document.getElementById("p2-daily").textContent = `${{dailyE2.toFixed(2)}} kWh/day`;
      const dailyDelta = document.getElementById("daily-delta");
      const dailyChange = dailyE2 - dailyE1;
      dailyDelta.textContent = `${{dailyChange.toFixed(2)}} kWh/day`;
      dailyDelta.className = dailyChange < 0 ? "delta-value positive" : "delta-value negative";
      document.getElementById("daily-energy-card").className =
        dailyChange < 0 ? "comparison-card savings" : "comparison-card increase";
    }}

    function renderCharts(period1, period2) {{
      const layoutBase = {{
        margin: {{ t: 16, l: 60, r: 24, b: 50 }},
        paper_bgcolor: theme.card, plot_bgcolor: theme.card,
        font: {{ family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color: theme.ink, size: 12 }},
        hovermode: "x unified",
        hoverlabel: {{ bgcolor: theme.inkStrong, font: {{ color: "#ffffff" }} }}
      }};

      const p1Label = `Period 1 (${{document.getElementById("p1-start").value}} to ${{document.getElementById("p1-end").value}})`;
      const p2Label = `Period 2 (${{document.getElementById("p2-start").value}} to ${{document.getElementById("p2-end").value}})`;

      Plotly.newPlot("load-comparison-chart", [
        {{
          x: period1.timestamps, y: period1.kw, mode: "lines", name: p1Label,
          line: {{ color: theme.accent, width: 2.5, shape: "spline" }}
        }},
        {{
          x: period2.timestamps, y: period2.kw, mode: "lines", name: p2Label,
          line: {{ color: theme.accentDark, width: 2.5, shape: "spline" }}
        }}
      ], {{
        ...layoutBase,
        xaxis: {{ title: {{ text: "Time", font: {{ size: 12, weight: 600 }} }}, type: "date", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
        yaxis: {{ title: {{ text: "kW", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
        legend: {{ orientation: "h", y: -0.15 }}
      }}, {{ displaylogo: false, responsive: true }});

      const p1Hourly = computeHourlyAverage(period1.timestamps, period1.kw);
      const p2Hourly = computeHourlyAverage(period2.timestamps, period2.kw);

      Plotly.newPlot("hourly-comparison-chart", [
        {{
          x: p1Hourly.hours, y: p1Hourly.averages, mode: "lines+markers", name: "Period 1",
          line: {{ color: theme.accent, width: 2.5, shape: "spline" }},
          marker: {{ size: 8, color: theme.accent }}
        }},
        {{
          x: p2Hourly.hours, y: p2Hourly.averages, mode: "lines+markers", name: "Period 2",
          line: {{ color: theme.accentDark, width: 2.5, shape: "spline" }},
          marker: {{ size: 8, color: theme.accentDark }}
        }}
      ], {{
        ...layoutBase,
        xaxis: {{ title: {{ text: "Hour of Day", font: {{ size: 12, weight: 600 }} }}, dtick: 2, gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
        yaxis: {{ title: {{ text: "Average kW", font: {{ size: 12, weight: 600 }} }}, rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid, showline: true, linecolor: theme.grid }},
        legend: {{ orientation: "h", y: -0.15 }}
      }}, {{ displaylogo: false, responsive: true }});

      const p1Weekday = computeWeekdayAverage(period1.timestamps, period1.kw);
      const p2Weekday = computeWeekdayAverage(period2.timestamps, period2.kw);

      Plotly.newPlot("weekday-comparison-chart", [
        {{ x: p1Weekday.weekdays, y: p1Weekday.averages, type: "bar", name: "Period 1", marker: {{ color: theme.accent }} }},
        {{ x: p2Weekday.weekdays, y: p2Weekday.averages, type: "bar", name: "Period 2", marker: {{ color: theme.accentDark }} }}
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

    document.getElementById("apply-btn").addEventListener("click", renderDashboard);

    init();
  </script>
</body>
</html>
"""


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

    # Build comparison dashboard with full dataset
    output_path = build_comparison_dashboard(df, period1_dates, period2_dates, output_dir)

    print(f"Generated interactive comparison dashboard: {output_path}")
    print(f"Features: Panel selection + period adjustment in-browser")


if __name__ == "__main__":
    main()
