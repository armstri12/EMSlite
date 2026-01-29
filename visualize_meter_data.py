#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_meter_data.py

Generate multiple visualizations from EMS meter data in a wide CSV format.

Expected input format:
- Timestamp column named "Timestamp".
- One or more numeric meter columns (amps).

Config:
- Defaults loaded from visualization_config.json (override with --config).
- Config loader accepts JSON with optional // or /* */ comments and trailing commas.

Outputs:
- total_kw_timeseries.html
- total_kw_rolling_1h.html
- daily_hour_heatmap.html
- group_columns_plot.html
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
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
    "dashboard_logo_url": "",
    "visualizations": {
        "total_kw_timeseries": {
            "enabled": True,
            "output": "total_kw_timeseries.html",
        },
        "total_kw_rolling": {
            "enabled": True,
            "output": "total_kw_rolling_1h.html",
        },
        "daily_hour_heatmap": {
            "enabled": True,
            "output": "daily_hour_heatmap.html",
        },
        "group_columns_plot": {
            "enabled": True,
            "output": "group_columns_plot.html",
        },
        "dashboard": {
            "enabled": True,
            "output": "dashboard.html",
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
    parser = argparse.ArgumentParser(description="Visualize EMS meter data from a wide CSV.")
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
        "--rolling-window",
        default=None,
        help="Rolling window for the smoothed total kW plot (overrides config).",
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


def resolve_utility_meters(available: Iterable[str]) -> list[dict[str, object]]:
    resolved = []
    utility_meters = CONFIG.get("utility_meters") or []
    for idx, meter in enumerate(utility_meters):
        if not isinstance(meter, dict):
            continue
        name = str(meter.get("name") or f"Meter {idx + 1}")
        panels = meter.get("panels") or []
        if not isinstance(panels, list):
            panels = []
        resolved_panels = resolve_columns(available, panels, f"utility_meters[{name}]")
        resolved.append({"name": name, "panels": resolved_panels})
    return resolved


def zero_series(length: int) -> list[float]:
    return [0.0 for _ in range(length)]


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


def plot_total_kw(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = create_total_kw_fig(df)
    apply_plot_theme(fig)
    output_path = output_dir / CONFIG["visualizations"]["total_kw_timeseries"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


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


def plot_total_kw_rolling(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    fig = create_total_kw_rolling_fig(df, window)
    apply_plot_theme(fig)
    output_path = output_dir / CONFIG["visualizations"]["total_kw_rolling"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_daily_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = create_daily_hour_heatmap_fig(df)
    apply_plot_theme(fig)
    output_path = output_dir / CONFIG["visualizations"]["daily_hour_heatmap"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_group_columns(df: pd.DataFrame, output_dir: Path) -> Path | None:
    fig = create_group_columns_fig(df)
    if fig is None:
        return None
    apply_plot_theme(fig)
    output_path = output_dir / CONFIG["visualizations"]["group_columns_plot"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def create_total_kw_fig(df: pd.DataFrame) -> px.line:
    return px.line(df, x="Timestamp", y=TOTAL_KW_COLUMN_NAME, title="Total kW Over Time")


def create_total_kw_rolling_fig(df: pd.DataFrame, window: str) -> px.line:
    normalized_window = normalize_rolling_window(window)
    rolling = (
        df.set_index("Timestamp")[TOTAL_KW_COLUMN_NAME]
        .rolling(normalized_window)
        .mean()
        .reset_index()
    )
    return px.line(
        rolling,
        x="Timestamp",
        y=TOTAL_KW_COLUMN_NAME,
        title=f"Total kW (Rolling {normalized_window})",
    )


def create_daily_hour_heatmap_fig(df: pd.DataFrame) -> px.imshow:
    temp = df.copy()
    temp["Date"] = temp["Timestamp"].dt.date
    temp["Hour"] = temp["Timestamp"].dt.hour
    pivot = temp.pivot_table(index="Hour", columns="Date", values=TOTAL_KW_COLUMN_NAME, aggfunc="mean")
    return px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        title="Average Total kW by Hour and Day",
        labels={"color": "kW"},
        color_continuous_scale=["#cbd0cc", "#2d363a", "#c4262e"],
    )


def create_group_columns_fig(df: pd.DataFrame) -> px.line | None:
    group_columns = [name for name in CONFIG["combo_columns"].keys() if name in df.columns]
    if not group_columns:
        return None
    plot_df = df[["Timestamp", *group_columns]].dropna(subset=["Timestamp"]).copy()
    if plot_df.empty:
        return None
    long_df = plot_df.melt(id_vars="Timestamp", var_name="Group", value_name="kW")
    group_count = long_df["Group"].nunique()
    fig = px.line(
        long_df,
        x="Timestamp",
        y="kW",
        facet_row="Group",
        title="Group Columns (kW)",
    )
    fig.update_layout(showlegend=False, height=max(300, 250 * group_count))
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    fig.update_yaxes(matches=None)
    return fig


def apply_plot_theme(fig: go.Figure) -> None:
    fig.update_layout(
        font={"family": PLOT_THEME["font_family"], "color": PLOT_THEME["ink"]},
        paper_bgcolor=PLOT_THEME["card"],
        plot_bgcolor=PLOT_THEME["card"],
        colorway=PLOT_THEME["colorway"],
    )
    fig.update_xaxes(gridcolor=PLOT_THEME["grid"], zerolinecolor=PLOT_THEME["grid"])
    fig.update_yaxes(gridcolor=PLOT_THEME["grid"], zerolinecolor=PLOT_THEME["grid"])


def compute_energy_metrics(df: pd.DataFrame) -> dict[str, float]:
    if TOTAL_KW_COLUMN_NAME not in df.columns:
        return {}
    ordered = df.sort_values("Timestamp").dropna(subset=["Timestamp"])
    deltas = ordered["Timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    total_kwh = (ordered[TOTAL_KW_COLUMN_NAME] * deltas).sum()
    return {
        "total_kwh": float(total_kwh),
        "average_kw": float(ordered[TOTAL_KW_COLUMN_NAME].mean()),
        "peak_kw": float(ordered[TOTAL_KW_COLUMN_NAME].max()),
    }


def build_dashboard(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    ordered = df.sort_values("Timestamp")
    timestamps = ordered["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    zero_values = zero_series(len(timestamps))
    total_kw = ordered.get(TOTAL_KW_COLUMN_NAME, pd.Series()).fillna(0).tolist()
    group_columns = [name for name in CONFIG["combo_columns"].keys() if name in ordered.columns]
    group_series = {name: ordered[name].fillna(0).tolist() for name in group_columns}
    panel_columns = meter_columns(ordered.columns)
    panel_series = {
        name: amps_to_kw(ordered[name].fillna(0)).fillna(0).tolist() for name in panel_columns
    }
    group_definitions = []
    for group_name, group_panels in CONFIG["combo_columns"].items():
        resolved_panels = resolve_columns(ordered.columns, group_panels, f"combo_columns[{group_name}]")
        group_definitions.append({"name": group_name, "panels": resolved_panels})
    meter_definitions = resolve_utility_meters(ordered.columns)
    meter_series = {}
    for meter in meter_definitions:
        panels = meter["panels"]
        if panels:
            amps_total = ordered[panels].fillna(0).sum(axis=1)
            meter_series[meter["name"]] = amps_to_kw(amps_total).fillna(0).tolist()
        else:
            meter_series[meter["name"]] = zero_values
    price_per_kwh = float(CONFIG["price_per_kwh"])
    rolling_hours = parse_window_to_hours(window)

    data_payload = {
        "timestamps": timestamps,
        "total_kw": total_kw,
        "group_series": group_series,
        "panel_series": panel_series,
        "panel_names": panel_columns,
        "group_definitions": group_definitions,
        "utility_meters": meter_definitions,
        "meter_series": meter_series,
        "rolling_hours": rolling_hours,
        "price_per_kwh": price_per_kwh,
    }
    logo_url = CONFIG.get("dashboard_logo_url") or ""

    group_card = ""
    group_script = ""
    if group_columns:
        group_card = """
          <div class="chart-card">
            <div class="chart-title">Group Loads</div>
            <div id="group-load-chart" class="chart"></div>
          </div>
        """
        group_script = """
          function renderGroupChart(data) {
            const traces = Object.entries(data.groupSeries).map(([name, values]) => ({
              x: data.timestamps,
              y: values,
              mode: "lines",
              name
            }));
            Plotly.newPlot("group-load-chart", traces, {
              margin: { t: 16, l: 50, r: 24, b: 40 },
              legend: { orientation: "h" },
              xaxis: { title: "Timestamp", type: "date", gridcolor: theme.grid, zerolinecolor: theme.grid },
              yaxis: { title: "kW", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid },
              paper_bgcolor: theme.card,
              plot_bgcolor: theme.card,
              font: { family: "Calibri, Segoe UI, Helvetica Neue, Arial, sans-serif", color: theme.ink },
              colorway: theme.series
            }, { displaylogo: false, responsive: true });
          }
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>EMS Energy Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
          :root {{
            --bg: #cbd0cc;
            --card: #ffffff;
            --ink: #2d363a;
            --ink-strong: #000000;
            --muted: #5b6468;
            --accent: #c4262e;
            --accent-soft: rgba(196, 38, 46, 0.12);
            --outline: rgba(45, 54, 58, 0.12);
            --pill: rgba(45, 54, 58, 0.08);
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Calibri", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
            color: var(--ink);
          }}
          .top-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 24px 40px 12px;
          }}
          .brand {{
            display: flex;
            align-items: center;
            gap: 16px;
          }}
          .logo-wrapper {{
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: var(--pill);
            border: 1px dashed var(--outline);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
          }}
          .logo-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
          }}
          .logo-placeholder {{
            font-size: 11px;
            color: var(--muted);
            text-align: center;
            padding: 4px;
          }}
          .logo-controls {{
            display: flex;
            flex-direction: column;
            gap: 6px;
          }}
          .logo-controls label {{
            font-size: 11px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }}
          .logo-controls input[type="file"] {{
            font-size: 12px;
          }}
          .header {{
            padding: 8px 40px 16px;
          }}
          .title {{
            font-size: 32px;
            font-weight: 700;
            margin: 0;
            color: var(--ink-strong);
          }}
          .subtitle {{
            margin-top: 6px;
            color: var(--muted);
            font-size: 14px;
          }}
          .nav-pills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
          }}
          .nav-pill {{
            padding: 8px 14px;
            border-radius: 999px;
            background: var(--pill);
            color: var(--ink);
            font-size: 13px;
            font-weight: 600;
            text-decoration: none;
            border: 1px solid transparent;
          }}
          .nav-pill:hover {{
            border-color: var(--outline);
          }}
          .filters {{
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
          }}
          .filters label {{
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }}
          .filters input {{
            border: 1px solid var(--outline);
            border-radius: 10px;
            padding: 8px 10px;
            font-size: 14px;
            color: var(--ink);
            background: var(--card);
          }}
          .filters button {{
            border: none;
            border-radius: 10px;
            padding: 8px 14px;
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            background: var(--accent);
            cursor: pointer;
            box-shadow: 0 8px 16px rgba(196, 38, 46, 0.25);
          }}
          .filters button.secondary {{
            background: #2d363a;
          }}
          .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            padding: 0 40px 24px;
          }}
          .meter-stat-grid {{
            margin-top: -8px;
            padding-top: 0;
          }}
          .stat-card {{
            background: var(--card);
            border-radius: 16px;
            padding: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--outline);
          }}
          .stat-card.meter-highlight {{
            border: 1px solid rgba(196, 38, 46, 0.3);
            background: linear-gradient(135deg, rgba(196, 38, 46, 0.1), rgba(45, 54, 58, 0.05));
          }}
          .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
          }}
          .stat-value {{
            font-size: 24px;
            font-weight: 600;
            margin-top: 8px;
          }}
          .stat-hint {{
            margin-top: 6px;
            font-size: 12px;
            color: var(--muted);
          }}
          .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(420px, 1fr));
            gap: 20px;
            padding: 0 40px 40px;
          }}
          .section-anchor {{
            scroll-margin-top: 80px;
          }}
          .usage-section {{
            padding: 0 40px 20px;
          }}
          .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
          }}
          .meter-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
          }}
          .meter-card {{
            background: linear-gradient(135deg, #2d363a, #000000);
            color: #fff;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.2);
          }}
          .meter-title {{
            font-size: 14px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.8;
          }}
          .meter-value {{
            font-size: 24px;
            font-weight: 600;
            margin-top: 10px;
          }}
          .meter-sub {{
            font-size: 13px;
            margin-top: 6px;
            opacity: 0.85;
          }}
          .usage-card {{
            background: linear-gradient(135deg, #c4262e, #2d363a);
          }}
          .chart-card {{
            background: var(--card);
            border-radius: 18px;
            padding: 12px 12px 4px;
            box-shadow: var(--shadow);
            border: 1px solid var(--outline);
          }}
          .chart-title {{
            padding: 10px 12px 0;
            font-weight: 600;
            color: var(--ink);
          }}
          .chart {{
            min-height: 380px;
          }}
          .chart-card .plotly-graph-div {{
            width: 100% !important;
          }}
          .chart-card.full-width {{
            grid-column: 1 / -1;
          }}
          .panel-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            padding: 6px 12px 0;
          }}
          .panel-controls label {{
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }}
          .panel-controls select {{
            min-width: 220px;
            padding: 6px 10px;
            border-radius: 10px;
            border: 1px solid var(--outline);
            background: #fff;
            font-size: 13px;
            color: var(--ink);
          }}
          .footer {{
            padding: 20px 40px 32px;
            color: var(--muted);
            font-size: 12px;
          }}
          @media (max-width: 980px) {{
            .top-bar {{
              flex-direction: column;
              align-items: flex-start;
            }}
            .charts-grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="top-bar">
          <div class="brand">
            <div class="logo-wrapper">
              <img id="logo-image" src="{logo_url}" alt="Logo" style="display: none;" />
              <div class="logo-placeholder" id="logo-placeholder">Your Logo</div>
            </div>
            <div>
              <h1 class="title">EMS Energy Dashboard</h1>
              <div class="subtitle">Interactive totals using ${price_per_kwh:.2f} / kWh</div>
            </div>
          </div>
          <div class="logo-controls">
            <label for="logo-input">Logo upload</label>
            <input type="file" id="logo-input" accept="image/*" />
          </div>
        </div>
        <div class="header">
          <div class="nav-pills">
            <a class="nav-pill" href="#kpi-section">KPIs</a>
            <a class="nav-pill" href="#usage-section">Usage</a>
            <a class="nav-pill" href="#charts-section">Charts</a>
            <a class="nav-pill" href="#panels-section">Panel Trends</a>
          </div>
          <div class="filters">
            <div>
              <label for="start-date">Start Date</label>
              <input type="date" id="start-date" />
            </div>
            <div>
              <label for="end-date">End Date</label>
              <input type="date" id="end-date" />
            </div>
            <button id="apply-filters">Apply Dates</button>
            <button id="reset-filters" class="secondary">Reset</button>
          </div>
        </div>
        <div class="stats-grid section-anchor" id="kpi-section">
          <div class="stat-card">
            <div class="stat-label">Total Energy</div>
            <div class="stat-value" id="total-energy">0.00 kWh</div>
            <div class="stat-hint">Integrated from total load</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Estimated Cost</div>
            <div class="stat-value" id="total-cost">$0.00</div>
            <div class="stat-hint">Rate: ${price_per_kwh:.2f} / kWh</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Average Load</div>
            <div class="stat-value" id="average-load">0.00 kW</div>
            <div class="stat-hint">Mean across timestamps</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Peak Load</div>
            <div class="stat-value" id="peak-load">0.00 kW</div>
            <div class="stat-hint">Highest observed value</div>
          </div>
        </div>
        <div class="stats-grid meter-stat-grid" id="meter-stat-grid" style="display: none;"></div>
        <div class="usage-section section-anchor" id="usage-section" style="display: none;">
          <div class="section-title">Usage by Group</div>
          <div class="meter-grid" id="usage-cards"></div>
        </div>
        <div class="charts-grid section-anchor" id="charts-section">
          <div class="chart-card">
            <div class="chart-title">Smoothed Load</div>
            <div id="rolling-load-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Load Heatmap</div>
            <div id="heatmap-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Average Hourly Profile</div>
            <div id="hourly-profile-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Weekday Load Mix</div>
            <div id="weekday-profile-chart" class="chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Daily Energy</div>
            <div id="daily-energy-chart" class="chart"></div>
          </div>
          {group_card}
          <div class="chart-card full-width section-anchor" id="panels-section" style="display: none;">
            <div class="chart-title">Panel Trends</div>
            <div class="panel-controls">
              <label for="panel-selector">Panels</label>
              <select id="panel-selector" multiple></select>
            </div>
            <div id="panel-series-chart" class="chart"></div>
          </div>
        </div>
        <div class="footer">Primary palette: #2d363a &amp; #c4262e · Secondary palette: #000000 &amp; #cbd0cc</div>
        <script>
          const dashboardData = {json.dumps(data_payload)};
          const logoUrl = {json.dumps(logo_url)};
          const theme = {{
            ink: "#2d363a",
            inkStrong: "#000000",
            muted: "#5b6468",
            card: "#ffffff",
            grid: "rgba(45, 54, 58, 0.12)",
            accent: "#c4262e",
            accentDark: "#2d363a",
            accentNeutral: "#cbd0cc",
            series: ["#c4262e", "#2d363a", "#000000", "#6b7280", "#cbd0cc"]
          }};

          const startInput = document.getElementById("start-date");
          const endInput = document.getElementById("end-date");
          const applyButton = document.getElementById("apply-filters");
          const resetButton = document.getElementById("reset-filters");
          const logoInput = document.getElementById("logo-input");
          const logoImage = document.getElementById("logo-image");
          const logoPlaceholder = document.getElementById("logo-placeholder");

          function applyLogo(src) {{
            if (!src) {{
              logoImage.style.display = "none";
              logoPlaceholder.style.display = "block";
              return;
            }}
            logoImage.src = src;
            logoImage.style.display = "block";
            logoPlaceholder.style.display = "none";
          }}

          function toDateOnly(date) {{
            return date.toISOString().slice(0, 10);
          }}

          function initDateInputs() {{
            const timestamps = dashboardData.timestamps;
            if (!timestamps.length) {{
              return;
            }}
            const dates = timestamps.map((ts) => new Date(ts));
            const minDate = new Date(Math.min(...dates));
            const maxDate = new Date(Math.max(...dates));
            startInput.min = toDateOnly(minDate);
            startInput.max = toDateOnly(maxDate);
            endInput.min = toDateOnly(minDate);
            endInput.max = toDateOnly(maxDate);
            startInput.value = toDateOnly(minDate);
            endInput.value = toDateOnly(maxDate);
          }}

          function filterData() {{
            const startDate = startInput.value ? new Date(`${{startInput.value}}T00:00:00Z`) : null;
            const endDate = endInput.value ? new Date(`${{endInput.value}}T23:59:59Z`) : null;
            const filtered = {{
              timestamps: [],
              totalKw: [],
              groupSeries: Object.fromEntries(Object.keys(dashboardData.group_series).map((key) => [key, []])),
              meterSeries: Object.fromEntries(Object.keys(dashboardData.meter_series).map((key) => [key, []])),
              panelSeries: Object.fromEntries(Object.keys(dashboardData.panel_series || {{}}).map((key) => [key, []]))
            }};
            dashboardData.timestamps.forEach((ts, idx) => {{
              const date = new Date(ts);
              if (startDate && date < startDate) {{
                return;
              }}
              if (endDate && date > endDate) {{
                return;
              }}
              filtered.timestamps.push(ts);
              filtered.totalKw.push(dashboardData.total_kw[idx]);
              Object.keys(dashboardData.group_series).forEach((key) => {{
                filtered.groupSeries[key].push(dashboardData.group_series[key][idx]);
              }});
              Object.keys(dashboardData.meter_series).forEach((key) => {{
                filtered.meterSeries[key].push(dashboardData.meter_series[key][idx]);
              }});
              Object.keys(dashboardData.panel_series || {{}}).forEach((key) => {{
                filtered.panelSeries[key].push(dashboardData.panel_series[key][idx]);
              }});
            }});
            return filtered;
          }}

          function computeMetrics(timestamps, totalKw) {{
            let totalKwh = 0;
            let peakKw = 0;
            let sumKw = 0;
            for (let i = 0; i < timestamps.length; i++) {{
              const kw = totalKw[i] ?? 0;
              sumKw += kw;
              if (kw > peakKw) {{
                peakKw = kw;
              }}
              if (i === 0) {{
                continue;
              }}
              const prev = new Date(timestamps[i - 1]).getTime();
              const curr = new Date(timestamps[i]).getTime();
              const hours = Math.max(0, (curr - prev) / 3600000);
              totalKwh += kw * hours;
            }}
            const avgKw = timestamps.length ? sumKw / timestamps.length : 0;
            return {{ totalKwh, avgKw, peakKw }};
          }}

          function rollingMean(timestamps, values, windowHours) {{
            const windowMs = windowHours * 3600000;
            const result = [];
            let startIndex = 0;
            let sum = 0;
            for (let i = 0; i < timestamps.length; i++) {{
              const currentTime = new Date(timestamps[i]).getTime();
              sum += values[i] ?? 0;
              while (currentTime - new Date(timestamps[startIndex]).getTime() > windowMs) {{
                sum -= values[startIndex] ?? 0;
                startIndex += 1;
              }}
              const count = i - startIndex + 1;
              result.push(count ? sum / count : 0);
            }}
            return result;
          }}

          function buildHeatmap(timestamps, totalKw) {{
            const buckets = {{}};
            timestamps.forEach((ts, idx) => {{
              const dateObj = new Date(ts);
              const dateKey = dateObj.toISOString().slice(0, 10);
              const hour = dateObj.getUTCHours();
              if (!buckets[dateKey]) {{
                buckets[dateKey] = {{}};
              }}
              if (!buckets[dateKey][hour]) {{
                buckets[dateKey][hour] = {{ sum: 0, count: 0 }};
              }}
              buckets[dateKey][hour].sum += totalKw[idx] ?? 0;
              buckets[dateKey][hour].count += 1;
            }});
            const dates = Object.keys(buckets).sort();
            const hours = Array.from({{ length: 24 }}, (_, i) => i);
            const z = hours.map((hour) =>
              dates.map((date) => {{
                const bucket = buckets[date][hour];
                if (!bucket) {{
                  return null;
                }}
                return bucket.sum / bucket.count;
              }})
            );
            return {{ dates, hours, z }};
          }}

          function buildHourlyProfile(timestamps, totalKw) {{
            const sums = Array(24).fill(0);
            const counts = Array(24).fill(0);
            timestamps.forEach((ts, idx) => {{
              const hour = new Date(ts).getUTCHours();
              sums[hour] += totalKw[idx] ?? 0;
              counts[hour] += 1;
            }});
            const averages = sums.map((sum, idx) => (counts[idx] ? sum / counts[idx] : 0));
            return {{ hours: Array.from({{ length: 24 }}, (_, i) => i), averages }};
          }}

          function buildWeekdayProfile(timestamps, totalKw) {{
            const sums = Array(7).fill(0);
            const counts = Array(7).fill(0);
            timestamps.forEach((ts, idx) => {{
              const weekday = new Date(ts).getUTCDay();
              sums[weekday] += totalKw[idx] ?? 0;
              counts[weekday] += 1;
            }});
            const averages = sums.map((sum, idx) => (counts[idx] ? sum / counts[idx] : 0));
            return {{
              weekdays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
              averages
            }};
          }}

          function buildDailyEnergy(timestamps, totalKw) {{
            const energyByDate = {{}};
            for (let i = 0; i < timestamps.length; i++) {{
              if (i === timestamps.length - 1) {{
                break;
              }}
              const start = new Date(timestamps[i]);
              const end = new Date(timestamps[i + 1]);
              const hours = Math.max(0, (end - start) / 3600000);
              const dateKey = start.toISOString().slice(0, 10);
              if (!energyByDate[dateKey]) {{
                energyByDate[dateKey] = 0;
              }}
              energyByDate[dateKey] += (totalKw[i] ?? 0) * hours;
            }}
            const dates = Object.keys(energyByDate).sort();
            const values = dates.map((date) => energyByDate[date]);
            return {{ dates, values }};
          }}

          function buildWeekendShapes(timestamps) {{
            if (!timestamps.length) {{
              return [];
            }}
            const shapes = [];
            let weekendStart = null;
            for (let i = 0; i < timestamps.length; i++) {{
              const ts = timestamps[i];
              const day = new Date(ts).getUTCDay();
              const isWeekend = day === 0 || day === 6;
              if (isWeekend && weekendStart === null) {{
                weekendStart = ts;
              }}
              if (!isWeekend && weekendStart !== null) {{
                shapes.push({{
                  type: "rect",
                  xref: "x",
                  yref: "paper",
                  x0: weekendStart,
                  x1: ts,
                  y0: 0,
                  y1: 1,
                  line: {{ width: 0 }},
                  fillcolor: "rgba(99, 102, 241, 0.14)"
                }});
                weekendStart = null;
              }}
            }}
            if (weekendStart !== null) {{
              shapes.push({{
                type: "rect",
                xref: "x",
                yref: "paper",
                x0: weekendStart,
                x1: timestamps[timestamps.length - 1],
                y0: 0,
                y1: 1,
                line: {{ width: 0 }},
                fillcolor: "rgba(99, 102, 241, 0.14)"
              }});
            }}
            return shapes;
          }}

          function renderCharts(data) {{
            const rollingValues = rollingMean(data.timestamps, data.totalKw, dashboardData.rolling_hours);
            const rollingTrace = {{
              x: data.timestamps,
              y: rollingValues,
              mode: "lines",
              line: {{ color: theme.accent, width: 3 }}
            }};
            const layoutBase = {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Timestamp", type: "date", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "kW", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              paper_bgcolor: theme.card,
              plot_bgcolor: theme.card,
              font: {{ family: "Calibri, Segoe UI, Helvetica Neue, Arial, sans-serif", color: theme.ink }},
              colorway: theme.series
            }};
            Plotly.newPlot("rolling-load-chart", [rollingTrace], {{
              ...layoutBase
            }}, {{ displaylogo: false, responsive: true }});

            const heatmap = buildHeatmap(data.timestamps, data.totalKw);
            Plotly.newPlot("heatmap-chart", [{{
              x: heatmap.dates,
              y: heatmap.hours,
              z: heatmap.z,
              type: "heatmap",
              colorscale: [
                [0, "#cbd0cc"],
                [0.5, "#2d363a"],
                [1, "#c4262e"]
              ],
              zsmooth: "best",
              connectgaps: true,
              colorbar: {{ title: "kW" }}
            }}], {{
              ...layoutBase,
              xaxis: {{ title: "Date", type: "category", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "Hour", autorange: "reversed", gridcolor: theme.grid, zerolinecolor: theme.grid }}
            }}, {{ displaylogo: false, responsive: true }});

            const hourlyProfile = buildHourlyProfile(data.timestamps, data.totalKw);
            const hourlyTrace = {{
              x: hourlyProfile.hours,
              y: hourlyProfile.averages,
              mode: "lines+markers",
              line: {{ color: theme.accentDark, width: 3 }},
              marker: {{ size: 6 }}
            }};
            Plotly.newPlot("hourly-profile-chart", [hourlyTrace], {{
              ...layoutBase,
              xaxis: {{ title: "Hour (UTC)", dtick: 1, gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "Average kW", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid }}
            }}, {{ displaylogo: false, responsive: true }});

            const weekdayProfile = buildWeekdayProfile(data.timestamps, data.totalKw);
            const weekdayTrace = {{
              x: weekdayProfile.weekdays,
              y: weekdayProfile.averages,
              type: "bar",
              marker: {{ color: theme.accent }}
            }};
            Plotly.newPlot("weekday-profile-chart", [weekdayTrace], {{
              ...layoutBase,
              xaxis: {{ title: "Day of Week", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "Average kW", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid }}
            }}, {{ displaylogo: false, responsive: true }});

            const dailyEnergy = buildDailyEnergy(data.timestamps, data.totalKw);
            const dailyEnergyTrace = {{
              x: dailyEnergy.dates,
              y: dailyEnergy.values,
              type: "bar",
              marker: {{ color: theme.accentDark }}
            }};
            Plotly.newPlot("daily-energy-chart", [dailyEnergyTrace], {{
              ...layoutBase,
              xaxis: {{ title: "Date", type: "category", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "Energy (kWh)", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid }}
            }}, {{ displaylogo: false, responsive: true }});

            if (Object.keys(data.groupSeries).length) {{
              renderGroupChart(data);
            }}

            renderPanelChart(data);
          }}

          function updateMetrics(data) {{
            const metrics = computeMetrics(data.timestamps, data.totalKw);
            document.getElementById("total-energy").textContent = `${{metrics.totalKwh.toFixed(2)}} kWh`;
            document.getElementById("average-load").textContent = `${{metrics.avgKw.toFixed(2)}} kW`;
            document.getElementById("peak-load").textContent = `${{metrics.peakKw.toFixed(2)}} kW`;
            const totalCost = metrics.totalKwh * dashboardData.price_per_kwh;
            document.getElementById("total-cost").textContent = `$${{totalCost.toFixed(2)}}`;
          }}

          function updateMeterCards(data) {{
            if (!dashboardData.utility_meters.length) {{
              return;
            }}
            const container = document.getElementById("meter-stat-grid");
            if (!container) {{
              return;
            }}
            container.style.display = "grid";
            if (!container.dataset.initialized) {{
              container.innerHTML = dashboardData.utility_meters
                .map((meter, idx) => {{
                  const panelList = (meter.panels || []).join(", ");
                  return `
                    <div class="stat-card meter-highlight">
                      <div class="stat-label">${{meter.name}} Energy</div>
                      <div class="stat-value" id="meter-energy-${{idx}}">0.00 kWh</div>
                      <div class="stat-hint" id="meter-cost-${{idx}}">$0.00</div>
                      <div class="stat-hint">Panels: ${{panelList || "None"}}</div>
                    </div>
                  `;
                }})
                .join("");
              container.dataset.initialized = "true";
            }}
            dashboardData.utility_meters.forEach((meter, idx) => {{
              const series = data.meterSeries[meter.name] || [];
              const metrics = computeMetrics(data.timestamps, series);
              const energyEl = document.getElementById(`meter-energy-${{idx}}`);
              const costEl = document.getElementById(`meter-cost-${{idx}}`);
              if (energyEl) {{
                energyEl.textContent = `${{metrics.totalKwh.toFixed(2)}} kWh`;
              }}
              if (costEl) {{
                const totalCost = metrics.totalKwh * dashboardData.price_per_kwh;
                costEl.textContent = `$${{totalCost.toFixed(2)}}`;
              }}
            }});
          }}

          function updateUsageCards(data) {{
            if (!dashboardData.group_definitions.length) {{
              return;
            }}
            const section = document.getElementById("usage-section");
            const container = document.getElementById("usage-cards");
            if (!section || !container) {{
              return;
            }}
            section.style.display = "block";
            if (!container.dataset.initialized) {{
              container.innerHTML = dashboardData.group_definitions
                .map((group, idx) => {{
                  const panelList = (group.panels || []).join(", ");
                  return `
                    <div class="meter-card usage-card">
                      <div class="meter-title">${{group.name}}</div>
                      <div class="meter-value" id="group-energy-${{idx}}">0.00 kWh</div>
                      <div class="meter-sub" id="group-cost-${{idx}}">$0.00</div>
                      <div class="meter-sub">Panels: ${{panelList || "None"}}</div>
                    </div>
                  `;
                }})
                .join("");
              container.dataset.initialized = "true";
            }}
            dashboardData.group_definitions.forEach((group, idx) => {{
              const series = data.groupSeries[group.name]
                || Array(data.timestamps.length).fill(0);
              const metrics = computeMetrics(data.timestamps, series);
              const energyEl = document.getElementById(`group-energy-${{idx}}`);
              const costEl = document.getElementById(`group-cost-${{idx}}`);
              if (energyEl) {{
                energyEl.textContent = `${{metrics.totalKwh.toFixed(2)}} kWh`;
              }}
              if (costEl) {{
                const totalCost = metrics.totalKwh * dashboardData.price_per_kwh;
                costEl.textContent = `$${{totalCost.toFixed(2)}}`;
              }}
            }});
          }}

          function initPanelSelector() {{
            if (!dashboardData.panel_names || !dashboardData.panel_names.length) {{
              return;
            }}
            const selector = document.getElementById("panel-selector");
            const card = document.getElementById("panels-section");
            if (!selector || !card) {{
              return;
            }}
            card.style.display = "block";
            if (selector.options.length) {{
              return;
            }}
            dashboardData.panel_names.forEach((panel) => {{
              const option = document.createElement("option");
              option.value = panel;
              option.textContent = panel;
              option.selected = true;
              selector.appendChild(option);
            }});
            selector.addEventListener("change", () => {{
              renderPanelChart(filterData());
            }});
          }}

          function getSelectedPanels() {{
            const selector = document.getElementById("panel-selector");
            if (!selector) {{
              return [];
            }}
            return Array.from(selector.selectedOptions).map((option) => option.value);
          }}

          function renderPanelChart(data) {{
            if (!dashboardData.panel_names || !dashboardData.panel_names.length) {{
              return;
            }}
            const selected = getSelectedPanels();
            const panelsToShow = selected.length ? selected : dashboardData.panel_names;
            const traces = panelsToShow.map((panel) => ({{
              x: data.timestamps,
              y: (data.panelSeries && data.panelSeries[panel]) || [],
              mode: "lines",
              name: panel
            }}));
            const weekendShapes = buildWeekendShapes(data.timestamps);
            Plotly.newPlot("panel-series-chart", traces, {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              legend: {{ orientation: "h" }},
              xaxis: {{ title: "Timestamp", type: "date", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              yaxis: {{ title: "kW", rangemode: "tozero", gridcolor: theme.grid, zerolinecolor: theme.grid }},
              shapes: weekendShapes,
              paper_bgcolor: theme.card,
              plot_bgcolor: theme.card,
              font: {{ family: "Calibri, Segoe UI, Helvetica Neue, Arial, sans-serif", color: theme.ink }},
              colorway: theme.series
            }}, {{ displaylogo: false, responsive: true }});
          }}

          function renderDashboard() {{
            const filtered = filterData();
            renderCharts(filtered);
            updateMetrics(filtered);
            updateMeterCards(filtered);
            updateUsageCards(filtered);
          }}

          applyButton.addEventListener("click", renderDashboard);
          resetButton.addEventListener("click", () => {{
            initDateInputs();
            renderDashboard();
          }});

          initDateInputs();
          initPanelSelector();
          renderDashboard();
          applyLogo(logoUrl);
          if (logoInput) {{
            logoInput.addEventListener("change", (event) => {{
              const file = event.target.files && event.target.files[0];
              if (!file) {{
                return;
              }}
              const reader = new FileReader();
              reader.onload = (loadEvent) => {{
                applyLogo(loadEvent.target.result);
              }};
              reader.readAsDataURL(file);
            }});
          }}
          {group_script}
        </script>
      </body>
    </html>
    """
    output_path = output_dir / CONFIG["visualizations"]["dashboard"]["output"]
    output_path.write_text(html, encoding="utf-8")
    return output_path


def describe_combo_columns(meters: list[str]) -> None:
    total_sources = resolve_columns(meters, CONFIG["total_amps_sources"], "total_amps_sources")
    total_display = total_sources if total_sources else []
    if CONFIG["total_amps_sources"] is None:
        total_display = meters
    print("Combo column definitions:")
    print(f"- {TOTAL_AMPS_COLUMN_NAME}/{TOTAL_KW_COLUMN_NAME}: {total_display}")
    for name, columns in CONFIG["combo_columns"].items():
        resolved = resolve_columns(meters, columns, f"combo_columns[{name}]")
        print(f"- {name}: {resolved}")


def main() -> None:
    args = parse_args()
    global CONFIG
    CONFIG = load_config(Path(args.config))
    input_path = Path(args.input or CONFIG["input_file"])
    output_dir = Path(args.output_dir or CONFIG["output_dir"])
    rolling_window = args.rolling_window or CONFIG["rolling_window"]
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    meters = meter_columns(df.columns)
    if not meters:
        raise ValueError("No meter columns found in the input file.")

    describe_combo_columns(meters)
    df = add_usage_columns(df, meters)

    outputs = []
    if (
        TOTAL_KW_COLUMN_NAME in df.columns
        and CONFIG["visualizations"]["total_kw_timeseries"]["enabled"]
    ):
        outputs.append(plot_total_kw(df, output_dir))
    if TOTAL_KW_COLUMN_NAME in df.columns and CONFIG["visualizations"]["total_kw_rolling"]["enabled"]:
        outputs.append(plot_total_kw_rolling(df, output_dir, rolling_window))
    if TOTAL_KW_COLUMN_NAME in df.columns and CONFIG["visualizations"]["daily_hour_heatmap"]["enabled"]:
        outputs.append(plot_daily_hour_heatmap(df, output_dir))
    if CONFIG["visualizations"]["group_columns_plot"]["enabled"]:
        group_plot = plot_group_columns(df, output_dir)
        if group_plot:
            outputs.append(group_plot)
    if CONFIG["visualizations"]["dashboard"]["enabled"]:
        outputs.append(build_dashboard(df, output_dir, rolling_window))

    print("Generated visualizations:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
