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
import plotly.io as pio

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
        if not resolved_panels:
            continue
        resolved.append({"name": name, "panels": resolved_panels})
    return resolved


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
    output_path = output_dir / CONFIG["visualizations"]["total_kw_rolling"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_daily_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = create_daily_hour_heatmap_fig(df)
    output_path = output_dir / CONFIG["visualizations"]["daily_hour_heatmap"]["output"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_group_columns(df: pd.DataFrame, output_dir: Path) -> Path | None:
    fig = create_group_columns_fig(df)
    if fig is None:
        return None
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
    total_kw = ordered.get(TOTAL_KW_COLUMN_NAME, pd.Series()).fillna(0).tolist()
    group_columns = [name for name in CONFIG["combo_columns"].keys() if name in ordered.columns]
    group_series = {name: ordered[name].fillna(0).tolist() for name in group_columns}
    meter_definitions = resolve_utility_meters(ordered.columns)
    meter_series = {}
    for meter in meter_definitions:
        panels = meter["panels"]
        amps_total = ordered[panels].fillna(0).sum(axis=1)
        meter_series[meter["name"]] = amps_to_kw(amps_total).fillna(0).tolist()
    price_per_kwh = float(CONFIG["price_per_kwh"])
    rolling_hours = parse_window_to_hours(window)

    data_payload = {
        "timestamps": timestamps,
        "total_kw": total_kw,
        "group_series": group_series,
        "utility_meters": meter_definitions,
        "meter_series": meter_series,
        "rolling_hours": rolling_hours,
        "price_per_kwh": price_per_kwh,
    }

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
              xaxis: { title: "Timestamp", type: "date" },
              yaxis: { title: "kW", rangemode: "tozero" },
              template: "plotly_white"
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
            --bg: #f5f7fb;
            --card: #ffffff;
            --ink: #0f172a;
            --muted: #64748b;
            --accent: #2563eb;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            background: var(--bg);
            color: var(--ink);
          }}
          .header {{
            padding: 32px 40px 16px;
          }}
          .title {{
            font-size: 32px;
            font-weight: 700;
            margin: 0;
          }}
          .subtitle {{
            margin-top: 6px;
            color: var(--muted);
            font-size: 14px;
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
            border: 1px solid #e2e8f0;
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
          }}
          .filters button.secondary {{
            background: #94a3b8;
          }}
          .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            padding: 0 40px 24px;
          }}
          .stat-card {{
            background: var(--card);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
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
          .meter-section {{
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
            background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
            color: #fff;
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.18);
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
          .chart-card {{
            background: var(--card);
            border-radius: 18px;
            padding: 12px 12px 4px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
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
        </style>
      </head>
      <body>
        <div class="header">
          <h1 class="title">EMS Energy Dashboard</h1>
          <div class="subtitle">Interactive totals using ${price_per_kwh:.2f} / kWh</div>
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
        <div class="stats-grid">
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
        <div class="meter-section" id="meter-section" style="display: none;">
          <div class="section-title">Utility Meters</div>
          <div class="meter-grid" id="meter-cards"></div>
        </div>
        <div class="charts-grid">
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
        </div>
        <script>
          const dashboardData = {json.dumps(data_payload)};

          const startInput = document.getElementById("start-date");
          const endInput = document.getElementById("end-date");
          const applyButton = document.getElementById("apply-filters");
          const resetButton = document.getElementById("reset-filters");

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
              meterSeries: Object.fromEntries(Object.keys(dashboardData.meter_series).map((key) => [key, []]))
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

          function renderCharts(data) {{
            const rollingValues = rollingMean(data.timestamps, data.totalKw, dashboardData.rolling_hours);
            const rollingTrace = {{
              x: data.timestamps,
              y: rollingValues,
              mode: "lines",
              line: {{ color: "#16a34a", width: 3 }}
            }};
            Plotly.newPlot("rolling-load-chart", [rollingTrace], {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Timestamp", type: "date" }},
              yaxis: {{ title: "kW", rangemode: "tozero" }},
              template: "plotly_white"
            }}, {{ displaylogo: false, responsive: true }});

            const heatmap = buildHeatmap(data.timestamps, data.totalKw);
            Plotly.newPlot("heatmap-chart", [{{
              x: heatmap.dates,
              y: heatmap.hours,
              z: heatmap.z,
              type: "heatmap",
              colorscale: "Turbo",
              zsmooth: "best",
              connectgaps: true,
              colorbar: {{ title: "kW" }}
            }}], {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Date", type: "category" }},
              yaxis: {{ title: "Hour", autorange: "reversed" }},
              template: "plotly_white"
            }}, {{ displaylogo: false, responsive: true }});

            const hourlyProfile = buildHourlyProfile(data.timestamps, data.totalKw);
            const hourlyTrace = {{
              x: hourlyProfile.hours,
              y: hourlyProfile.averages,
              mode: "lines+markers",
              line: {{ color: "#f97316", width: 3 }},
              marker: {{ size: 6 }}
            }};
            Plotly.newPlot("hourly-profile-chart", [hourlyTrace], {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Hour (UTC)", dtick: 1 }},
              yaxis: {{ title: "Average kW", rangemode: "tozero" }},
              template: "plotly_white"
            }}, {{ displaylogo: false, responsive: true }});

            const weekdayProfile = buildWeekdayProfile(data.timestamps, data.totalKw);
            const weekdayTrace = {{
              x: weekdayProfile.weekdays,
              y: weekdayProfile.averages,
              type: "bar",
              marker: {{ color: "#0ea5e9" }}
            }};
            Plotly.newPlot("weekday-profile-chart", [weekdayTrace], {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Day of Week" }},
              yaxis: {{ title: "Average kW", rangemode: "tozero" }},
              template: "plotly_white"
            }}, {{ displaylogo: false, responsive: true }});

            const dailyEnergy = buildDailyEnergy(data.timestamps, data.totalKw);
            const dailyEnergyTrace = {{
              x: dailyEnergy.dates,
              y: dailyEnergy.values,
              type: "bar",
              marker: {{ color: "#8b5cf6" }}
            }};
            Plotly.newPlot("daily-energy-chart", [dailyEnergyTrace], {{
              margin: {{ t: 16, l: 50, r: 24, b: 40 }},
              xaxis: {{ title: "Date", type: "category" }},
              yaxis: {{ title: "Energy (kWh)", rangemode: "tozero" }},
              template: "plotly_white"
            }}, {{ displaylogo: false, responsive: true }});

            if (Object.keys(data.groupSeries).length) {{
              renderGroupChart(data);
            }}
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
            const section = document.getElementById("meter-section");
            const container = document.getElementById("meter-cards");
            if (!section || !container) {{
              return;
            }}
            section.style.display = "block";
            if (!container.dataset.initialized) {{
              container.innerHTML = dashboardData.utility_meters
                .map((meter, idx) => {{
                  const panelList = (meter.panels || []).join(", ");
                  return `
                    <div class="meter-card">
                      <div class="meter-title">${{meter.name}}</div>
                      <div class="meter-value" id="meter-energy-${{idx}}">0.00 kWh</div>
                      <div class="meter-sub" id="meter-cost-${{idx}}">$0.00</div>
                      <div class="meter-sub">Panels: ${{panelList || "None"}}</div>
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

          function renderDashboard() {{
            const filtered = filterData();
            renderCharts(filtered);
            updateMetrics(filtered);
            updateMeterCards(filtered);
          }}

          applyButton.addEventListener("click", renderDashboard);
          resetButton.addEventListener("click", () => {{
            initDateInputs();
            renderDashboard();
          }});

          initDateInputs();
          renderDashboard();
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
