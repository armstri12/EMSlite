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
    metrics = compute_energy_metrics(df)
    price_per_kwh = float(CONFIG["price_per_kwh"])
    total_cost = metrics.get("total_kwh", 0.0) * price_per_kwh
    date_range = ""
    if not df.empty:
        start = df["Timestamp"].min()
        end = df["Timestamp"].max()
        if pd.notna(start) and pd.notna(end):
            date_range = f"{start.date()} → {end.date()}"

    figures = [
        ("Total Load", create_total_kw_fig(df)),
        ("Smoothed Load", create_total_kw_rolling_fig(df, window)),
        ("Load Heatmap", create_daily_hour_heatmap_fig(df)),
    ]
    group_fig = create_group_columns_fig(df)
    if group_fig is not None:
        figures.append(("Group Loads", group_fig))

    chart_cards = "\n".join(
        f"""
        <div class="chart-card">
          <div class="chart-title">{title}</div>
          {pio.to_html(fig, include_plotlyjs=False, full_html=False)}
        </div>
        """
        for title, fig in figures
    )

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
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            padding: 0 40px 40px;
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
          .chart-card .plotly-graph-div {{
            width: 100% !important;
          }}
        </style>
      </head>
      <body>
        <div class="header">
          <h1 class="title">EMS Energy Dashboard</h1>
          <div class="subtitle">Total cost uses ${price_per_kwh:.2f} / kWh · {date_range}</div>
        </div>
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-label">Total Energy</div>
            <div class="stat-value">{metrics.get("total_kwh", 0.0):,.2f} kWh</div>
            <div class="stat-hint">Integrated from total load</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Estimated Cost</div>
            <div class="stat-value">${total_cost:,.2f}</div>
            <div class="stat-hint">Rate: ${price_per_kwh:.2f} / kWh</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Average Load</div>
            <div class="stat-value">{metrics.get("average_kw", 0.0):,.2f} kW</div>
            <div class="stat-hint">Mean across timestamps</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Peak Load</div>
            <div class="stat-value">{metrics.get("peak_kw", 0.0):,.2f} kW</div>
            <div class="stat-hint">Highest observed value</div>
          </div>
        </div>
        <div class="charts-grid">
          {chart_cards}
        </div>
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
