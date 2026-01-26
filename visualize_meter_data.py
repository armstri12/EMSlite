#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_meter_data.py

Generate multiple visualizations from EMS meter data in a wide CSV format.

Expected input format:
- Timestamp column named "Timestamp".
- One or more numeric meter columns (amps).

Outputs:
- total_kw_timeseries.html
- total_kw_rolling_1h.html
- daily_hour_heatmap.html
- group_columns_plot.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px

LINE_VOLTAGE = 480.0
POWER_FACTOR = 1.0
DEFAULT_ROLLING_WINDOW = "1h"
TOTAL_AMPS_COLUMN_NAME = "Total_Amps"
TOTAL_KW_COLUMN_NAME = "Total_kW"

# ==============================
# ======== CONFIG BLOCK ========
# ==============================
# Update these defaults to match your environment and desired outputs.
CONFIG = {
    # Location of the source data file (wide CSV with Timestamp + meter columns).
    "input_file": "RawPanelUsageHistory_UPDATED.csv",
    # Output folder for HTML visualizations.
    "output_dir": "visualizations",
    # Electrical conversion constants used for amps -> kW.
    "line_voltage": LINE_VOLTAGE,
    "power_factor": POWER_FACTOR,
    # Optional list of meter columns to include in Total_Amps/Total_kW.
    # Use None to include all meter columns.
    "total_amps_sources": None,
    # Grouped kW columns to compute and optionally plot.
    # Keys are output column names, values are meter column names to include.
    "kw_group_columns": {
        "Production_kW": [],
        "Facilities_kW": [],
        "Engineering_kW": [],
    },
    # Plot tuning.
    "rolling_window": DEFAULT_ROLLING_WINDOW,
    # Output file names (set to customize or rename artifacts).
    "outputs": {
        "total_kw_timeseries": "total_kw_timeseries.html",
        "total_kw_rolling": "total_kw_rolling_1h.html",
        "daily_hour_heatmap": "daily_hour_heatmap.html",
        "group_columns_plot": "group_columns_plot.html",
    },
}


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
        "--input",
        default=CONFIG["input_file"],
        help="Path to the input CSV file (wide format with Timestamp column).",
    )
    parser.add_argument(
        "--output-dir",
        default=CONFIG["output_dir"],
        help="Directory to write HTML plots (default: visualizations).",
    )
    parser.add_argument(
        "--rolling-window",
        default=CONFIG["rolling_window"],
        help=f"Rolling window for the smoothed total kW plot (default: {CONFIG['rolling_window']}).",
    )
    return parser.parse_args()


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
        *CONFIG["kw_group_columns"].keys(),
    }
    return [col for col in columns if col != "Timestamp" and col not in computed_names]


def add_usage_columns(df: pd.DataFrame, meters: list[str]) -> pd.DataFrame:
    df = df.copy()
    total_sources = resolve_columns(meters, CONFIG["total_amps_sources"], "total_amps_sources")
    if total_sources:
        total_amps = df[total_sources].fillna(0).sum(axis=1)
        df[TOTAL_AMPS_COLUMN_NAME] = total_amps
        df[TOTAL_KW_COLUMN_NAME] = amps_to_kw(total_amps)

    for group_name, group_columns in CONFIG["kw_group_columns"].items():
        resolved = resolve_columns(meters, group_columns, f"kw_group_columns[{group_name}]")
        if not resolved:
            continue
        group_amps = df[resolved].fillna(0).sum(axis=1)
        df[group_name] = amps_to_kw(group_amps)

    return df


def plot_total_kw(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = px.line(df, x="Timestamp", y=TOTAL_KW_COLUMN_NAME, title="Total kW Over Time")
    output_path = output_dir / CONFIG["outputs"]["total_kw_timeseries"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def normalize_rolling_window(window: str) -> str:
    return window.replace("H", "h")


def plot_total_kw_rolling(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    normalized_window = normalize_rolling_window(window)
    rolling = (
        df.set_index("Timestamp")[TOTAL_KW_COLUMN_NAME]
        .rolling(normalized_window)
        .mean()
        .reset_index()
    )
    fig = px.line(
        rolling,
        x="Timestamp",
        y=TOTAL_KW_COLUMN_NAME,
        title=f"Total kW (Rolling {normalized_window})",
    )
    output_path = output_dir / CONFIG["outputs"]["total_kw_rolling"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_daily_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    temp = df.copy()
    temp["Date"] = temp["Timestamp"].dt.date
    temp["Hour"] = temp["Timestamp"].dt.hour
    pivot = temp.pivot_table(index="Hour", columns="Date", values=TOTAL_KW_COLUMN_NAME, aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        title="Average Total kW by Hour and Day",
        labels={"color": "kW"},
    )
    output_path = output_dir / CONFIG["outputs"]["daily_hour_heatmap"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_group_columns(df: pd.DataFrame, output_dir: Path) -> Path | None:
    group_columns = [name for name in CONFIG["kw_group_columns"].keys() if name in df.columns]
    if not group_columns:
        return None
    plot_df = df[["Timestamp", *group_columns]].dropna(subset=["Timestamp"]).copy()
    if plot_df.empty:
        return None
    long_df = plot_df.melt(id_vars="Timestamp", var_name="Group", value_name="kW")
    fig = px.line(long_df, x="Timestamp", y="kW", color="Group", title="Group Columns (kW)")
    fig.update_layout(legend_title_text="Group")
    output_path = output_dir / CONFIG["outputs"]["group_columns_plot"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    meters = meter_columns(df.columns)
    if not meters:
        raise ValueError("No meter columns found in the input file.")

    df = add_usage_columns(df, meters)

    outputs = []
    if TOTAL_KW_COLUMN_NAME in df.columns:
        outputs.extend(
            [
                plot_total_kw(df, output_dir),
                plot_total_kw_rolling(df, output_dir, args.rolling_window),
                plot_daily_hour_heatmap(df, output_dir),
            ]
        )
    group_plot = plot_group_columns(df, output_dir)
    if group_plot:
        outputs.append(group_plot)

    print("Generated visualizations:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
