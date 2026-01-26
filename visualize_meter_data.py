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
- top_meters_timeseries.html
- total_kw_histogram.html
- daily_hour_heatmap.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px

LINE_VOLTAGE = 480.0
POWER_FACTOR = 1.0
DEFAULT_TOP_N = 6
DEFAULT_ROLLING_WINDOW = "1h"

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
    # Plot tuning.
    "top_n_meters": DEFAULT_TOP_N,
    "rolling_window": DEFAULT_ROLLING_WINDOW,
    # Output file names (set to customize or rename artifacts).
    "outputs": {
        "total_kw_timeseries": "total_kw_timeseries.html",
        "total_kw_rolling": "total_kw_rolling_1h.html",
        "top_meters_timeseries": "top_meters_timeseries.html",
        "total_kw_histogram": "total_kw_histogram.html",
        "daily_hour_heatmap": "daily_hour_heatmap.html",
    },
}


def amps_to_kw(amps: pd.Series) -> pd.Series:
    return amps * (CONFIG["line_voltage"] * 3 ** 0.5 * CONFIG["power_factor"]) / 1000.0


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
        "--top-n",
        type=int,
        default=CONFIG["top_n_meters"],
        help=f"Number of top meters to plot (default: {CONFIG['top_n_meters']}).",
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
    return [col for col in columns if col != "Timestamp"]


def add_total_kw(df: pd.DataFrame, meters: list[str]) -> pd.DataFrame:
    total_amps = df[meters].fillna(0).sum(axis=1)
    df = df.copy()
    df["Total_Amps"] = total_amps
    df["Total_kW"] = amps_to_kw(total_amps)
    return df


def plot_total_kw(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = px.line(df, x="Timestamp", y="Total_kW", title="Total kW Over Time")
    output_path = output_dir / CONFIG["outputs"]["total_kw_timeseries"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def normalize_rolling_window(window: str) -> str:
    return window.replace("H", "h")


def plot_total_kw_rolling(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    normalized_window = normalize_rolling_window(window)
    rolling = (
        df.set_index("Timestamp")["Total_kW"]
        .rolling(normalized_window)
        .mean()
        .reset_index()
    )
    fig = px.line(
        rolling,
        x="Timestamp",
        y="Total_kW",
        title=f"Total kW (Rolling {normalized_window})",
    )
    output_path = output_dir / CONFIG["outputs"]["total_kw_rolling"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_top_meters(df: pd.DataFrame, meters: list[str], top_n: int, output_dir: Path) -> Path:
    ranked = df[meters].mean().sort_values(ascending=False)
    top_meters = ranked.head(top_n).index.tolist()
    long_df = df[["Timestamp", *top_meters]].melt(
        id_vars="Timestamp",
        var_name="Meter",
        value_name="Amps",
    )
    fig = px.line(long_df, x="Timestamp", y="Amps", color="Meter", title=f"Top {top_n} Meters (Amps)")
    output_path = output_dir / CONFIG["outputs"]["top_meters_timeseries"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_total_kw_histogram(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = px.histogram(df, x="Total_kW", nbins=60, title="Distribution of Total kW")
    output_path = output_dir / CONFIG["outputs"]["total_kw_histogram"]
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_daily_hour_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    temp = df.copy()
    temp["Date"] = temp["Timestamp"].dt.date
    temp["Hour"] = temp["Timestamp"].dt.hour
    pivot = temp.pivot_table(index="Hour", columns="Date", values="Total_kW", aggfunc="mean")
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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    meters = meter_columns(df.columns)
    if not meters:
        raise ValueError("No meter columns found in the input file.")

    df = add_total_kw(df, meters)

    outputs = [
        plot_total_kw(df, output_dir),
        plot_total_kw_rolling(df, output_dir, args.rolling_window),
        plot_top_meters(df, meters, args.top_n, output_dir),
        plot_total_kw_histogram(df, output_dir),
        plot_daily_hour_heatmap(df, output_dir),
    ]

    print("Generated visualizations:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
