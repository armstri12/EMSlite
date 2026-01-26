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
DEFAULT_ROLLING_WINDOW = "1H"


def amps_to_kw(amps: pd.Series) -> pd.Series:
    return amps * (LINE_VOLTAGE * 3 ** 0.5 * POWER_FACTOR) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EMS meter data from a wide CSV.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file (wide format with Timestamp column).",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory to write HTML plots (default: visualizations).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top meters to plot (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--rolling-window",
        default=DEFAULT_ROLLING_WINDOW,
        help=f"Rolling window for the smoothed total kW plot (default: {DEFAULT_ROLLING_WINDOW}).",
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
    output_path = output_dir / "total_kw_timeseries.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_total_kw_rolling(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    rolling = df.set_index("Timestamp")["Total_kW"].rolling(window).mean().reset_index()
    fig = px.line(rolling, x="Timestamp", y="Total_kW", title=f"Total kW (Rolling {window})")
    output_path = output_dir / "total_kw_rolling_1h.html"
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
    output_path = output_dir / "top_meters_timeseries.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_total_kw_histogram(df: pd.DataFrame, output_dir: Path) -> Path:
    fig = px.histogram(df, x="Total_kW", nbins=60, title="Distribution of Total kW")
    output_path = output_dir / "total_kw_histogram.html"
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
    output_path = output_dir / "daily_hour_heatmap.html"
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
