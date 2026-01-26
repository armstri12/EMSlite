
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge_BMS_History.py

Append-only merger for Edwards BMS 4-day panel dumps into an existing wide master CSV.

Behavior:
- Reads RawPanelUsageHistory.csv ("master") in wide format (Timestamp + one column per meter)
- Discovers and loads all Meter*_SystemCurrent.csv files (each is Timestamp + one meter column)
- Normalizes timestamps to America/New_York, robust to 'EST'/'EDT' suffixes and mixed formats
- Builds a wide 'new snapshot' from all panel files
- APPENDS ONLY rows whose Timestamp is NOT already in the master
- Adds new meter columns if discovered; older rows remain blank for those meters
- Does NOT overwrite or backfill any existing master rows

Compatible with pandas >= 2.1 (no infer_datetime_format). Uses 'min' for rounding frequency.
"""

from __future__ import annotations
from pathlib import Path
import math
import re
import sys
from typing import Iterable, List

import pandas as pd
import plotly.express as px

# ==============================
# ========== CONSTANTS =========
# ==============================

# Paths & discovery
DATA_DIR: Path = Path(r"C:\Users\a00544090\OneDrive - ONEVIRTUALOFFICE\SHE Team\Energy Management\EMS Python\BMS Power Dump 1_23_2026")                       # Directory containing master + dumps
MASTER_FILENAME: str = "RawPanelUsageHistory.csv"
GLOB_PATTERN: str = "Meter*_SystemCurrent.csv"   # All 40+ panel files

# Timestamp handling
TIMEZONE: str = "America/New_York"
STRIP_TZ_TOKENS: bool = True                     # Remove trailing 'EST'/'EDT' from strings, then localize
ROUND_TO_MINUTE: bool = True                     # Round parsed datetimes to nearest minute ("min")

# Output behavior
OVERWRITE_IN_PLACE: bool = False                 # If False, writes <master>_UPDATED.csv
FILL_MISSING: str = "blank"                      # "blank" or "0" for missing values in the final CSV

# Parsing assumptions
ASSUME_SECOND_COLUMN_IS_METER: bool = True       # If header doesn't match file name, rename 2nd column to meter

# Aggregations & reporting
TOTAL_AMPS_COLUMN_NAME: str = "Total_Amps"
TOTAL_KW_COLUMN_NAME: str = "Total_kW"
TOTAL_AMPS_SOURCE_COLUMNS: list[str] | None = None  # None => all meter columns; otherwise list of meter columns

# Grouped kW columns to compute and plot.
# - Keys are the *output* column names that will be added to the final CSV.
# - Values are lists of *meter column names* to include in each group.
# - Use exact meter column names as they appear in the master/panel files (after header cleanup).
#   Example (replace with your meter names):
#   KW_GROUP_COLUMNS = {
#       "Production_kW": ["MeterATS01_SystemCurrent", "MeterATS02_SystemCurrent"],
#       "Facilities_kW": ["MeterATS03_SystemCurrent"],
#       "Engineering_kW": ["MeterATS04_SystemCurrent", "MeterATS05_SystemCurrent"],
#   }
KW_GROUP_COLUMNS: dict[str, list[str]] = {
    "Production_kW": [],
    "Facilities_kW": [],
    "Engineering_kW": [],
}

LINE_VOLTAGE: float = 480.0
POWER_FACTOR: float = 1.0

# Plotting
PLOT_GROUP_COLUMNS: bool = True
PLOT_OUTPUT_HTML: str = "group_columns_plot.html"

# ==============================
# ======== UTILITIES ===========
# ==============================

def _normalize_meter_header(raw: str) -> str:
    """Drop trailing parenthetical suffixes like '(A)' and trim whitespace."""
    name = re.sub(r"\s*\(.*?\)\s*$", "", raw or "")
    return name.strip()

def _meter_name_from_file(file_path: Path) -> str:
    """
    Use the file stem as the canonical meter name:
    'MeterATS01_SystemCurrent.csv' -> 'MeterATS01_SystemCurrent'
    """
    return _normalize_meter_header(file_path.stem)

def parse_timestamps_to_tz(series: pd.Series) -> pd.Series:
    """
    Parse mixed timestamp styles, then localize/convert to TIMEZONE.
    Handles examples like:
      - '18-Jan-26 3:15 AM EST'
      - '12/11/25 7:45 AM'
    Compatible with pandas >= 2.1 (no infer_datetime_format).
    """
    s = series.astype(str)

    # Drop explicit 'EST'/'EDT' tokens (no fixed offset info)
    if STRIP_TZ_TOKENS:
        s = s.str.replace(r"\s+(?:EST|EDT)\s*$", "", regex=True)

    # Try explicit formats first to reduce per-element parsing
    # 1) '18-Jan-26 3:15 AM'
    dt = pd.to_datetime(s, errors="coerce", format="%d-%b-%y %I:%M %p")

    # 2) '12/11/25 7:45 AM'
    if dt.isna().any():
        mask = dt.isna()
        try_dt = pd.to_datetime(s[mask], errors="coerce", format="%m/%d/%y %I:%M %p")
        dt = dt.mask(mask, try_dt)

    # 3) Fallback to dateutil (mixed leftovers)
    if dt.isna().any():
        mask = dt.isna()
        try_dt = pd.to_datetime(s[mask], errors="coerce", format=None)
        dt = dt.mask(mask, try_dt)

    # Ensure we ended with a datetime-like Series before using .dt
    if not (pd.api.types.is_datetime64_any_dtype(dt) or pd.api.types.is_datetime64tz_dtype(dt)):
        dt = pd.to_datetime(dt, errors="coerce")

    # Localize or convert to target timezone
    if getattr(dt.dt, "tz", None) is None:
        try:
            dt = dt.dt.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            dt = dt.dt.tz_localize(TIMEZONE)
    else:
        dt = dt.dt.tz_convert(TIMEZONE)

    if ROUND_TO_MINUTE:
        dt = dt.dt.round("min")  # use modern alias; 'T' may error on newer pandas
    return dt

def amps_to_kw(amps: pd.Series) -> pd.Series:
    """Convert 3-phase amps to kW using line voltage and power factor."""
    return amps * (LINE_VOLTAGE * math.sqrt(3) * POWER_FACTOR) / 1000.0

def _resolve_columns(
    available: Iterable[str],
    requested: list[str] | None,
    label: str,
) -> list[str]:
    available_set = set(available)
    if requested is None:
        return [c for c in available if c in available_set]
    missing = [c for c in requested if c not in available_set]
    if missing:
        print(f"Warning: {label} missing columns skipped: {missing}")
    return [c for c in requested if c in available_set]

def add_usage_columns(df: pd.DataFrame) -> pd.DataFrame:
    computed_names = {TOTAL_AMPS_COLUMN_NAME, TOTAL_KW_COLUMN_NAME, *KW_GROUP_COLUMNS.keys()}
    meter_columns = [c for c in df.columns if c != "Timestamp" and c not in computed_names]

    total_sources = _resolve_columns(meter_columns, TOTAL_AMPS_SOURCE_COLUMNS, "TOTAL_AMPS_SOURCE_COLUMNS")
    if total_sources:
        total_amps = df[total_sources].fillna(0).sum(axis=1)
        df[TOTAL_AMPS_COLUMN_NAME] = total_amps
        df[TOTAL_KW_COLUMN_NAME] = amps_to_kw(total_amps)

    for group_name, group_columns in KW_GROUP_COLUMNS.items():
        resolved = _resolve_columns(meter_columns, group_columns, f"KW_GROUP_COLUMNS[{group_name}]")
        if not resolved:
            continue
        group_amps = df[resolved].fillna(0).sum(axis=1)
        df[group_name] = amps_to_kw(group_amps)

    return df

def plot_group_columns(df: pd.DataFrame, output_html: str) -> None:
    group_columns = [name for name in KW_GROUP_COLUMNS.keys() if name in df.columns]
    if not group_columns:
        print("No group columns available to plot.")
        return
    plot_df = df[["Timestamp", *group_columns]].dropna(subset=["Timestamp"]).copy()
    if plot_df.empty:
        print("No data available for plotting group columns.")
        return
    long_df = plot_df.melt(id_vars="Timestamp", var_name="Group", value_name="kW")
    fig = px.line(long_df, x="Timestamp", y="kW", color="Group", title="Group Columns (kW)")
    fig.update_layout(legend_title_text="Group")
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Plot written to: {output_html}")

# ==============================
# ====== I/O: LOAD FRAMES ======
# ==============================

def read_master(master_path: Path) -> pd.DataFrame:
    """
    Load the existing wide master file exactly as-is, keeping column order.
    Ensures Timestamp is tz-aware and meter columns are numeric.
    """
    if not master_path.exists():
        # Start empty skeleton with proper dtype
        return pd.DataFrame(columns=["Timestamp"]).assign(
            Timestamp=pd.Series(dtype=f"datetime64[ns, {TIMEZONE}]")
        )

    df = pd.read_csv(master_path)
    # Preserve original order but normalize header whitespace
    df.columns = [c.strip() for c in df.columns]
    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' column not found in master file: {master_path.name}")

    df["Timestamp"] = parse_timestamps_to_tz(df["Timestamp"])
    for c in df.columns:
        if c != "Timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Do not alter row order beyond ensuring unique timestamps within master
    df = df.drop_duplicates(subset=["Timestamp"], keep="first")
    return df

def load_panel_file(csv_path: Path) -> pd.DataFrame:
    """
    Load a single panel dump (2-column CSV) as ['Timestamp', <meter>].
    The meter column name is taken from the file name; if the header differs,
    we rename the data column accordingly. Coerces data to numeric.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' column not found in {csv_path.name}")

    meter_col = _meter_name_from_file(csv_path)

    # If the expected meter column isn't present, assume 2nd column is the data and rename it
    if meter_col not in df.columns:
        if ASSUME_SECOND_COLUMN_IS_METER and len(df.columns) >= 2:
            # Pick the first non-Timestamp column
            second_col = [c for c in df.columns if c != "Timestamp"][0]
            df = df.rename(columns={second_col: meter_col})
        else:
            raise ValueError(
                f"Meter column '{meter_col}' not found in {csv_path.name} "
                f"and ASSUME_SECOND_COLUMN_IS_METER={ASSUME_SECOND_COLUMN_IS_METER}."
            )

    # Keep only the two columns we need
    df = df[["Timestamp", meter_col]].copy()
    df["Timestamp"] = parse_timestamps_to_tz(df["Timestamp"])
    df[meter_col] = pd.to_numeric(df[meter_col], errors="coerce")

    # Drop invalid/duplicate timestamps within the single panel file
    df = df.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="last")
    return df

def build_new_data_wide(panel_files: List[Path]) -> pd.DataFrame:
    """
    Outer-merge all panel files into a **single wide** dataframe:
    ['Timestamp', meter1, meter2, ...].
    """
    combined: pd.DataFrame | None = None

    for file in panel_files:
        df = load_panel_file(file)  # ['Timestamp', <meter>]
        if combined is None:
            combined = df
            continue

        combined = pd.merge(
            combined,
            df,
            on="Timestamp",
            how="outer",
            suffixes=("", "__dup")
        )

        # If any duplicate suffixes appear, prefer the left-hand value and drop the dup
        dup_cols = [c for c in combined.columns if c.endswith("__dup")]
        for dc in dup_cols:
            base = dc[:-5]
            if base in combined.columns:
                combined[base] = combined[base].combine_first(combined[dc])
            else:
                combined = combined.rename(columns={dc: base})
        if dup_cols:
            combined = combined.drop(columns=dup_cols)

    if combined is None:
        combined = pd.DataFrame(columns=["Timestamp"])

    # Sort & de-dup timestamps in the new snapshot
    if "Timestamp" in combined.columns:
        combined = combined.drop_duplicates(subset=["Timestamp"], keep="last").sort_values("Timestamp")

    return combined

# ==============================
# ====== APPEND-ONLY LOGIC =====
# ==============================

def append_only_new_timestamps(master_df: pd.DataFrame, new_wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append rows from new_wide_df where 'Timestamp' is NOT already present in master_df.
    Does NOT overwrite values in existing master rows.
    """
    if new_wide_df.empty or "Timestamp" not in new_wide_df.columns:
        return master_df

    # Identify new timestamps only
    existing_ts = set(master_df["Timestamp"].dropna().unique())
    new_rows = new_wide_df[~new_wide_df["Timestamp"].isin(existing_ts)].copy()

    if new_rows.empty:
        return master_df

    # Build the union of columns (Timestamp first, rest sorted for reproducibility)
    master_cols = list(master_df.columns)
    new_cols = [c for c in new_rows.columns if c not in master_cols]
    all_cols = ["Timestamp"] + [c for c in master_cols if c != "Timestamp"] + sorted([c for c in new_cols if c != "Timestamp"])

    # Reindex frames to the same schema (adds missing columns as NaN)
    master_aligned = master_df.reindex(columns=all_cols)
    new_rows_aligned = new_rows.reindex(columns=all_cols)

    # Concatenate and keep original master rows intact
    out = pd.concat([master_aligned, new_rows_aligned], ignore_index=True)
    # Ensure unique timestamps (master rows already unique; new rows chosen as unique)
    out = out.drop_duplicates(subset=["Timestamp"], keep="first").sort_values("Timestamp")

    return out

# ==============================
# ============ MAIN ============
# ==============================

def main():
    master_path = DATA_DIR / MASTER_FILENAME
    panel_files = sorted(DATA_DIR.glob(GLOB_PATTERN))

    # Load current master
    master_df = read_master(master_path)
    rows_before, cols_before = len(master_df), len(master_df.columns)

    # Build wide snapshot from all panel dumps
    new_data = build_new_data_wide(panel_files)

    # Append-only merge (no modification to existing timestamps)
    updated = append_only_new_timestamps(master_df, new_data)

    # Optional fill policy
    if FILL_MISSING == "0":
        for c in updated.columns:
            if c != "Timestamp":
                updated[c] = updated[c].fillna(0)

    # Add aggregate usage columns (amps + kW)
    updated = add_usage_columns(updated)

    if PLOT_GROUP_COLUMNS:
        plot_group_columns(updated, PLOT_OUTPUT_HTML)

    # Output
    updated = updated.sort_values("Timestamp")

    if OVERWRITE_IN_PLACE:
        out_path = master_path
    else:
        out_path = master_path.with_name(master_path.stem + "_UPDATED.csv")

    out_df = updated.copy()
    # Persist ISO8601 with offset, Excel-friendly
    out_df["Timestamp"] = out_df["Timestamp"].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out_df.to_csv(out_path, index=False)

    # Console summary
    print("=== Append-Only BMS Merge Summary ===")
    print(f"Directory                 : {DATA_DIR.resolve()}")
    print(f"Master in                 : {master_path.name}")
    print(f"Master out                : {out_path.name} {'(in-place)' if OVERWRITE_IN_PLACE else ''}")
    print(f"Discovered panel files    : {len(panel_files)}")
    print(f"Rows before / after       : {rows_before} -> {len(updated)}  (+{len(updated) - rows_before})")
    print(f"Columns before / after    : {cols_before} -> {len(updated.columns)}")
    if len(updated) == rows_before:
        print("No new timestamps appended (panel dump timestamps already exist in master).")

if __name__ == "__main__":
    # Avoid super-wide console wrapping if pandas prints anything
    pd.set_option("display.width", 160)
    try:
        main()
    except Exception as e:
        print("\nERROR:", e, file=sys.stderr)
        raise
