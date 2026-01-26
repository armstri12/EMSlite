#!/usr/bin/env python3
"""Merge per-meter CSV dumps into a master CSV.

The master CSV uses column A as timestamps and subsequent columns for each meter
(e.g. MeterATS01_SystemCurrent). Each per-meter dump CSV contains two columns:
Timestamp and the meter's amperage data, with a header row.

Configuration (edit the CONFIG dict below):
  - master_path: Path to your master CSV (can be overridden via --master).
  - dumps_path: Path to the folder with per-meter CSV dumps (--dumps overrides).
  - output_path: Where to write the updated master (optional, defaults to master).
  - timestamp_fallback: Header name to use when creating a brand new master file.
  - dump_glob: Pattern for discovering per-meter CSV files in the dumps directory.
  - overwrite_existing: If False, only fill empty cells in the master file.
  - encoding: File encoding used for reading/writing CSVs.

Usage:
  merge_meter_data.py --master master.csv --dumps ./dumps --output master.csv

Quick start (edit CONFIG with your paths, then run without flags):
  1) Update CONFIG["master_path"] and CONFIG["dumps_path"] below.
  2) Optional: set CONFIG["output_path"] if you want a new output file.
  3) Run: ./merge_meter_data.py

By default, the script fills missing data and appends new timestamps without
overwriting existing non-empty values.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List


# ----------------------------
# Configuration (edit as needed)
# ----------------------------
CONFIG = {
    # Copy/paste your file paths here.
    "master_path": "/path/to/master.csv",
    "dumps_path": "/path/to/dump_files",
    # Leave blank ("") to overwrite the master in place.
    "output_path": "",
    "timestamp_fallback": "Timestamp",
    "dump_glob": "*.csv",
    "overwrite_existing": False,
    "encoding": "utf-8",
}


def read_master(path: pathlib.Path) -> tuple[List[str], Dict[str, Dict[str, str]]]:
    """Load the master CSV into memory.

    Returns:
        headers: list of column names (timestamp column first).
        data: mapping of timestamp -> row dict.
    """
    if not path.exists():
        return [CONFIG["timestamp_fallback"]], {}

    with path.open(newline="", encoding=CONFIG["encoding"]) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Master file {path} is missing a header row.")
        headers = list(reader.fieldnames)
        if not headers:
            raise ValueError(f"Master file {path} has empty headers.")
        data: Dict[str, Dict[str, str]] = {}
        timestamp_key = headers[0]
        for row in reader:
            # Use the first header as the timestamp column, matching the master file.
            timestamp = row.get(timestamp_key, "").strip()
            if not timestamp:
                continue
            data[timestamp] = row
    return headers, data


def ensure_column(headers: List[str], column: str) -> None:
    """Add the meter column to the master header list if it's missing."""
    if column not in headers:
        headers.append(column)


def merge_dump(
    dump_path: pathlib.Path,
    headers: List[str],
    data: Dict[str, Dict[str, str]],
    timestamp_header: str,
) -> None:
    """Merge a single per-meter dump file into the master dataset."""
    meter_name = dump_path.stem
    ensure_column(headers, meter_name)

    with dump_path.open(newline="", encoding=CONFIG["encoding"]) as handle:
        reader = csv.reader(handle)
        header_row = next(reader, None)
        if header_row is None:
            return

        for row in reader:
            if not row:
                continue
            # Step 1: normalize input values.
            timestamp = row[0].strip() if len(row) > 0 else ""
            value = row[1].strip() if len(row) > 1 else ""
            if not timestamp:
                continue

            # Step 2: ensure the timestamp exists in the master dataset.
            if timestamp not in data:
                data[timestamp] = {timestamp_header: timestamp}

            existing = data[timestamp].get(meter_name, "").strip()
            # Step 3: decide whether to write new data.
            if CONFIG["overwrite_existing"]:
                if value:
                    data[timestamp][meter_name] = value
            else:
                if not existing and value:
                    data[timestamp][meter_name] = value


def write_master(
    path: pathlib.Path,
    headers: List[str],
    data: Dict[str, Dict[str, str]],
) -> None:
    """Write the updated master CSV with ordered timestamps."""
    timestamp_header = headers[0]
    ordered_timestamps = sorted(data.keys())

    with path.open("w", newline="", encoding=CONFIG["encoding"]) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for timestamp in ordered_timestamps:
            # Fill blanks for columns that have no data for this timestamp.
            row = {header: "" for header in headers}
            row[timestamp_header] = timestamp
            row.update(data[timestamp])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-meter CSV dumps into a master CSV."
    )
    parser.add_argument(
        "--master",
        default=CONFIG["master_path"],
        type=pathlib.Path,
        help="Path to the master CSV file.",
    )
    parser.add_argument(
        "--dumps",
        default=CONFIG["dumps_path"],
        type=pathlib.Path,
        help="Directory containing per-meter dump CSV files.",
    )
    parser.add_argument(
        "--output",
        default=CONFIG["output_path"] or None,
        type=pathlib.Path,
        help="Output path for the updated master CSV. Defaults to --master.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.master:
        raise ValueError(
            "Master path is required. Set CONFIG['master_path'] or pass --master."
        )

    if not args.dumps:
        raise ValueError(
            "Dumps path is required. Set CONFIG['dumps_path'] or pass --dumps."
        )

    # If --output (or CONFIG output_path) is omitted, update the master in place.
    output_path = args.output if args.output else args.master

    # Step 1: read the existing master file (or initialize headers).
    headers, data = read_master(args.master)
    timestamp_header = headers[0]

    # Step 2: validate the dump directory exists and contains CSV files.
    if not args.dumps.exists() or not args.dumps.is_dir():
        raise FileNotFoundError(f"Dump directory not found: {args.dumps}")

    dump_files = sorted(args.dumps.glob(CONFIG["dump_glob"]))
    if not dump_files:
        raise FileNotFoundError(f"No CSV files found in {args.dumps}")

    # Step 3: merge each dump file into the in-memory dataset.
    for dump_file in dump_files:
        merge_dump(dump_file, headers, data, timestamp_header)

    # Step 4: write the updated master CSV.
    write_master(output_path, headers, data)


if __name__ == "__main__":
    main()
