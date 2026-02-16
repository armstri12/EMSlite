"""File watcher and auto-ingestion for CSV panel dumps.

Monitors a drop folder for new Meter*_SystemCurrent.csv files,
merges them into the master CSV, and auto-discovers new devices.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


def _clean_display_name(raw_id: str) -> str:
    """Derive a human-friendly display name from a CSV column name.

    'MeterATS01_SystemCurrent' -> 'Meter ATS01'
    """
    name = re.sub(r"_SystemCurrent$", "", raw_id)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    return name.strip()


def _ensure_device(device_id: str) -> None:
    """Create a Device record if one doesn't exist for this panel."""
    from .database import get_session
    from .models import Device

    session = get_session()
    try:
        existing = session.get(Device, device_id)
        if existing is None:
            device = Device(
                id=device_id,
                display_name=_clean_display_name(device_id),
                enabled=True,
            )
            session.add(device)
            session.commit()
            logger.info("Auto-discovered new device: %s -> %s", device_id, device.display_name)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _log_ingest(filename: str, status: str, rows_added: int = 0,
                device_id: str | None = None, error: str | None = None) -> None:
    from .database import get_session
    from .models import IngestLog

    session = get_session()
    try:
        entry = IngestLog(
            filename=filename,
            status=status,
            rows_added=rows_added,
            device_id=device_id,
            error_message=error,
        )
        session.add(entry)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()


def ingest_file(csv_path: Path, master_path: Path, processed_dir: Path, failed_dir: Path) -> None:
    """Process a single panel CSV: parse, merge into master, move to processed."""
    # Import merge_meter_data functions (reuse existing ETL)
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import pandas as pd
    from merge_meter_data import load_panel_file, read_master

    filename = csv_path.name
    logger.info("Ingesting: %s", filename)

    try:
        panel_df = load_panel_file(csv_path)
        panel_cols = [c for c in panel_df.columns if c != "Timestamp"]
        device_id = panel_cols[0] if panel_cols else None

        # Auto-discover device
        if device_id:
            _ensure_device(device_id)

        # Load master and merge
        master_df = read_master(master_path)
        rows_before = len(master_df)

        # merge_meter_data.append_only_new_timestamps only adds rows with
        # NEW timestamps.  When multiple single-panel CSVs share the same
        # timestamps (the normal case for a batch of panel dumps), only the
        # first file's data would be kept; subsequent panels would be
        # silently skipped because those timestamps already exist.
        #
        # Fix: merge on Timestamp so that *new columns* (or NaN cells in
        # existing columns) are filled in for rows that already exist,
        # while truly new timestamps are still appended.
        meter_col = [c for c in panel_df.columns if c != "Timestamp"][0]

        # Ensure the column exists in master (may be brand-new meter)
        if meter_col not in master_df.columns:
            master_df[meter_col] = float("nan")

        # Set Timestamp as index for aligned update
        master_idx = master_df.set_index("Timestamp")
        panel_idx = panel_df.set_index("Timestamp")

        # Update existing rows where master has NaN for this meter
        master_idx[meter_col] = master_idx[meter_col].combine_first(panel_idx[meter_col])

        # Append rows with timestamps not yet in master
        new_ts = panel_idx.index.difference(master_idx.index)
        if len(new_ts) > 0:
            new_rows = panel_idx.loc[new_ts]
            master_idx = pd.concat([master_idx, new_rows])

        updated = master_idx.sort_index().reset_index()
        rows_added = len(updated) - rows_before

        # Save updated master
        out_df = updated.copy()
        tz = "America/New_York"
        if hasattr(out_df["Timestamp"].dt, "tz") and out_df["Timestamp"].dt.tz is not None:
            out_df["Timestamp"] = out_df["Timestamp"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M:%S%z")
        else:
            out_df["Timestamp"] = out_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out_df.to_csv(master_path, index=False)

        # Move to processed
        processed_dir.mkdir(parents=True, exist_ok=True)
        dest = processed_dir / filename
        if dest.exists():
            dest = processed_dir / f"{csv_path.stem}_{int(time.time())}{csv_path.suffix}"
        shutil.move(str(csv_path), str(dest))

        _log_ingest(filename, "success", rows_added, device_id)
        logger.info("Ingested %s: +%d rows", filename, rows_added)

    except Exception as e:
        logger.error("Failed to ingest %s: %s", filename, e)
        failed_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(csv_path), str(failed_dir / filename))
        except Exception:
            pass
        _log_ingest(filename, "failed", error=str(e))


class _CSVHandler(FileSystemEventHandler):
    """Watchdog handler that triggers ingestion on new CSV files."""

    def __init__(self, glob_pattern: str, master_path: Path,
                 processed_dir: Path, failed_dir: Path):
        super().__init__()
        self.glob_pattern = glob_pattern
        self.master_path = master_path
        self.processed_dir = processed_dir
        self.failed_dir = failed_dir
        self._debounce: dict[str, float] = {}

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if not path.match(self.glob_pattern):
            return

        # Debounce: wait for file to be fully written
        now = time.time()
        if path.name in self._debounce and now - self._debounce[path.name] < 2:
            return
        self._debounce[path.name] = now

        # Small delay to let file finish writing
        time.sleep(1)
        if path.exists():
            ingest_file(path, self.master_path, self.processed_dir, self.failed_dir)


def start_watcher(
    drops_dir: str | Path,
    master_path: str | Path,
    glob_pattern: str = "Meter*_SystemCurrent.csv",
) -> Observer:
    """Start a background file watcher on the drops directory."""
    drops = Path(drops_dir).resolve()
    drops.mkdir(parents=True, exist_ok=True)
    master = Path(master_path).resolve()
    processed = drops / "processed"
    failed = drops / "failed"

    handler = _CSVHandler(glob_pattern, master, processed, failed)
    observer = Observer()
    observer.schedule(handler, str(drops), recursive=False)
    observer.daemon = True
    observer.start()
    logger.info("File watcher started on: %s", drops)
    return observer


def sync_devices_from_master(master_path: str | Path) -> int:
    """Read the master CSV headers and ensure a Device record exists for each column.

    This handles the case where a user places the master CSV directly into data/
    without going through the file-drop ingestion pipeline.
    """
    import pandas as pd

    master = Path(master_path).resolve()
    if not master.exists():
        return 0

    try:
        cols = pd.read_csv(master, nrows=0).columns.tolist()
    except Exception as e:
        logger.warning("Could not read master CSV headers: %s", e)
        return 0

    count = 0
    for col in cols:
        if col.strip().lower() == "timestamp":
            continue
        _ensure_device(col)
        count += 1

    if count:
        logger.info("Synced %d devices from master CSV columns", count)
    return count


def ingest_existing(
    drops_dir: str | Path,
    master_path: str | Path,
    glob_pattern: str = "Meter*_SystemCurrent.csv",
) -> int:
    """Process any CSV files already sitting in the drops folder."""
    drops = Path(drops_dir).resolve()
    master = Path(master_path).resolve()
    processed = drops / "processed"
    failed = drops / "failed"
    count = 0

    for csv_file in sorted(drops.glob(glob_pattern)):
        if csv_file.is_file():
            ingest_file(csv_file, master, processed, failed)
            count += 1

    return count
