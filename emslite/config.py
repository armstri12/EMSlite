"""Configuration loading and management."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG: dict = {
    "input_file": "RawPanelUsageHistory_UPDATED.csv",
    "output_dir": "visualizations",
    "line_voltage": 480.0,
    "power_factor": 1.0,
    "price_per_kwh": 0.25,
    "carbon_kg_per_kwh": 0.4,
    "total_amps_sources": None,
    "utility_meters": [],
    "combo_columns": {},
    "rolling_window": "1h",
    "dashboard_logo_path": "",
    "drops_dir": "drops",
    "data_dir": "data",
    "master_filename": "RawPanelUsageHistory.csv",
    "glob_pattern": "Meter*_SystemCurrent.csv",
}


def merge_config(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def strip_json_noise(raw_text: str) -> str:
    """Remove JS-style comments and trailing commas from JSON text."""
    no_block = re.sub(r"/\*.*?\*/", "", raw_text, flags=re.DOTALL)
    no_line = re.sub(r"//.*?$", "", no_block, flags=re.MULTILINE)
    return re.sub(r",(\s*[}\]])", r"\1", no_line)


def load_config(path: str | Path | None = None) -> dict:
    """Load config from a JSON file, falling back to defaults."""
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    if not p.exists():
        return deepcopy(DEFAULT_CONFIG)
    raw_text = p.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        loaded = json.loads(strip_json_noise(raw_text))
    return merge_config(DEFAULT_CONFIG, loaded)


def save_config(config: dict, path: str | Path) -> None:
    """Persist config to a JSON file."""
    p = Path(path)
    p.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
