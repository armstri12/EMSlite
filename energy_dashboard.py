#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
energy_dashboard.py

Next-Generation Energy Monitoring Dashboard for EMSlite.

Implements the admin-shell design specified in ui-design-brief.jsonc and
ui-implementation-prompt.md, adapted to EMS meter data.

Design features:
- Fixed sidebar with dark navy gradient + vertical navigation
- Sticky topbar with user actions and dark mode toggle
- KPI summary cards with trend badges, decorative icons, sparklines
- Donut chart for energy utilization efficiency (load factor)
- Vertical bar chart with navy-to-green gradient for daily energy
- Horizontal progress bars for top panels by consumption
- Heatmap, hourly profile, weekday distribution analytics
- Period comparison with savings banner
- Searchable, sortable data table with CSV export
- Full dark mode with class-based toggling
- Responsive breakpoints at 640/768/1024/1280/1536px

Config:
- Defaults loaded from visualization_config.json (override with --config).
- Config loader accepts JSON with optional // or /* */ comments and trailing commas.

Outputs:
- next_gen_dashboard.html (standalone self-contained interactive dashboard)
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd

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
    "dashboard_logo_path": "",
    "visualizations": {
        "total_kw_timeseries": {"enabled": False, "output": "total_kw_timeseries.html"},
        "total_kw_rolling": {"enabled": False, "output": "total_kw_rolling_1h.html"},
        "daily_hour_heatmap": {"enabled": False, "output": "daily_hour_heatmap.html"},
        "group_columns_plot": {"enabled": False, "output": "group_columns_plot.html"},
        "dashboard": {"enabled": True, "output": "dashboard.html"},
        "comparison_dashboard": {"enabled": True, "output": "comparison_dashboard.html"},
        "next_gen_dashboard": {"enabled": True, "output": "next_gen_dashboard.html"},
    },
}

CONFIG: dict[str, object] = deepcopy(DEFAULT_CONFIG)


# ── Shared helpers (same as visualize_meter_data.py) ──

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
    parser = argparse.ArgumentParser(description="Generate next-gen energy monitoring dashboard.")
    parser.add_argument("--config", default="visualization_config.json")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rolling-window", default=None)
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
    return re.sub(r",(\s*[}\]])", r"\1", no_line)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}.")
    raw_text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError:
        loaded = json.loads(strip_json_noise(raw_text))
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
    computed_names = {TOTAL_AMPS_COLUMN_NAME, TOTAL_KW_COLUMN_NAME, *CONFIG["combo_columns"].keys()}
    return [col for col in columns if col != "Timestamp" and col not in computed_names]


def resolve_utility_meters(available: Iterable[str]) -> list[dict[str, object]]:
    resolved = []
    for idx, meter in enumerate(CONFIG.get("utility_meters") or []):
        if not isinstance(meter, dict):
            continue
        name = str(meter.get("name") or f"Meter {idx + 1}")
        panels = meter.get("panels") or []
        if not isinstance(panels, list):
            panels = []
        resolved.append({"name": name, "panels": resolve_columns(available, panels, f"utility_meters[{name}]")})
    return resolved


def add_usage_columns(df: pd.DataFrame, meters: list[str]) -> pd.DataFrame:
    df = df.copy()
    total_sources = resolve_columns(meters, CONFIG["total_amps_sources"], "total_amps_sources")
    if total_sources:
        total_amps = df[total_sources].fillna(0).sum(axis=1)
        df[TOTAL_AMPS_COLUMN_NAME] = total_amps
        df[TOTAL_KW_COLUMN_NAME] = amps_to_kw(total_amps)
    for group_name, group_cols in CONFIG["combo_columns"].items():
        resolved = resolve_columns(meters, group_cols, f"combo_columns[{group_name}]")
        if resolved:
            df[group_name] = amps_to_kw(df[resolved].fillna(0).sum(axis=1))
    return df


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


# ── Dashboard builder ──

def build_next_gen_dashboard(df: pd.DataFrame, output_dir: Path, window: str) -> Path:
    ordered = df.sort_values("Timestamp")
    timestamps = ordered["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    total_kw = ordered.get(TOTAL_KW_COLUMN_NAME, pd.Series()).fillna(0).tolist()
    group_columns = [n for n in CONFIG["combo_columns"].keys() if n in ordered.columns]
    group_series = {n: ordered[n].fillna(0).tolist() for n in group_columns}
    panel_cols = meter_columns(ordered.columns)
    panel_series = {n: amps_to_kw(ordered[n].fillna(0)).fillna(0).tolist() for n in panel_cols}
    group_definitions = []
    for gn, gp in CONFIG["combo_columns"].items():
        group_definitions.append({"name": gn, "panels": resolve_columns(ordered.columns, gp, f"combo_columns[{gn}]")})
    meter_definitions = resolve_utility_meters(ordered.columns)
    meter_series = {}
    zero_values = [0.0] * len(timestamps)
    for meter in meter_definitions:
        if meter["panels"]:
            meter_series[meter["name"]] = amps_to_kw(ordered[meter["panels"]].fillna(0).sum(axis=1)).fillna(0).tolist()
        else:
            meter_series[meter["name"]] = zero_values
    price_per_kwh = float(CONFIG["price_per_kwh"])
    rolling_hours = parse_window_to_hours(window)
    logo_path = CONFIG.get("dashboard_logo_path") or ""

    data_payload = {
        "timestamps": timestamps,
        "total_kw": total_kw,
        "group_series": group_series,
        "panel_series": panel_series,
        "panel_names": panel_cols,
        "group_definitions": group_definitions,
        "utility_meters": meter_definitions,
        "meter_series": meter_series,
        "rolling_hours": rolling_hours,
        "price_per_kwh": price_per_kwh,
    }

    html = _generate_html(data_payload, logo_path, price_per_kwh)
    vis_config = CONFIG.get("visualizations", {})
    ng_config = vis_config.get("next_gen_dashboard", {})
    output_path = output_dir / ng_config.get("output", "next_gen_dashboard.html")
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _generate_html(data_payload: dict, logo_path: str, price_per_kwh: float) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>EMSlite Energy Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
<style>
/* ═══════════════════════════════════════════════════════
   DESIGN TOKENS  — from ui-design-brief.jsonc
   ═══════════════════════════════════════════════════════ */
:root {{
  /* Primary – Chartreuse / lime-green */
  --p-50:#F4FDE8; --p-100:#E6FACB; --p-200:#CCEF9D; --p-300:#ABDF65;
  --p-400:#8BD435; --p-500:#6DBF1A; --p-600:#539A11; --p-700:#407512;
  --p-800:#355D14; --p-900:#2E4F16; --p-950:#152B06;
  /* Secondary – Deep navy */
  --s-50:#EEF6FA; --s-100:#D4EAF2; --s-200:#ADD6E8; --s-300:#74B6D5;
  --s-400:#398EB9; --s-500:#24729D; --s-600:#1D5B83; --s-700:#1B4A6C;
  --s-800:#1B3E5A; --s-900:#132D42; --s-950:#0B1E2E;
  /* Gray */
  --g-50:#F8FAFB; --g-100:#F1F4F6; --g-200:#E3E8EC; --g-300:#CDD4DB;
  --g-400:#9BA5B0; --g-500:#6C7784; --g-600:#4F5966; --g-700:#3A424D;
  --g-800:#252C35; --g-900:#171C23; --g-950:#0D1117;
  /* Accents */
  --positive-bg:#ECFDF3; --positive-text:#16A34A;
  --negative-bg:#FEF2F2; --negative-text:#DC2626;
  --warning:#F97316; --danger:#EF4444; --info:#38BDF8;
  /* Sidebar */
  --sidebar-from:#0F2A3C; --sidebar-to:#0B1E2E;
  /* Shadows */
  --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  /* Sizing */
  --sidebar-w: 17rem;
  --sidebar-collapsed: 5rem;
  --topbar-h: 4rem;
  --card-radius: 1rem;
  --card-pad: 1.5rem;
  --card-gap: 1.5rem;
  --badge-radius: 9999px;
  /* Surfaces — light mode defaults */
  --page-bg: var(--g-50);
  --card-bg: #ffffff;
  --card-border: var(--g-200);
  --topbar-bg: #ffffff;
  --topbar-border: var(--g-200);
  --heading: var(--g-700);
  --body-text: var(--g-600);
  --muted: var(--g-400);
  --input-bg: #ffffff;
  --bar-track: var(--g-100);
  --badge-pos-bg: var(--positive-bg);
  --badge-pos-text: var(--positive-text);
  --badge-neg-bg: var(--negative-bg);
  --badge-neg-text: var(--negative-text);
  --plotly-bg: #ffffff;
  --plotly-grid: rgba(0,0,0,0.06);
}}

/* ── DARK MODE ── */
.dark {{
  --page-bg: var(--g-950);
  --card-bg: var(--g-900);
  --card-border: var(--g-800);
  --shadow-card: 0 1px 3px rgba(0,0,0,0.30), 0 1px 2px rgba(0,0,0,0.20);
  --topbar-bg: var(--g-900);
  --topbar-border: var(--g-800);
  --heading: var(--g-50);
  --body-text: var(--g-300);
  --muted: var(--g-500);
  --input-bg: var(--g-800);
  --bar-track: var(--g-800);
  --badge-pos-bg: rgba(22,163,74,0.15);
  --badge-pos-text: #4ADE80;
  --badge-neg-bg: rgba(220,38,38,0.15);
  --badge-neg-text: #F87171;
  --plotly-bg: var(--g-900);
  --plotly-grid: rgba(255,255,255,0.06);
}}

/* ═══════════════════════════════════════════════════════
   RESET & BASE
   ═══════════════════════════════════════════════════════ */
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ scroll-behavior:smooth; }}
body {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: var(--page-bg);
  color: var(--body-text);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  transition: background 0.3s, color 0.3s;
  overflow-x: hidden;
}}

/* ═══════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════ */
.sidebar {{
  position: fixed;
  left: 0; top: 0;
  width: var(--sidebar-w);
  height: 100vh;
  background: linear-gradient(180deg, var(--sidebar-from), var(--sidebar-to));
  display: flex;
  flex-direction: column;
  z-index: 200;
  transition: width 0.3s cubic-bezier(0.4,0,0.2,1), transform 0.3s;
  overflow: hidden;
}}
.sidebar-logo {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 20px 16px;
  flex-shrink: 0;
}}
.sidebar-logo-icon {{
  width: 36px; height: 36px;
  background: var(--p-400);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}
.sidebar-logo-icon svg {{ width: 20px; height: 20px; }}
.sidebar-wordmark {{
  font-size: 1rem;
  font-weight: 600;
  color: #ffffff;
  white-space: nowrap;
  opacity: 1;
  transition: opacity 0.2s;
}}
.sidebar-nav {{
  flex: 1;
  overflow-y: auto;
  padding: 0 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: rgba(255,255,255,0.7);
  cursor: pointer;
  transition: all 0.2s;
  text-decoration: none;
  white-space: nowrap;
}}
.nav-item:hover {{
  background: rgba(255,255,255,0.08);
  color: #ffffff;
}}
.nav-item.active {{
  background: var(--p-400);
  color: var(--s-950);
  font-weight: 600;
}}
.nav-item svg {{
  width: 1.25rem; height: 1.25rem;
  flex-shrink: 0;
  stroke: currentColor;
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}}
.nav-label {{ transition: opacity 0.2s; }}
.sidebar-footer {{
  padding: 16px 12px;
  border-top: 1px solid rgba(255,255,255,0.1);
}}

/* ═══════════════════════════════════════════════════════
   TOPBAR
   ═══════════════════════════════════════════════════════ */
.topbar {{
  position: sticky;
  top: 0;
  height: var(--topbar-h);
  margin-left: var(--sidebar-w);
  background: var(--topbar-bg);
  border-bottom: 1px solid var(--topbar-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
  z-index: 100;
  transition: margin-left 0.3s, background 0.3s, border-color 0.3s;
}}
.topbar-left {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.hamburger {{
  display: none;
  background: none;
  border: none;
  cursor: pointer;
  color: var(--heading);
  padding: 4px;
}}
.hamburger svg {{ width: 24px; height: 24px; stroke: currentColor; fill: none; stroke-width: 2; }}
.breadcrumb {{
  font-size: 0.875rem;
  color: var(--muted);
  font-weight: 500;
}}
.topbar-right {{
  display: flex;
  align-items: center;
  gap: 16px;
}}
.topbar-btn {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 0.5rem;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  color: var(--body-text);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}}
.topbar-btn:hover {{
  background: var(--bar-track);
}}
.topbar-btn svg {{
  width: 1.25rem; height: 1.25rem;
  stroke: currentColor; fill: none; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round;
}}
.topbar-icon-btn {{
  position: relative;
  width: 40px; height: 40px;
  border-radius: 0.5rem;
  border: 1px solid var(--card-border);
  background: var(--card-bg);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--body-text);
}}
.topbar-icon-btn:hover {{ background: var(--bar-track); }}
.topbar-icon-btn svg {{
  width: 1.25rem; height: 1.25rem;
  stroke: currentColor; fill: none; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round;
}}
.notif-dot {{
  position: absolute;
  top: 6px; right: 6px;
  width: 8px; height: 8px;
  background: var(--p-400);
  border-radius: 50%;
  border: 2px solid var(--card-bg);
}}
.avatar {{
  width: 40px; height: 40px;
  border-radius: 50%;
  background: var(--s-700);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  border: 2px solid var(--card-border);
  transition: border-color 0.2s;
}}
.avatar:hover {{ border-color: var(--p-400); }}

/* ═══════════════════════════════════════════════════════
   MAIN CONTENT
   ═══════════════════════════════════════════════════════ */
.main {{
  margin-left: var(--sidebar-w);
  padding: 2rem;
  min-height: calc(100vh - var(--topbar-h));
  transition: margin-left 0.3s;
}}
.main-container {{
  max-width: 1536px;
  margin: 0 auto;
}}

/* Tab navigation (within main) */
.tab-bar {{
  display: flex;
  gap: 4px;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--card-border);
  padding-bottom: 0;
  overflow-x: auto;
  scrollbar-width: none;
}}
.tab-bar::-webkit-scrollbar {{ display:none; }}
.tab-btn {{
  padding: 10px 20px;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--muted);
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s;
  margin-bottom: -1px;
}}
.tab-btn:hover {{ color: var(--heading); }}
.tab-btn.active {{
  color: var(--p-500);
  border-bottom-color: var(--p-400);
}}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* ═══════════════════════════════════════════════════════
   KPI CARDS
   ═══════════════════════════════════════════════════════ */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--card-gap);
  margin-bottom: var(--card-gap);
}}
.kpi-card {{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  padding: var(--card-pad);
  box-shadow: var(--shadow-card);
  position: relative;
  transition: all 0.3s;
  overflow: hidden;
}}
.kpi-card:hover {{
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}}
.kpi-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}}
.kpi-title {{
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--muted);
}}
.kpi-icon {{
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  background: var(--p-100);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}}
.dark .kpi-icon {{ background: rgba(139,212,53,0.15); }}
.kpi-icon svg {{
  width: 1.25rem; height: 1.25rem;
  stroke: var(--p-600);
  fill: none; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round;
}}
.dark .kpi-icon svg {{ stroke: var(--p-400); }}
.kpi-value {{
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--heading);
  margin: 8px 0;
  transition: color 0.3s;
}}
.kpi-badge {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: var(--badge-radius);
  font-size: 0.75rem;
  font-weight: 600;
}}
.kpi-badge.positive {{
  background: var(--badge-pos-bg);
  color: var(--badge-pos-text);
}}
.kpi-badge.negative {{
  background: var(--badge-neg-bg);
  color: var(--badge-neg-text);
}}
.kpi-badge svg {{
  width: 12px; height: 12px;
  stroke: currentColor; fill: none;
  stroke-width: 2.5;
}}
.kpi-subtitle {{
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 4px;
}}
.kpi-sparkline {{
  height: 32px;
  margin-top: 12px;
}}
.kpi-sparkline svg {{
  width: 100%;
  height: 100%;
}}

/* ═══════════════════════════════════════════════════════
   CHART PANELS  — design spec grid
   ═══════════════════════════════════════════════════════ */
.charts-grid {{
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--card-gap);
  margin-bottom: var(--card-gap);
}}
.panel {{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  padding: var(--card-pad);
  box-shadow: var(--shadow-card);
  transition: all 0.3s;
}}
.panel:hover {{
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.panel-title {{
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--heading);
  margin-bottom: 16px;
  transition: color 0.3s;
}}
.panel-5 {{ grid-column: span 5; }}
.panel-3 {{ grid-column: span 3; }}
.panel-4 {{ grid-column: span 4; }}
.panel-6 {{ grid-column: span 6; }}
.panel-7 {{ grid-column: span 7; }}
.panel-12 {{ grid-column: span 12; }}
.chart-area {{ min-height: 320px; }}

/* ── Donut legend ── */
.donut-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  justify-content: center;
  margin-top: 12px;
}}
.donut-legend-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--body-text);
}}
.donut-legend-dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}}

/* ── Horizontal bars (top panels) ── */
.hbar-list {{
  display: flex;
  flex-direction: column;
  gap: 1rem;
}}
.hbar-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.hbar-label {{
  width: 120px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--heading);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
  transition: color 0.3s;
}}
.hbar-track {{
  flex: 1;
  height: 10px;
  background: var(--bar-track);
  border-radius: var(--badge-radius);
  overflow: hidden;
  transition: background 0.3s;
}}
.hbar-fill {{
  height: 100%;
  background: var(--s-800);
  border-radius: var(--badge-radius);
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}}
.dark .hbar-fill {{ background: var(--s-600); }}
.hbar-value {{
  width: 80px;
  text-align: right;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--muted);
  white-space: nowrap;
  flex-shrink: 0;
}}

/* ═══════════════════════════════════════════════════════
   FILTER BAR
   ═══════════════════════════════════════════════════════ */
.filter-bar {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: var(--card-gap);
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: var(--card-radius);
  padding: 12px var(--card-pad);
  box-shadow: var(--shadow-card);
  transition: all 0.3s;
}}
.filter-group {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.filter-group label {{
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.filter-input {{
  border: 1px solid var(--card-border);
  border-radius: 0.5rem;
  padding: 7px 12px;
  font-size: 0.875rem;
  color: var(--heading);
  background: var(--input-bg);
  transition: all 0.2s;
  outline: none;
  font-family: inherit;
}}
.filter-input:focus {{
  border-color: var(--p-400);
  box-shadow: 0 0 0 3px rgba(139,212,53,0.2);
}}
.filter-divider {{
  width: 1px;
  height: 28px;
  background: var(--card-border);
  flex-shrink: 0;
}}
.btn {{
  border: none;
  border-radius: 0.5rem;
  padding: 7px 16px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: inherit;
}}
.btn-primary {{
  background: var(--p-400);
  color: var(--s-950);
}}
.btn-primary:hover {{
  background: var(--p-500);
  box-shadow: 0 4px 12px rgba(139,212,53,0.3);
  transform: translateY(-1px);
}}
.btn-ghost {{
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--card-border);
}}
.btn-ghost:hover {{ background: var(--bar-track); }}
.btn-sm {{ padding: 5px 12px; font-size: 0.75rem; }}

/* Panel filter dropdown */
.dropdown-wrap {{ position: relative; }}
.dropdown-trigger {{
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--card-border);
  border-radius: 0.5rem;
  padding: 7px 12px;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--heading);
  background: var(--input-bg);
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}}
.dropdown-trigger:hover {{ border-color: var(--p-400); }}
.dd-badge {{
  background: var(--p-400);
  color: var(--s-950);
  border-radius: 10px;
  padding: 1px 8px;
  font-size: 0.75rem;
  font-weight: 700;
}}
.dropdown-menu {{
  display: none;
  position: absolute;
  top: calc(100% + 4px);
  left: 0;
  min-width: 260px;
  max-height: 320px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 0.75rem;
  box-shadow: 0 12px 32px rgba(0,0,0,0.15);
  z-index: 300;
  flex-direction: column;
  overflow: hidden;
  transition: background 0.3s;
}}
.dropdown-menu.open {{ display: flex; }}
.dd-actions {{
  display: flex;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--card-border);
}}
.dd-actions button {{
  border: none;
  background: none;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--p-500);
  cursor: pointer;
  padding: 2px;
  font-family: inherit;
}}
.dd-actions button:hover {{ text-decoration: underline; }}
.dd-list {{
  overflow-y: auto;
  padding: 4px 0;
}}
.dd-item {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 0.875rem;
  color: var(--heading);
  transition: background 0.15s;
}}
.dd-item:hover {{ background: rgba(139,212,53,0.08); }}
.dd-item input[type="checkbox"] {{
  width: 15px; height: 15px;
  cursor: pointer;
  accent-color: var(--p-400);
}}

/* ═══════════════════════════════════════════════════════
   COMPARISON TAB
   ═══════════════════════════════════════════════════════ */
.comparison-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: end;
  margin-bottom: var(--card-gap);
  background: var(--card-bg);
  border-radius: var(--card-radius);
  padding: var(--card-pad);
  border: 1px solid var(--card-border);
  box-shadow: var(--shadow-card);
  transition: all 0.3s;
}}
.comp-field {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.comp-field label {{
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

/* Savings banner */
.savings-banner {{
  border-radius: var(--card-radius);
  padding: 2rem;
  margin-bottom: var(--card-gap);
  color: #fff;
  position: relative;
  overflow: hidden;
}}
.savings-banner.positive {{ background: linear-gradient(135deg, #059669, #10b981); }}
.savings-banner.negative {{ background: linear-gradient(135deg, #d97706, #f59e0b); }}
.savings-banner h3 {{
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 16px;
}}
.savings-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
}}
.sav-item {{
  background: rgba(255,255,255,0.15);
  border-radius: 0.75rem;
  padding: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
}}
.sav-label {{
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  opacity: 0.9;
  margin-bottom: 8px;
}}
.sav-value {{
  font-size: 1.5rem;
  font-weight: 800;
  line-height: 1;
}}

/* Comparison metric cards */
.comp-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--card-gap);
  margin-bottom: var(--card-gap);
}}
.comp-card {{
  background: var(--card-bg);
  border-radius: var(--card-radius);
  padding: var(--card-pad);
  border: 2px solid var(--card-border);
  box-shadow: var(--shadow-card);
  transition: all 0.3s;
}}
.comp-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }}
.comp-card.savings {{ border-color: var(--positive-text); }}
.comp-card.increase {{ border-color: var(--warning); }}
.comp-card-label {{
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 12px;
}}
.comp-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}}
.comp-period {{
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.comp-val {{
  font-size: 1.125rem;
  font-weight: 700;
  color: var(--heading);
  transition: color 0.3s;
}}
.comp-delta {{
  margin-top: 12px;
  padding-top: 12px;
  border-top: 2px solid var(--card-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.delta-label {{
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
}}
.delta-val {{
  font-size: 1rem;
  font-weight: 800;
}}
.delta-val.pos {{ color: var(--positive-text); }}
.dark .delta-val.pos {{ color: #4ADE80; }}
.delta-val.neg {{ color: var(--warning); }}

/* ═══════════════════════════════════════════════════════
   DATA TABLE TAB
   ═══════════════════════════════════════════════════════ */
.table-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
}}
.table-search {{
  flex: 1;
  min-width: 200px;
  border: 1px solid var(--card-border);
  border-radius: 0.5rem;
  padding: 8px 14px;
  font-size: 0.875rem;
  color: var(--heading);
  background: var(--card-bg);
  outline: none;
  transition: all 0.2s;
  font-family: inherit;
}}
.table-search:focus {{
  border-color: var(--p-400);
  box-shadow: 0 0 0 3px rgba(139,212,53,0.2);
}}
.data-table-wrap {{
  background: var(--card-bg);
  border-radius: var(--card-radius);
  border: 1px solid var(--card-border);
  overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: all 0.3s;
}}
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}}
.data-table th {{
  background: var(--bar-track);
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
  border-bottom: 2px solid var(--card-border);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: all 0.3s;
}}
.data-table th:hover {{ color: var(--p-500); }}
.data-table th .sort-icon {{ margin-left: 4px; font-size: 0.625rem; }}
.data-table td {{
  padding: 8px 14px;
  border-bottom: 1px solid var(--card-border);
  color: var(--body-text);
  transition: all 0.3s;
}}
.data-table tr:hover td {{ background: rgba(139,212,53,0.06); }}
.data-table .anomaly td {{ background: var(--badge-neg-bg); }}
.table-footer {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  background: var(--bar-track);
  font-size: 0.75rem;
  color: var(--muted);
  transition: all 0.3s;
}}
.pagination {{ display: flex; gap: 4px; }}
.pagination button {{
  border: 1px solid var(--card-border);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 0.75rem;
  background: var(--card-bg);
  color: var(--body-text);
  cursor: pointer;
  transition: all 0.2s;
  font-family: inherit;
}}
.pagination button:hover {{ background: rgba(139,212,53,0.1); color: var(--p-500); }}
.pagination button.active {{ background: var(--p-400); color: var(--s-950); border-color: var(--p-400); }}

/* ═══════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════ */
.footer {{
  margin-left: var(--sidebar-w);
  padding: 1.5rem 2rem;
  text-align: center;
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 500;
  border-top: 1px solid var(--card-border);
  transition: all 0.3s;
}}

/* ═══════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════ */
@media (max-width: 1536px) {{
  .main-container {{ max-width: 100%; }}
}}
@media (max-width: 1023px) {{
  .sidebar {{ width: var(--sidebar-collapsed); }}
  .sidebar-wordmark {{ opacity: 0; width: 0; }}
  .nav-label {{ opacity: 0; width: 0; overflow: hidden; }}
  .topbar {{ margin-left: var(--sidebar-collapsed); }}
  .main {{ margin-left: var(--sidebar-collapsed); }}
  .footer {{ margin-left: var(--sidebar-collapsed); }}
  .charts-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .panel-5, .panel-3, .panel-4, .panel-7 {{ grid-column: span 1; }}
  .panel-12, .panel-6 {{ grid-column: span 2; }}
}}
@media (max-width: 767px) {{
  .sidebar {{ transform: translateX(-100%); }}
  .sidebar.open {{ transform: translateX(0); width: var(--sidebar-w); }}
  .topbar {{ margin-left: 0; }}
  .main {{ margin-left: 0; padding: 1rem; }}
  .footer {{ margin-left: 0; }}
  .hamburger {{ display: flex; }}
  .kpi-grid {{ grid-template-columns: 1fr; }}
  .charts-grid {{ grid-template-columns: 1fr; }}
  .panel-5, .panel-3, .panel-4, .panel-6, .panel-7, .panel-12 {{ grid-column: span 1; }}
  .comp-grid {{ grid-template-columns: 1fr; }}
  .comparison-controls {{ flex-direction: column; }}
}}
@media (min-width: 768px) and (max-width: 1023px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

/* ═══════════════════════════════════════════════════════
   ANIMATIONS
   ═══════════════════════════════════════════════════════ */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.animate {{ animation: fadeUp 0.4s ease forwards; }}
.kpi-card:nth-child(1) {{ animation-delay: 0.05s; }}
.kpi-card:nth-child(2) {{ animation-delay: 0.1s; }}
.kpi-card:nth-child(3) {{ animation-delay: 0.15s; }}

.hidden {{ display: none !important; }}

/* Sidebar overlay for mobile */
.sidebar-overlay {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  z-index: 190;
}}
.sidebar-overlay.active {{ display: block; }}
</style>
</head>
<body>

<!-- ═══════ SIDEBAR ═══════ -->
<aside class="sidebar" id="sidebar">
  <div class="sidebar-logo">
    <div class="sidebar-logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="#0B1E2E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
      </svg>
    </div>
    <span class="sidebar-wordmark">EMSlite</span>
  </div>
  <nav class="sidebar-nav">
    <a class="nav-item active" data-tab="overview">
      <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span class="nav-label">Dashboard</span>
    </a>
    <a class="nav-item" data-tab="analytics">
      <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <span class="nav-label">Analytics</span>
    </a>
    <a class="nav-item" data-tab="comparison">
      <svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
      <span class="nav-label">Comparison</span>
    </a>
    <a class="nav-item" data-tab="data">
      <svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
      <span class="nav-label">Data Table</span>
    </a>
  </nav>
  <div class="sidebar-footer">
    <a class="nav-item" style="cursor:default;opacity:0.5">
      <svg viewBox="0 0 24 24"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      <span class="nav-label">Logout</span>
    </a>
  </div>
</aside>
<div class="sidebar-overlay" id="sidebar-overlay"></div>

<!-- ═══════ TOPBAR ═══════ -->
<header class="topbar">
  <div class="topbar-left">
    <button class="hamburger" id="hamburger">
      <svg viewBox="0 0 24 24"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
    </button>
    <span class="breadcrumb">Energy Dashboard</span>
  </div>
  <div class="topbar-right">
    <button class="topbar-btn" id="theme-toggle">
      <svg viewBox="0 0 24 24" id="theme-icon"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
      <span id="theme-label">Dark</span>
    </button>
    <button class="topbar-icon-btn" title="Notifications">
      <svg viewBox="0 0 24 24"><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/></svg>
      <span class="notif-dot"></span>
    </button>
    <div class="avatar">EM</div>
  </div>
</header>

<!-- ═══════ MAIN ═══════ -->
<main class="main">
  <div class="main-container">

    <!-- Tab contents -->
    <!-- ═══════ OVERVIEW ═══════ -->
    <div class="tab-content active" id="tab-overview">
      <div class="filter-bar" id="ov-filter-bar"></div>
      <div class="kpi-grid" id="kpi-grid">
        <div class="kpi-card animate">
          <div class="kpi-header">
            <div>
              <div class="kpi-title">Total Energy Consumption</div>
              <div class="kpi-value" id="kpi-energy">0 kWh</div>
              <div class="kpi-badge positive" id="kpi-energy-badge">
                <svg viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
                <span id="kpi-energy-pct">0%</span>
              </div>
              <div class="kpi-subtitle">Compared Last Period</div>
            </div>
            <div class="kpi-icon">
              <svg viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
            </div>
          </div>
          <div class="kpi-sparkline" id="spark-energy"></div>
        </div>
        <div class="kpi-card animate">
          <div class="kpi-header">
            <div>
              <div class="kpi-title">Estimated Cost</div>
              <div class="kpi-value" id="kpi-cost">$0</div>
              <div class="kpi-badge positive" id="kpi-cost-badge">
                <svg viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
                <span id="kpi-cost-pct">0%</span>
              </div>
              <div class="kpi-subtitle">Compared Last Period</div>
            </div>
            <div class="kpi-icon">
              <svg viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>
            </div>
          </div>
          <div class="kpi-sparkline" id="spark-cost"></div>
        </div>
        <div class="kpi-card animate">
          <div class="kpi-header">
            <div>
              <div class="kpi-title">Peak Demand</div>
              <div class="kpi-value" id="kpi-peak">0 kW</div>
              <div class="kpi-badge positive" id="kpi-peak-badge">
                <svg viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/></svg>
                <span id="kpi-peak-pct">0%</span>
              </div>
              <div class="kpi-subtitle">Compared Last Period</div>
            </div>
            <div class="kpi-icon">
              <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            </div>
          </div>
          <div class="kpi-sparkline" id="spark-peak"></div>
        </div>
      </div>

      <!-- Charts Row — design spec grid -->
      <div class="charts-grid">
        <div class="panel panel-5">
          <div class="panel-title">Energy Use</div>
          <div class="chart-area" id="chart-energy-bars"></div>
        </div>
        <div class="panel panel-3">
          <div class="panel-title">Energy Utilisation</div>
          <div class="chart-area" id="chart-donut" style="min-height:280px"></div>
          <div class="donut-legend" id="donut-legend"></div>
        </div>
        <div class="panel panel-4">
          <div class="panel-title">Top Panels by Consumption</div>
          <div class="hbar-list" id="hbar-panels"></div>
        </div>
        <div class="panel panel-12">
          <div class="panel-title">Facility Load Profile</div>
          <div class="chart-area" id="chart-load"></div>
        </div>
      </div>

      <div id="meter-section" class="hidden">
        <div class="charts-grid">
          <div class="panel panel-12" id="meter-cards-panel">
            <div class="panel-title">Utility Meters</div>
            <div class="hbar-list" id="meter-bars"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══════ ANALYTICS ═══════ -->
    <div class="tab-content" id="tab-analytics">
      <div class="filter-bar" id="an-filter-bar"></div>
      <div class="charts-grid">
        <div class="panel panel-6">
          <div class="panel-title">Time-of-Day Heatmap</div>
          <div class="chart-area" id="chart-heatmap"></div>
        </div>
        <div class="panel panel-6">
          <div class="panel-title">Average Hourly Profile</div>
          <div class="chart-area" id="chart-hourly"></div>
        </div>
        <div class="panel panel-6">
          <div class="panel-title">Weekday Distribution</div>
          <div class="chart-area" id="chart-weekday"></div>
        </div>
        <div class="panel panel-6">
          <div class="panel-title">Daily Energy Consumption</div>
          <div class="chart-area" id="chart-daily-energy"></div>
        </div>
        <div class="panel panel-12" id="group-chart-panel" style="display:none">
          <div class="panel-title">Group Loads</div>
          <div class="chart-area" id="chart-groups"></div>
        </div>
        <div class="panel panel-12">
          <div class="panel-title">Panel Trends</div>
          <div class="chart-area" id="chart-panels"></div>
        </div>
      </div>
    </div>

    <!-- ═══════ COMPARISON ═══════ -->
    <div class="tab-content" id="tab-comparison">
      <div class="filter-bar" id="cp-filter-bar"></div>
      <div class="savings-banner positive" id="sav-banner">
        <h3 id="sav-title">Energy Savings Impact</h3>
        <div class="savings-grid">
          <div class="sav-item"><div class="sav-label">Energy Savings</div><div class="sav-value" id="sav-energy">0 kWh</div></div>
          <div class="sav-item"><div class="sav-label">Cost Savings</div><div class="sav-value" id="sav-cost">$0</div></div>
          <div class="sav-item"><div class="sav-label">Peak Reduction</div><div class="sav-value" id="sav-peak">0 kW</div></div>
          <div class="sav-item"><div class="sav-label">Load Factor Change</div><div class="sav-value" id="sav-lf">0%</div></div>
        </div>
      </div>
      <div class="comp-grid" id="comp-grid">
        <div class="comp-card" id="cc-energy"><div class="comp-card-label">Total Energy</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-e1">0</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-e2">0</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-ed">0</span></div></div>
        <div class="comp-card" id="cc-cost"><div class="comp-card-label">Total Cost</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-c1">$0</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-c2">$0</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-cd">$0</span></div></div>
        <div class="comp-card" id="cc-avg"><div class="comp-card-label">Average Load</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-a1">0</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-a2">0</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-ad">0</span></div></div>
        <div class="comp-card" id="cc-peak"><div class="comp-card-label">Peak Load</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-p1">0</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-p2">0</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-pd">0</span></div></div>
        <div class="comp-card" id="cc-lf"><div class="comp-card-label">Load Factor</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-l1">0%</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-l2">0%</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-ld">0</span></div></div>
        <div class="comp-card" id="cc-daily"><div class="comp-card-label">Daily Energy (Avg)</div>
          <div class="comp-row"><span class="comp-period">Period 1</span><span class="comp-val" id="cc-d1">0</span></div>
          <div class="comp-row"><span class="comp-period">Period 2</span><span class="comp-val" id="cc-d2">0</span></div>
          <div class="comp-delta"><span class="delta-label">Change</span><span class="delta-val" id="cc-dd">0</span></div></div>
      </div>
      <div class="charts-grid">
        <div class="panel panel-12"><div class="panel-title">Load Profile Overlay</div><div class="chart-area" id="chart-comp-load"></div></div>
        <div class="panel panel-6"><div class="panel-title">Hourly Comparison</div><div class="chart-area" id="chart-comp-hourly"></div></div>
        <div class="panel panel-6"><div class="panel-title">Weekday Comparison</div><div class="chart-area" id="chart-comp-weekday"></div></div>
      </div>
    </div>

    <!-- ═══════ DATA TABLE ═══════ -->
    <div class="tab-content" id="tab-data">
      <div class="filter-bar" id="dt-filter-bar"></div>
      <div class="table-controls">
        <input type="text" class="table-search" id="table-search" placeholder="Search timestamps, values..."/>
        <button class="btn btn-ghost btn-sm" id="export-csv">Export CSV</button>
      </div>
      <div class="data-table-wrap">
        <table class="data-table" id="data-table">
          <thead id="table-head"></thead>
          <tbody id="table-body"></tbody>
        </table>
        <div class="table-footer">
          <span id="table-info">0 rows</span>
          <div class="pagination" id="pagination"></div>
        </div>
      </div>
    </div>

  </div>
</main>

<footer class="footer">
  EMSlite Next-Gen Energy Dashboard &middot; Design tokens: Chartreuse #8BD435 &amp; Navy #0B1E2E
</footer>

<!-- ═══════════════════════════════════════════════════════
     JAVASCRIPT
     ═══════════════════════════════════════════════════════ -->
<script>
const D = {json.dumps(data_payload)};
const LOGO = {json.dumps(logo_path)};
const PRICE = {price_per_kwh};
const ALL_PANELS = D.panel_names || [];
const DATE_MIN = D.timestamps.length ? new Date(D.timestamps[0]).toISOString().slice(0,10) : "";
const DATE_MAX = D.timestamps.length ? new Date(D.timestamps[D.timestamps.length-1]).toISOString().slice(0,10) : "";

/* ─── Theme ─── */
const lightC = {{
  ink:"#3A424D", muted:"#9BA5B0", card:"#ffffff", bg:"#F8FAFB",
  grid:"rgba(0,0,0,0.06)", accent:"#8BD435", accentDark:"#0B1E2E",
  gradFrom:"#0B1E2E", gradTo:"#8BD435", donutA:"#8BD435", donutB:"#132D42",
  series:["#8BD435","#0B1E2E","#398EB9","#F97316","#EF4444","#38BDF8","#6DBF1A","#1D5B83"]
}};
const darkC = {{
  ink:"#E3E8EC", muted:"#6C7784", card:"#171C23", bg:"#0D1117",
  grid:"rgba(255,255,255,0.06)", accent:"#8BD435", accentDark:"#E3E8EC",
  gradFrom:"#132D42", gradTo:"#8BD435", donutA:"#8BD435", donutB:"#1B4A6C",
  series:["#8BD435","#38BDF8","#F97316","#EF4444","#74B6D5","#6DBF1A","#D4EAF2","#ABDF65"]
}};
let isDark = false;
function T() {{ return isDark ? darkC : lightC; }}

document.getElementById("theme-toggle").addEventListener("click", () => {{
  isDark = !isDark;
  document.documentElement.classList.toggle("dark", isDark);
  document.getElementById("theme-label").textContent = isDark ? "Light" : "Dark";
  document.getElementById("theme-icon").innerHTML = isDark
    ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
    : '<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>';
  renderCurrentTab();
}});

/* ─── Sidebar / Mobile ─── */
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
document.getElementById("hamburger").addEventListener("click", () => {{
  sidebar.classList.toggle("open");
  sidebarOverlay.classList.toggle("active");
}});
sidebarOverlay.addEventListener("click", () => {{
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("active");
}});

/* ─── Tab Navigation ─── */
let activeTab = "overview";
function switchTab(tabId) {{
  activeTab = tabId;
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".nav-item[data-tab]").forEach(n => n.classList.remove("active"));
  document.getElementById("tab-" + tabId).classList.add("active");
  document.querySelectorAll('.nav-item[data-tab="' + tabId + '"]').forEach(n => n.classList.add("active"));
  renderCurrentTab();
  window.dispatchEvent(new Event("resize"));
}}
document.querySelectorAll(".nav-item[data-tab]").forEach(n => {{
  n.addEventListener("click", e => {{
    e.preventDefault();
    switchTab(n.dataset.tab);
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("active");
  }});
}});

/* ═══════════════════════════════════════════════════════
   PER-TAB STATE & FILTER BAR BUILDER
   ═══════════════════════════════════════════════════════ */
const tabState = {{
  overview:   {{ panels: new Set(ALL_PANELS), startDate: DATE_MIN, endDate: DATE_MAX }},
  analytics:  {{ panels: new Set(ALL_PANELS), startDate: DATE_MIN, endDate: DATE_MAX }},
  comparison: {{ panels: new Set(ALL_PANELS), p1Start: "", p1End: "", p2Start: "", p2End: "" }},
  data:       {{ panels: new Set(ALL_PANELS), startDate: DATE_MIN, endDate: DATE_MAX }}
}};

/* Auto-init comparison periods */
(function() {{
  const totalDays = DATE_MIN && DATE_MAX ? (new Date(DATE_MAX) - new Date(DATE_MIN)) / 86400000 : 0;
  const st = tabState.comparison;
  if (totalDays >= 14) {{
    const e2 = new Date(DATE_MAX), s2 = new Date(e2); s2.setDate(s2.getDate()-6);
    const e1 = new Date(s2); e1.setDate(e1.getDate()-1);
    const s1 = new Date(e1); s1.setDate(s1.getDate()-6);
    st.p1Start = s1.toISOString().slice(0,10);
    st.p1End   = e1.toISOString().slice(0,10);
    st.p2Start = s2.toISOString().slice(0,10);
    st.p2End   = e2.toISOString().slice(0,10);
  }} else if (totalDays > 0) {{
    const mid = new Date(DATE_MIN); mid.setDate(mid.getDate()+Math.floor(totalDays/2));
    const midN = new Date(mid); midN.setDate(midN.getDate()+1);
    st.p1Start = DATE_MIN;
    st.p1End   = mid.toISOString().slice(0,10);
    st.p2Start = midN.toISOString().slice(0,10);
    st.p2End   = DATE_MAX;
  }}
}})();

/* Build a reusable filter bar inside a container element.
   mode = "daterange" (Overview / Analytics / Data Table)  or  "comparison" */
function buildFilterBar(containerId, tabKey, mode) {{
  const container = document.getElementById(containerId);
  if (!container) return;
  const st = tabState[tabKey];
  const uid = tabKey; // unique prefix for IDs

  let html = '';

  /* Panel dropdown */
  if (ALL_PANELS.length) {{
    html += `<div class="dropdown-wrap" id="dd-${{uid}}">
      <button class="dropdown-trigger" id="ddt-${{uid}}">Panels <span class="dd-badge" id="ddb-${{uid}}">All</span></button>
      <div class="dropdown-menu" id="ddm-${{uid}}">
        <div class="dd-actions">
          <button id="dda-${{uid}}">Select All</button>
          <button id="ddn-${{uid}}">Clear</button>
        </div>
        <div class="dd-list" id="ddl-${{uid}}"></div>
      </div>
    </div>
    <div class="filter-divider"></div>`;
  }}

  if (mode === "daterange") {{
    html += `<div class="filter-group"><label>From</label>
      <input type="date" id="fs-${{uid}}" class="filter-input" value="${{st.startDate}}" min="${{DATE_MIN}}" max="${{DATE_MAX}}"/></div>
    <div class="filter-group"><label>To</label>
      <input type="date" id="fe-${{uid}}" class="filter-input" value="${{st.endDate}}" min="${{DATE_MIN}}" max="${{DATE_MAX}}"/></div>`;
  }} else if (mode === "comparison") {{
    html += `<div class="filter-group"><label>Period 1 Start</label>
      <input type="date" id="cp1s-${{uid}}" class="filter-input" value="${{st.p1Start}}"/></div>
    <div class="filter-group"><label>Period 1 End</label>
      <input type="date" id="cp1e-${{uid}}" class="filter-input" value="${{st.p1End}}"/></div>
    <div class="filter-divider"></div>
    <div class="filter-group"><label>Period 2 Start</label>
      <input type="date" id="cp2s-${{uid}}" class="filter-input" value="${{st.p2Start}}"/></div>
    <div class="filter-group"><label>Period 2 End</label>
      <input type="date" id="cp2e-${{uid}}" class="filter-input" value="${{st.p2End}}"/></div>`;
  }}

  html += `<button class="btn btn-primary" id="apply-${{uid}}">Apply</button>
    <button class="btn btn-ghost" id="reset-${{uid}}">Reset</button>`;

  container.innerHTML = html;

  /* Wire panel dropdown */
  if (ALL_PANELS.length) {{
    const list = document.getElementById("ddl-" + uid);
    ALL_PANELS.forEach(p => {{
      const lbl = document.createElement("label"); lbl.className = "dd-item";
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.value = p; cb.checked = st.panels.has(p);
      cb.addEventListener("change", () => {{ cb.checked ? st.panels.add(p) : st.panels.delete(p); syncDDBadge(uid, st); }});
      const sp = document.createElement("span"); sp.textContent = p;
      lbl.appendChild(cb); lbl.appendChild(sp); list.appendChild(lbl);
    }});
    document.getElementById("ddt-" + uid).addEventListener("click", e => {{
      e.stopPropagation(); document.getElementById("ddm-" + uid).classList.toggle("open");
    }});
    document.addEventListener("click", e => {{
      const wrap = document.getElementById("dd-" + uid);
      if (wrap && !wrap.contains(e.target)) document.getElementById("ddm-" + uid).classList.remove("open");
    }});
    document.getElementById("dda-" + uid).addEventListener("click", () => {{
      st.panels = new Set(ALL_PANELS);
      document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = true);
      syncDDBadge(uid, st);
    }});
    document.getElementById("ddn-" + uid).addEventListener("click", () => {{
      st.panels.clear();
      document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = false);
      syncDDBadge(uid, st);
    }});
    syncDDBadge(uid, st);
  }}

  /* Wire Apply */
  document.getElementById("apply-" + uid).addEventListener("click", () => {{
    if (mode === "daterange") {{
      st.startDate = document.getElementById("fs-" + uid).value || DATE_MIN;
      st.endDate   = document.getElementById("fe-" + uid).value || DATE_MAX;
    }} else {{
      st.p1Start = document.getElementById("cp1s-" + uid).value;
      st.p1End   = document.getElementById("cp1e-" + uid).value;
      st.p2Start = document.getElementById("cp2s-" + uid).value;
      st.p2End   = document.getElementById("cp2e-" + uid).value;
    }}
    renderTab(tabKey);
  }});

  /* Wire Reset */
  document.getElementById("reset-" + uid).addEventListener("click", () => {{
    st.panels = new Set(ALL_PANELS);
    document.querySelectorAll("#ddl-" + uid + " input").forEach(c => c.checked = true);
    syncDDBadge(uid, st);
    if (mode === "daterange") {{
      st.startDate = DATE_MIN; st.endDate = DATE_MAX;
      document.getElementById("fs-" + uid).value = DATE_MIN;
      document.getElementById("fe-" + uid).value = DATE_MAX;
    }} else {{
      /* re-init comparison defaults */
      const totalDays = DATE_MIN && DATE_MAX ? (new Date(DATE_MAX) - new Date(DATE_MIN)) / 86400000 : 0;
      if (totalDays >= 14) {{
        const e2 = new Date(DATE_MAX), s2 = new Date(e2); s2.setDate(s2.getDate()-6);
        const e1 = new Date(s2); e1.setDate(e1.getDate()-1);
        const s1 = new Date(e1); s1.setDate(s1.getDate()-6);
        st.p1Start = s1.toISOString().slice(0,10); st.p1End = e1.toISOString().slice(0,10);
        st.p2Start = s2.toISOString().slice(0,10); st.p2End = e2.toISOString().slice(0,10);
      }} else {{
        const mid = new Date(DATE_MIN); mid.setDate(mid.getDate()+Math.floor(totalDays/2));
        const midN = new Date(mid); midN.setDate(midN.getDate()+1);
        st.p1Start = DATE_MIN; st.p1End = mid.toISOString().slice(0,10);
        st.p2Start = midN.toISOString().slice(0,10); st.p2End = DATE_MAX;
      }}
      document.getElementById("cp1s-" + uid).value = st.p1Start;
      document.getElementById("cp1e-" + uid).value = st.p1End;
      document.getElementById("cp2s-" + uid).value = st.p2Start;
      document.getElementById("cp2e-" + uid).value = st.p2End;
    }}
    renderTab(tabKey);
  }});
}}

function syncDDBadge(uid, st) {{
  const el = document.getElementById("ddb-" + uid);
  if (el) el.textContent = st.panels.size === ALL_PANELS.length ? "All" : st.panels.size === 0 ? "None" : st.panels.size;
}}

/* ═══════════════════════════════════════════════════════
   PER-TAB DATA FILTERING
   ═══════════════════════════════════════════════════════ */
function filterForTab(tabKey) {{
  const st = tabState[tabKey];
  const sD = st.startDate ? new Date(st.startDate + "T00:00:00Z") : null;
  const eD = st.endDate   ? new Date(st.endDate   + "T23:59:59Z") : null;
  const active = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  const aSet = new Set(active);
  const gMap = {{}}, mMap = {{}};
  D.group_definitions.forEach(g => {{ gMap[g.name] = (g.panels||[]).filter(p=>aSet.has(p)); }});
  D.utility_meters.forEach(m => {{ mMap[m.name] = (m.panels||[]).filter(p=>aSet.has(p)); }});
  const f = {{ timestamps:[], totalKw:[],
    groupSeries: Object.fromEntries(Object.keys(D.group_series).map(k=>[k,[]])),
    meterSeries: Object.fromEntries(Object.keys(D.meter_series).map(k=>[k,[]])),
    panelSeries: Object.fromEntries(ALL_PANELS.map(k=>[k,[]]))
  }};
  D.timestamps.forEach((ts,i) => {{
    const d = new Date(ts);
    if (sD && d < sD) return; if (eD && d > eD) return;
    f.timestamps.push(ts);
    let tot=0; active.forEach(p => {{ tot += (D.panel_series[p]||[])[i]||0; }}); f.totalKw.push(tot);
    Object.keys(D.group_series).forEach(g => {{
      let s=0; (gMap[g]||[]).forEach(p => {{ s += (D.panel_series[p]||[])[i]||0; }}); f.groupSeries[g].push(s);
    }});
    Object.keys(D.meter_series).forEach(m => {{
      let s=0; (mMap[m]||[]).forEach(p => {{ s += (D.panel_series[p]||[])[i]||0; }}); f.meterSeries[m].push(s);
    }});
    ALL_PANELS.forEach(p => {{ f.panelSeries[p].push((D.panel_series[p]||[])[i]||0); }});
  }});
  return f;
}}

/* ═══════════════════════════════════════════════════════
   SHARED HELPERS
   ═══════════════════════════════════════════════════════ */
function metrics(ts, kw) {{
  let kwh=0, pk=0, sum=0;
  for (let i=0;i<ts.length;i++) {{ const v=kw[i]??0; sum+=v; if(v>pk)pk=v;
    if(i>0) {{ const h=Math.max(0,(new Date(ts[i])-new Date(ts[i-1]))/3600000); kwh+=v*h; }}
  }}
  return {{ totalKwh:kwh, avgKw:ts.length?sum/ts.length:0, peakKw:pk }};
}}
function rollingMean(ts,v,wH) {{
  const wMs=wH*3600000, r=[]; let si=0,s=0;
  for(let i=0;i<ts.length;i++) {{ const ct=new Date(ts[i]).getTime(); s+=v[i]??0;
    while(ct-new Date(ts[si]).getTime()>wMs){{ s-=v[si]??0;si++; }}
    r.push((i-si+1)?s/(i-si+1):0);
  }} return r;
}}
function dailyEnergy(ts,kw) {{
  const ed={{}};
  for(let i=0;i<ts.length-1;i++) {{ const h=Math.max(0,(new Date(ts[i+1])-new Date(ts[i]))/3600000);
    const dk=new Date(ts[i]).toISOString().slice(0,10); if(!ed[dk])ed[dk]=0; ed[dk]+=(kw[i]??0)*h;
  }} const dates=Object.keys(ed).sort(); return {{ dates, values:dates.map(d=>ed[d]) }};
}}
function hourlyProfile(ts,kw) {{
  const s=Array(24).fill(0),c=Array(24).fill(0);
  ts.forEach((t,i)=>{{ const h=new Date(t).getUTCHours(); s[h]+=kw[i]??0; c[h]++; }});
  return {{ hours:Array.from({{length:24}},(_,i)=>i), avgs:s.map((v,i)=>c[i]?v/c[i]:0) }};
}}
function weekdayProfile(ts,kw) {{
  const s=Array(7).fill(0),c=Array(7).fill(0);
  ts.forEach((t,i)=>{{ const w=new Date(t).getUTCDay(); s[w]+=kw[i]??0; c[w]++; }});
  return {{ days:["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], avgs:s.map((v,i)=>c[i]?v/c[i]:0) }};
}}
function sparkSVG(vals, color) {{
  if(!vals.length) return "";
  const w=200,h=32, mn=Math.min(...vals), mx=Math.max(...vals), rng=mx-mn||1;
  const step=w/Math.max(vals.length-1,1);
  const pts=vals.map((v,i)=>`${{(i*step).toFixed(1)}},${{(h-((v-mn)/rng)*h*0.8-h*0.1).toFixed(1)}}`);
  const poly=pts.join(" ");
  return `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none">
    <polygon points="0,${{h}} ${{poly}} ${{((vals.length-1)*step).toFixed(1)}},${{h}}" fill="${{color}}" opacity="0.12"/>
    <polyline points="${{poly}}" fill="none" stroke="${{color}}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
}}

/* ─── Plotly helpers ─── */
function pLayout(ov) {{
  const t=T();
  return Object.assign({{ margin:{{t:16,l:55,r:20,b:45}}, paper_bgcolor:t.card, plot_bgcolor:t.card,
    font:{{family:"Inter,system-ui,sans-serif",color:t.ink,size:12}}, colorway:t.series,
    hovermode:"x unified", hoverlabel:{{bgcolor:t.ink,font:{{color:t.card}}}} }}, ov);
}}
function xA(ov) {{ const t=T(); return Object.assign({{gridcolor:t.grid,zerolinecolor:t.grid,showline:true,linecolor:t.grid}},ov); }}
function yA(ov) {{ const t=T(); return Object.assign({{rangemode:"tozero",gridcolor:t.grid,zerolinecolor:t.grid,showline:true,linecolor:t.grid}},ov); }}
const pCfg = {{displaylogo:false, responsive:true}};
function weekendShapes(ts) {{
  if(!ts.length)return[];const shapes=[];let ws=null;
  for(let i=0;i<ts.length;i++){{ const d=new Date(ts[i]).getUTCDay();const isWe=d===0||d===6;
    if(isWe&&ws===null)ws=ts[i]; if(!isWe&&ws!==null){{ shapes.push({{type:"rect",xref:"x",yref:"paper",x0:ws,x1:ts[i],y0:0,y1:1,line:{{width:0}},fillcolor:"rgba(139,212,53,0.06)"}}); ws=null; }}
  }} if(ws!==null)shapes.push({{type:"rect",xref:"x",yref:"paper",x0:ws,x1:ts[ts.length-1],y0:0,y1:1,line:{{width:0}},fillcolor:"rgba(139,212,53,0.06)"}}); return shapes;
}}

/* ═══════════════════════════════════════════════════════
   TAB RENDERERS — each is self-contained
   ═══════════════════════════════════════════════════════ */

/* ═══════ OVERVIEW ═══════ */
function renderOverview() {{
  const data = filterForTab("overview");
  const st = tabState.overview;
  const t=T(), m=metrics(data.timestamps,data.totalKw), cost=m.totalKwh*PRICE;
  const mid=Math.floor(data.timestamps.length/2);
  const m1=metrics(data.timestamps.slice(0,mid),data.totalKw.slice(0,mid));
  const m2=metrics(data.timestamps.slice(mid),data.totalKw.slice(mid));

  function setBadge(id, v1, v2, invert) {{
    const pct=v1?((v2-v1)/v1*100):0;
    const el=document.getElementById(id+"-badge");
    const pctEl=document.getElementById(id+"-pct");
    const isUp=pct>0, isBad=invert?isUp:!isUp;
    el.className="kpi-badge "+(isBad?"negative":"positive");
    el.querySelector("svg").innerHTML=isUp
      ?'<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
      :'<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>';
    pctEl.textContent=(isUp?"+":"")+pct.toFixed(1)+"%";
  }}

  document.getElementById("kpi-energy").textContent=m.totalKwh.toFixed(0)+" kWh";
  document.getElementById("kpi-cost").textContent="$"+cost.toFixed(0);
  document.getElementById("kpi-peak").textContent=m.peakKw.toFixed(1)+" kW";
  setBadge("kpi-energy",m1.totalKwh,m2.totalKwh,true);
  setBadge("kpi-cost",m1.totalKwh*PRICE,m2.totalKwh*PRICE,true);
  setBadge("kpi-peak",m1.peakKw,m2.peakKw,true);

  const de=dailyEnergy(data.timestamps,data.totalKw);
  document.getElementById("spark-energy").innerHTML=sparkSVG(de.values,"#8BD435");
  document.getElementById("spark-cost").innerHTML=sparkSVG(de.values.map(v=>v*PRICE),"#8BD435");
  const byDay={{}};
  data.timestamps.forEach((ts,i)=>{{ const dk=new Date(ts).toISOString().slice(0,10); if(!byDay[dk])byDay[dk]=[]; byDay[dk].push(data.totalKw[i]??0); }});
  const dayPeaks=Object.keys(byDay).sort().map(dk=>Math.max(...byDay[dk]));
  document.getElementById("spark-peak").innerHTML=sparkSVG(dayPeaks,"#8BD435");

  Plotly.newPlot("chart-energy-bars",[{{
    x:de.dates, y:de.values, type:"bar",
    marker:{{ color:de.values.map((_,i)=>{{ const frac=de.values.length>1?i/(de.values.length-1):0;
      return `rgb(${{Math.round(11+(139-11)*frac)}},${{Math.round(30+(212-30)*frac)}},${{Math.round(46+(53-46)*frac)}})`;
    }}) }}
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Date",font:{{size:12}}}},type:"category"}}),
    yaxis:yA({{title:{{text:"kWh",font:{{size:12}}}}}}), bargap:0.2
  }}),pCfg);

  const lf=m.peakKw>0?m.avgKw/m.peakKw*100:0, offPeak=100-lf;
  Plotly.newPlot("chart-donut",[{{
    values:[lf,offPeak], labels:["Active Load","Reserve Capacity"],
    type:"pie", hole:0.65, marker:{{ colors:[t.donutA, t.donutB] }},
    textinfo:"none", hovertemplate:"%{{label}}: %{{value:.1f}}%<extra></extra>"
  }}],pLayout({{ margin:{{t:10,b:10,l:10,r:10}}, showlegend:false, height:280,
    annotations:[{{ text:lf.toFixed(0)+"%", font:{{size:28,color:t.ink,family:"Inter"}}, showarrow:false }}]
  }}),pCfg);
  document.getElementById("donut-legend").innerHTML=`
    <div class="donut-legend-item"><div class="donut-legend-dot" style="background:${{t.donutA}}"></div>Active Load — ${{lf.toFixed(1)}}%</div>
    <div class="donut-legend-item"><div class="donut-legend-dot" style="background:${{t.donutB}}"></div>Reserve Capacity — ${{offPeak.toFixed(1)}}%</div>`;

  const active = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  const panelEnergies = active.map(p => {{
    const series=data.panelSeries[p]||[];
    return {{ name:p, kwh:metrics(data.timestamps,series).totalKwh }};
  }}).sort((a,b)=>b.kwh-a.kwh).slice(0,8);
  const maxKwh=panelEnergies.length?panelEnergies[0].kwh:1;
  document.getElementById("hbar-panels").innerHTML=panelEnergies.map(pe=>`
    <div class="hbar-row">
      <div class="hbar-label">${{pe.name}}</div>
      <div class="hbar-track"><div class="hbar-fill" style="width:${{(pe.kwh/maxKwh*100).toFixed(1)}}%"></div></div>
      <div class="hbar-value">${{pe.kwh>=1000?(pe.kwh/1000).toFixed(1)+"k":pe.kwh.toFixed(0)}} kWh</div>
    </div>`).join("");

  const rolling=rollingMean(data.timestamps,data.totalKw,D.rolling_hours);
  Plotly.newPlot("chart-load",[{{
    x:data.timestamps, y:rolling, mode:"lines",
    line:{{color:t.accent,width:2.5,shape:"spline"}},
    fill:"tozeroy", fillcolor:"rgba(139,212,53,0.1)",
    hovertemplate:"%{{x}}<br>%{{y:.1f}} kW<extra></extra>"
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Time",font:{{size:12}}}},type:"date"}}),
    yaxis:yA({{title:{{text:"kW",font:{{size:12}}}}}})
  }}),pCfg);

  if(D.utility_meters.length) {{
    document.getElementById("meter-section").classList.remove("hidden");
    const meterEnergies=D.utility_meters.map(mt=>{{
      const s=data.meterSeries[mt.name]||[];
      return {{name:mt.name, kwh:metrics(data.timestamps,s).totalKwh}};
    }});
    const mMax=meterEnergies.length?Math.max(...meterEnergies.map(m=>m.kwh)):1;
    document.getElementById("meter-bars").innerHTML=meterEnergies.map(me=>`
      <div class="hbar-row">
        <div class="hbar-label">${{me.name}}</div>
        <div class="hbar-track"><div class="hbar-fill" style="width:${{(me.kwh/(mMax||1)*100).toFixed(1)}}%"></div></div>
        <div class="hbar-value">${{me.kwh>=1000?(me.kwh/1000).toFixed(1)+"k":me.kwh.toFixed(0)}} kWh</div>
      </div>`).join("");
  }}
}}

/* ═══════ ANALYTICS ═══════ */
function renderAnalytics() {{
  const data = filterForTab("analytics");
  const st = tabState.analytics;
  const t=T();

  const bk={{}};
  data.timestamps.forEach((ts,i)=>{{ const d=new Date(ts),dk=d.toISOString().slice(0,10),h=d.getUTCHours();
    if(!bk[dk])bk[dk]={{}}; if(!bk[dk][h])bk[dk][h]={{s:0,c:0}}; bk[dk][h].s+=data.totalKw[i]??0; bk[dk][h].c++; }});
  const dates=Object.keys(bk).sort(), hours=Array.from({{length:24}},(_,i)=>i);
  const z=hours.map(h=>dates.map(d=>{{ const b=(bk[d]||{{}})[h]; return b?b.s/b.c:null; }}));
  Plotly.newPlot("chart-heatmap",[{{
    x:dates,y:hours,z, type:"heatmap",
    colorscale:isDark?[[0,"#0D1117"],[0.3,"#132D42"],[0.6,"#1D5B83"],[0.8,"#8BD435"],[1,"#F4FDE8"]]
      :[[0,"#F8FAFB"],[0.3,"#E3E8EC"],[0.6,"#6C7784"],[0.8,"#1B4A6C"],[1,"#8BD435"]],
    zsmooth:"best",connectgaps:true,
    colorbar:{{title:{{text:"kW",font:{{size:11}}}},thickness:15}}
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Date",font:{{size:12}}}},type:"category"}}),
    yaxis:Object.assign(yA({{title:{{text:"Hour",font:{{size:12}}}}}}),{{autorange:"reversed",rangemode:undefined}})
  }}),pCfg);

  const hp=hourlyProfile(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-hourly",[{{
    x:hp.hours,y:hp.avgs, mode:"lines+markers",
    line:{{color:t.accentDark,width:2.5,shape:"spline"}},
    marker:{{size:7,color:t.accentDark,line:{{color:t.card,width:2}}}},
    fill:"tozeroy",fillcolor:t.accentDark+"12"
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Hour (UTC)",font:{{size:12}}}},dtick:2}}),
    yaxis:yA({{title:{{text:"Avg kW",font:{{size:12}}}}}})
  }}),pCfg);

  const wp=weekdayProfile(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-weekday",[{{
    x:wp.days,y:wp.avgs, type:"bar", marker:{{color:t.accent}}
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Day",font:{{size:12}}}}}}),
    yaxis:yA({{title:{{text:"Avg kW",font:{{size:12}}}}}}), bargap:0.2
  }}),pCfg);

  const de=dailyEnergy(data.timestamps,data.totalKw);
  Plotly.newPlot("chart-daily-energy",[{{
    x:de.dates,y:de.values, type:"bar", marker:{{color:t.accentDark}}
  }}],pLayout({{
    xaxis:xA({{title:{{text:"Date",font:{{size:12}}}},type:"category"}}),
    yaxis:yA({{title:{{text:"kWh",font:{{size:12}}}}}}), bargap:0.15
  }}),pCfg);

  const gNames=Object.keys(data.groupSeries).filter(g=>{{ const def=D.group_definitions.find(d=>d.name===g); return def&&def.panels&&def.panels.length; }});
  if(gNames.length) {{
    document.getElementById("group-chart-panel").style.display="";
    Plotly.newPlot("chart-groups",gNames.map(n=>({{
      x:data.timestamps,y:data.groupSeries[n], mode:"lines",name:n,line:{{width:2.5,shape:"spline"}}
    }})),pLayout({{
      legend:{{orientation:"h",y:-0.15}},
      xaxis:xA({{title:{{text:"Time",font:{{size:12}}}},type:"date"}}),
      yaxis:yA({{title:{{text:"kW",font:{{size:12}}}}}})
    }}),pCfg);
  }}

  const pShow = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  if(pShow.length) {{
    Plotly.newPlot("chart-panels",pShow.map(p=>({{
      x:data.timestamps, y:data.panelSeries[p]||[], mode:"lines",name:p,line:{{width:2,shape:"spline"}}
    }})),pLayout({{
      legend:{{orientation:"h",y:-0.15}},
      xaxis:xA({{title:{{text:"Time",font:{{size:12}}}},type:"date"}}),
      yaxis:yA({{title:{{text:"kW",font:{{size:12}}}}}}),
      shapes:weekendShapes(data.timestamps)
    }}),pCfg);
  }}
}}

/* ═══════ COMPARISON ═══════ */
function renderComparison() {{
  const st = tabState.comparison;
  const t=T();
  const active = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  /* Build total kW from raw D using this tab's panel selection */
  const allTs = D.timestamps, allKw = [];
  for (let i = 0; i < allTs.length; i++) {{
    let tot = 0; active.forEach(p => {{ tot += (D.panel_series[p]||[])[i]||0; }}); allKw.push(tot);
  }}

  const p1s=new Date(st.p1Start+"T00:00:00Z"), p1e=new Date(st.p1End+"T23:59:59Z");
  const p2s=new Date(st.p2Start+"T00:00:00Z"), p2e=new Date(st.p2End+"T23:59:59Z");
  if(isNaN(p1s)||isNaN(p1e)||isNaN(p2s)||isNaN(p2e)) return;

  const pd1={{ts:[],kw:[]}},pd2={{ts:[],kw:[]}};
  allTs.forEach((ts,i)=>{{ const d=new Date(ts);
    if(d>=p1s&&d<=p1e){{ pd1.ts.push(ts); pd1.kw.push(allKw[i]); }}
    if(d>=p2s&&d<=p2e){{ pd2.ts.push(ts); pd2.kw.push(allKw[i]); }}
  }});
  const m1=metrics(pd1.ts,pd1.kw), m2=metrics(pd2.ts,pd2.kw);
  const c1=m1.totalKwh*PRICE, c2=m2.totalKwh*PRICE;
  const lf1=m1.peakKw>0?m1.avgKw/m1.peakKw*100:0, lf2=m2.peakKw>0?m2.avgKw/m2.peakKw*100:0;
  const eSav=m1.totalKwh-m2.totalKwh, cSav=c1-c2, pRed=m1.peakKw-m2.peakKw, lfC=lf2-lf1;

  const ban=document.getElementById("sav-banner");
  if(eSav>0) {{
    ban.className="savings-banner positive";
    document.getElementById("sav-title").textContent="Energy Savings Impact";
    document.getElementById("sav-energy").textContent=eSav.toFixed(0)+" kWh";
    document.getElementById("sav-cost").textContent="$"+cSav.toFixed(0);
  }} else {{
    ban.className="savings-banner negative";
    document.getElementById("sav-title").textContent="Energy Increase";
    document.getElementById("sav-energy").textContent=(-eSav).toFixed(0)+" kWh increase";
    document.getElementById("sav-cost").textContent="$"+(-cSav).toFixed(0)+" increase";
  }}
  document.getElementById("sav-peak").textContent=pRed.toFixed(1)+" kW";
  document.getElementById("sav-lf").textContent=lfC.toFixed(1)+" pts";

  function setCC(id,cls) {{ document.getElementById("cc-"+id).className="comp-card "+cls; }}
  document.getElementById("cc-e1").textContent=m1.totalKwh.toFixed(0)+" kWh";
  document.getElementById("cc-e2").textContent=m2.totalKwh.toFixed(0)+" kWh";
  const ced=document.getElementById("cc-ed"); ced.textContent=(m2.totalKwh-m1.totalKwh).toFixed(0)+" kWh";
  ced.className="delta-val "+(eSav>0?"pos":"neg"); setCC("energy",eSav>0?"savings":"increase");

  document.getElementById("cc-c1").textContent="$"+c1.toFixed(0);
  document.getElementById("cc-c2").textContent="$"+c2.toFixed(0);
  const ccd=document.getElementById("cc-cd"); ccd.textContent="$"+(c2-c1).toFixed(0);
  ccd.className="delta-val "+(cSav>0?"pos":"neg"); setCC("cost",cSav>0?"savings":"increase");

  document.getElementById("cc-a1").textContent=m1.avgKw.toFixed(1)+" kW";
  document.getElementById("cc-a2").textContent=m2.avgKw.toFixed(1)+" kW";
  const cad=document.getElementById("cc-ad"); cad.textContent=(m2.avgKw-m1.avgKw).toFixed(1)+" kW";
  cad.className="delta-val "+(m2.avgKw<m1.avgKw?"pos":"neg"); setCC("avg",m2.avgKw<m1.avgKw?"savings":"increase");

  document.getElementById("cc-p1").textContent=m1.peakKw.toFixed(1)+" kW";
  document.getElementById("cc-p2").textContent=m2.peakKw.toFixed(1)+" kW";
  const cpd=document.getElementById("cc-pd"); cpd.textContent=(m2.peakKw-m1.peakKw).toFixed(1)+" kW";
  cpd.className="delta-val "+(pRed>0?"pos":"neg"); setCC("peak",pRed>0?"savings":"increase");

  document.getElementById("cc-l1").textContent=lf1.toFixed(1)+"%";
  document.getElementById("cc-l2").textContent=lf2.toFixed(1)+"%";
  const cld=document.getElementById("cc-ld"); cld.textContent=lfC.toFixed(1)+" pts";
  cld.className="delta-val "+(lfC>0?"pos":"neg"); setCC("lf",lfC>0?"savings":"increase");

  const p1D=pd1.ts.length>0?Math.max(1,(new Date(pd1.ts[pd1.ts.length-1])-new Date(pd1.ts[0]))/86400000+1):1;
  const p2D=pd2.ts.length>0?Math.max(1,(new Date(pd2.ts[pd2.ts.length-1])-new Date(pd2.ts[0]))/86400000+1):1;
  document.getElementById("cc-d1").textContent=(m1.totalKwh/p1D).toFixed(0)+" kWh/d";
  document.getElementById("cc-d2").textContent=(m2.totalKwh/p2D).toFixed(0)+" kWh/d";
  const cdd=document.getElementById("cc-dd"); const dDiff=(m2.totalKwh/p2D)-(m1.totalKwh/p1D);
  cdd.textContent=dDiff.toFixed(0)+" kWh/d"; cdd.className="delta-val "+(dDiff<0?"pos":"neg");
  setCC("daily",dDiff<0?"savings":"increase");

  Plotly.newPlot("chart-comp-load",[
    {{x:pd1.ts,y:pd1.kw,mode:"lines",name:"Period 1",line:{{color:t.accent,width:2.5,shape:"spline"}}}},
    {{x:pd2.ts,y:pd2.kw,mode:"lines",name:"Period 2",line:{{color:t.accentDark,width:2.5,shape:"spline"}}}}
  ],pLayout({{ legend:{{orientation:"h",y:-0.15}},
    xaxis:xA({{title:{{text:"Time",font:{{size:12}}}},type:"date"}}),
    yaxis:yA({{title:{{text:"kW",font:{{size:12}}}}}})
  }}),pCfg);

  const hp1=hourlyProfile(pd1.ts,pd1.kw),hp2=hourlyProfile(pd2.ts,pd2.kw);
  Plotly.newPlot("chart-comp-hourly",[
    {{x:hp1.hours,y:hp1.avgs,mode:"lines+markers",name:"P1",line:{{color:t.accent,width:2.5,shape:"spline"}},marker:{{size:7,color:t.accent}}}},
    {{x:hp2.hours,y:hp2.avgs,mode:"lines+markers",name:"P2",line:{{color:t.accentDark,width:2.5,shape:"spline"}},marker:{{size:7,color:t.accentDark}}}}
  ],pLayout({{ legend:{{orientation:"h",y:-0.15}},
    xaxis:xA({{title:{{text:"Hour",font:{{size:12}}}},dtick:2}}),
    yaxis:yA({{title:{{text:"Avg kW",font:{{size:12}}}}}})
  }}),pCfg);

  const wp1=weekdayProfile(pd1.ts,pd1.kw),wp2=weekdayProfile(pd2.ts,pd2.kw);
  Plotly.newPlot("chart-comp-weekday",[
    {{x:wp1.days,y:wp1.avgs,type:"bar",name:"P1",marker:{{color:t.accent}}}},
    {{x:wp2.days,y:wp2.avgs,type:"bar",name:"P2",marker:{{color:t.accentDark}}}}
  ],pLayout({{ legend:{{orientation:"h",y:-0.15}},
    xaxis:xA({{title:{{text:"Day",font:{{size:12}}}}}}),
    yaxis:yA({{title:{{text:"Avg kW",font:{{size:12}}}}}}), barmode:"group"
  }}),pCfg);
}}

/* ═══════ DATA TABLE ═══════ */
let tData=[],tSortCol=0,tSortAsc=true,tPage=0;
const PAGE_SZ=50;
function renderDataTable() {{
  const data = filterForTab("data");
  const st = tabState.data;
  const panels = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  tData=data.timestamps.map((ts,i)=>{{ const r=[ts,(data.totalKw[i]??0).toFixed(2)];
    panels.forEach(p=>r.push((data.panelSeries[p]?.[i]??0).toFixed(2))); return r; }});
  const thead=document.getElementById("table-head");
  thead.innerHTML=`<tr><th data-col="0">Timestamp <span class="sort-icon">&#9650;</span></th>
    <th data-col="1">Total kW <span class="sort-icon"></span></th>
    ${{panels.map((p,i)=>`<th data-col="${{i+2}}">${{p}} <span class="sort-icon"></span></th>`).join("")}}</tr>`;
  thead.querySelectorAll("th").forEach(th=>{{
    th.addEventListener("click",()=>{{ const c=parseInt(th.dataset.col);
      if(tSortCol===c)tSortAsc=!tSortAsc; else{{tSortCol=c;tSortAsc=true;}} renderTableRows(); }});
  }});
  tSortCol=0;tSortAsc=true;tPage=0; renderTableRows();
}}
function renderTableRows() {{
  const q=document.getElementById("table-search").value.toLowerCase();
  let f=tData; if(q) f=tData.filter(r=>r.some(c=>c.toLowerCase().includes(q)));
  f.sort((a,b)=>{{ let va=a[tSortCol],vb=b[tSortCol];
    if(tSortCol>0){{va=parseFloat(va);vb=parseFloat(vb);}}
    if(va<vb)return tSortAsc?-1:1; if(va>vb)return tSortAsc?1:-1; return 0; }});
  const tv=f.map(r=>parseFloat(r[1])), mean=tv.reduce((a,b)=>a+b,0)/(tv.length||1);
  const std=Math.sqrt(tv.reduce((a,b)=>a+(b-mean)**2,0)/(tv.length||1));
  const anomT=mean+2*std;
  const tp=Math.ceil(f.length/PAGE_SZ); tPage=Math.min(tPage,Math.max(0,tp-1));
  const s=tPage*PAGE_SZ, pg=f.slice(s,s+PAGE_SZ);
  document.getElementById("table-body").innerHTML=pg.map(r=>{{
    const isA=parseFloat(r[1])>anomT;
    return `<tr${{isA?' class="anomaly"':''}}>${{r.map(c=>`<td>${{c}}</td>`).join("")}}</tr>`;
  }}).join("");
  document.getElementById("table-info").textContent=`Showing ${{s+1}}-${{Math.min(s+PAGE_SZ,f.length)}} of ${{f.length}}`;
  const pag=document.getElementById("pagination"); pag.innerHTML="";
  if(tPage>0){{ const b=document.createElement("button"); b.textContent="Prev"; b.addEventListener("click",()=>{{tPage--;renderTableRows();}}); pag.appendChild(b); }}
  const sP=Math.max(0,tPage-3),eP=Math.min(tp,sP+7);
  for(let p=sP;p<eP;p++){{ const b=document.createElement("button"); b.textContent=p+1; if(p===tPage)b.className="active";
    b.addEventListener("click",()=>{{tPage=p;renderTableRows();}}); pag.appendChild(b); }}
  if(tPage<tp-1){{ const b=document.createElement("button"); b.textContent="Next"; b.addEventListener("click",()=>{{tPage++;renderTableRows();}}); pag.appendChild(b); }}
  document.querySelectorAll("#table-head th").forEach(th=>{{ const ic=th.querySelector(".sort-icon"), c=parseInt(th.dataset.col);
    ic.innerHTML=c===tSortCol?(tSortAsc?"&#9650;":"&#9660;"):""; }});
}}

/* ─── CSV Export ─── */
document.getElementById("export-csv").addEventListener("click",()=>{{
  const st = tabState.data;
  const panels = st.panels.size ? Array.from(st.panels) : ALL_PANELS;
  let csv=["Timestamp","Total_kW",...panels].join(",")+"\\n";
  tData.forEach(r=>{{ csv+=r.join(",")+"\\n"; }});
  const blob=new Blob([csv],{{type:"text/csv"}});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="energy_export.csv"; a.click(); URL.revokeObjectURL(a.href);
}});
document.getElementById("table-search").addEventListener("input",()=>{{ tPage=0; renderTableRows(); }});

/* ═══════════════════════════════════════════════════════
   TAB DISPATCH
   ═══════════════════════════════════════════════════════ */
function renderTab(tabKey) {{
  if      (tabKey === "overview")   renderOverview();
  else if (tabKey === "analytics")  renderAnalytics();
  else if (tabKey === "comparison") renderComparison();
  else if (tabKey === "data")       renderDataTable();
}}

function renderCurrentTab() {{
  renderTab(activeTab);
}}

/* ═══════════════════════════════════════════════════════
   INIT — build filter bars, render default tab
   ═══════════════════════════════════════════════════════ */
buildFilterBar("ov-filter-bar", "overview",   "daterange");
buildFilterBar("an-filter-bar", "analytics",  "daterange");
buildFilterBar("cp-filter-bar", "comparison", "comparison");
buildFilterBar("dt-filter-bar", "data",       "daterange");

renderOverview();
</script>
</body>
</html>"""


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

    df = add_usage_columns(df, meters)
    output_path = build_next_gen_dashboard(df, output_dir, rolling_window)
    print(f"Generated next-gen dashboard: {output_path}")


if __name__ == "__main__":
    main()
