#!/usr/bin/env python3
"""Reconcile computed energy against utility bills — over-reporting diagnostic.

Standalone tool (no running server, no app config cache). Reconciles energy
**per meter**, since a facility typically has one bill per utility meter and
panels are split across meters. It loads the master CSV directly, integrates
kWh over a date range for the panels on the chosen meter, and reports the
computed-vs-bill ratio. It also audits for two over-reporting traps:

  * an aggregate / main-feed column being summed on top of its branch panels;
  * a panel assigned to more than one meter (cross-meter double-count).

Meter → panel mapping is resolved in this order:
  1. ``--panels`` explicit list (highest priority)
  2. ``utility_meters`` in the config file (if a matching name has panels)
  3. the database ``Device.meter_name`` (same source the API uses)

Examples
--------
    # See the meters and their panels (and any overlaps)
    python scripts/reconcile_bill.py --list-meters

    # Reconcile one meter against its bill
    python scripts/reconcile_bill.py --meter "Meter A" \
        --start 2026-05-01 --end 2026-05-31 --pf 0.82 --voltage 470 --bill-kwh 80000

    # Explicit panel list instead of a meter name
    python scripts/reconcile_bill.py --panels Panel_1,Panel_2 \
        --start 2026-05-01 --end 2026-05-31 --bill-kwh 80000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emslite.config import load_config  # noqa: E402
from emslite.core import amps_to_kw, excluded_columns, load_csv, meter_columns  # noqa: E402
from emslite.metrics import integrate_kwh  # noqa: E402

# Substrings that suggest a column is an aggregate / main feed (and therefore
# already includes other branch panels — summing it double-counts).
AGGREGATE_HINTS = ("total", "main", "mdp", "utility", "sum", "feed", "service", "incomer")


def _master_path(cfg: dict, root: Path) -> Path:
    data_dir = root / cfg.get("data_dir", "data")
    return data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")


def _meters_from_config(cfg: dict) -> dict[str, list[str]]:
    """{meter_name: [panel_ids]} from config utility_meters (only non-empty)."""
    out: dict[str, list[str]] = {}
    for m in cfg.get("utility_meters", []) or []:
        name = m.get("name")
        panels = m.get("panels") or []
        if name and panels:
            out[name] = list(panels)
    return out


def _meters_from_db() -> dict[str, list[str]]:
    """{meter_name: [enabled device ids]} from the database, if reachable."""
    try:
        from emslite.database import get_session
        from emslite.models import Device
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    try:
        session = get_session()
    except Exception:
        return {}
    try:
        for d in session.query(Device).filter(Device.enabled.is_(True)).all():
            if d.meter_name:
                out.setdefault(d.meter_name, []).append(d.id)
    except Exception:
        return {}
    finally:
        session.close()
    return out


def _resolve_meters(cfg: dict) -> tuple[dict[str, list[str]], str]:
    """Return ({meter: panels}, source-label), preferring config over DB."""
    cfg_meters = _meters_from_config(cfg)
    if cfg_meters:
        return cfg_meters, "config utility_meters"
    db_meters = _meters_from_db()
    if db_meters:
        return db_meters, "database Device.meter_name"
    return {}, "none"


def _print_meters(meters: dict[str, list[str]], source: str) -> None:
    print("=" * 64)
    print(f"Meter → panel mapping (source: {source})")
    print("-" * 64)
    if not meters:
        print("  No meter mapping found. Populate utility_meters[].panels in the")
        print("  config, or assign meter_name on devices, or use --panels.")
        print("=" * 64)
        return
    for name, panels in meters.items():
        print(f"  {name}  ({len(panels)} panels): {panels}")
    # Cross-meter overlap detection (a panel on two meters double-counts).
    seen: dict[str, list[str]] = {}
    for name, panels in meters.items():
        for p in panels:
            seen.setdefault(p, []).append(name)
    dupes = {p: ms for p, ms in seen.items() if len(ms) > 1}
    if dupes:
        print("-" * 64)
        print("  WARNING: panels assigned to MORE THAN ONE meter (double-counted):")
        for p, ms in dupes.items():
            print(f"           {p}  →  {ms}")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-meters", action="store_true", help="List meters and their panels, then exit")
    ap.add_argument("--meter", help="Reconcile only the panels on this meter (by name)")
    ap.add_argument("--panels", help="Explicit comma-separated panel/column IDs (overrides --meter)")
    ap.add_argument("--start", help="Start date/timestamp (ISO, inclusive)")
    ap.add_argument("--end", help="End date/timestamp (ISO, inclusive)")
    ap.add_argument("--pf", type=float, default=None, help="Power factor (default: from config)")
    ap.add_argument("--voltage", type=float, default=None, help="Line voltage (default: from config)")
    ap.add_argument("--calibration", type=float, default=None, help="Calibration factor (default: from config)")
    ap.add_argument("--price", type=float, default=None, help="Price per kWh (default: from config)")
    ap.add_argument("--bill-kwh", type=float, default=None, help="Utility bill energy (kWh) for this meter")
    ap.add_argument("--bill-amount", type=float, default=None, help="Utility bill cost for this meter")
    ap.add_argument("--config", default="visualization_config.json", help="Path to config file")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / args.config
    cfg = load_config(cfg_path if cfg_path.exists() else None)

    meters, source = _resolve_meters(cfg)

    if args.list_meters:
        _print_meters(meters, source)
        return 0

    voltage = args.voltage if args.voltage is not None else float(cfg.get("line_voltage", 480.0))
    pf = args.pf if args.pf is not None else float(cfg.get("power_factor", 1.0))
    calibration = args.calibration if args.calibration is not None else float(cfg.get("calibration_factor", 1.0))
    price = args.price if args.price is not None else float(cfg.get("price_per_kwh", 0.25))

    master = _master_path(cfg, root)
    if not master.exists():
        print(f"ERROR: master CSV not found at {master}", file=sys.stderr)
        return 2

    df = load_csv(master)
    if args.start:
        df = df[df["Timestamp"] >= pd.to_datetime(args.start, utc=True)]
    if args.end:
        end_ts = args.end + "T23:59:59" if len(args.end) == 10 else args.end
        df = df[df["Timestamp"] <= pd.to_datetime(end_ts, utc=True)]
    if df.empty:
        print("ERROR: no rows in the selected date range", file=sys.stderr)
        return 2

    # ── Resolve which panels to sum ──
    available = meter_columns(df.columns, exclude=excluded_columns(cfg))
    scope_label = "ALL panels"
    if args.panels:
        requested = [p.strip() for p in args.panels.split(",") if p.strip()]
        scope_label = "explicit --panels"
    elif args.meter:
        if args.meter not in meters:
            print(f"ERROR: meter '{args.meter}' not found. Known meters: {list(meters)}", file=sys.stderr)
            print(f"       (mapping source: {source}; use --list-meters)", file=sys.stderr)
            return 2
        requested = meters[args.meter]
        scope_label = f"meter '{args.meter}' ({source})"
    else:
        requested = available[:]
        if len(meters) > 1:
            print("NOTE: you have multiple meters but did not pass --meter; reconciling")
            print(f"      ALL panels together. Bills are per-meter — run per --meter.\n")

    panels = [p for p in requested if p in df.columns]
    missing = [p for p in requested if p not in df.columns]

    dt_hours = df["Timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0

    # Per-column kWh share, to spot an oversized aggregate/main column.
    shares: list[tuple[str, float]] = []
    grand = 0.0
    for col in panels:
        kw = amps_to_kw(df[col].fillna(0), voltage, pf, calibration)
        kwh = integrate_kwh(kw, dt_hours)
        shares.append((col, kwh))
        grand += kwh
    shares.sort(key=lambda x: x[1], reverse=True)
    suspected = [c for c, _ in shares if any(h in c.lower() for h in AGGREGATE_HINTS)]

    print("=" * 64)
    print(f"Master CSV : {master}")
    print(f"Range      : {df['Timestamp'].min()}  →  {df['Timestamp'].max()}  ({len(df)} rows)")
    print(f"Settings   : V={voltage}  PF={pf}  calibration={calibration}  price={price}")
    print(f"Scope      : {scope_label} — {len(panels)} panels")
    if missing:
        print(f"  (skipped {len(missing)} configured panel(s) not present in CSV: {missing})")
    print("-" * 64)
    print("Top columns by kWh share (within scope):")
    for col, kwh in shares[:12]:
        pct = (kwh / grand * 100) if grand else 0.0
        flag = "  <-- looks like an AGGREGATE/MAIN feed" if any(h in col.lower() for h in AGGREGATE_HINTS) else ""
        print(f"  {col:<28} {kwh:14,.1f} kWh  {pct:5.1f}%{flag}")
    if suspected:
        print("\n  WARNING: suspected aggregate column(s) in scope:")
        print(f"           {suspected}")
        print("           If these already include branch panels, add them to")
        print("           'aggregate_columns' in the config to stop double-counting.")
    print("-" * 64)

    computed_kwh = grand
    computed_cost = computed_kwh * price
    print(f"COMPUTED   : {computed_kwh:,.1f} kWh   (${computed_cost:,.2f})")

    if args.bill_kwh:
        ratio = computed_kwh / args.bill_kwh if args.bill_kwh else float("nan")
        print(f"BILL (kWh) : {args.bill_kwh:,.1f} kWh")
        print(f"RATIO      : {ratio:.4f}  ({(ratio - 1) * 100:+.1f}% vs bill)")
        if ratio:
            print(f"Suggested calibration_factor to match this meter: {calibration / ratio:.4f}")
    if args.bill_amount:
        ratio = computed_cost / args.bill_amount if args.bill_amount else float("nan")
        print(f"BILL ($)   : ${args.bill_amount:,.2f}")
        print(f"RATIO      : {ratio:.4f}  ({(ratio - 1) * 100:+.1f}% vs bill)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
