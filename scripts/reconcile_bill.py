#!/usr/bin/env python3
"""Reconcile computed energy against a utility bill — over-reporting diagnostic.

Standalone tool (no running server, no app config cache): loads the master CSV
directly, integrates kWh over a date range for the given conversion settings, and
reports the ratio of computed-vs-bill energy. It also audits the columns for a
likely aggregate/main-feed column that would be double-counted into the total.

Examples
--------
    python scripts/reconcile_bill.py --start 2026-05-01 --end 2026-05-31 \
        --pf 0.82 --voltage 470 --bill-kwh 120000

    # compare PF=1.0 vs PF=0.82 to confirm power factor is the lever
    python scripts/reconcile_bill.py --start 2026-05-01 --end 2026-05-31 --pf 1.0
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="Start date/timestamp (ISO, inclusive)")
    ap.add_argument("--end", help="End date/timestamp (ISO, inclusive)")
    ap.add_argument("--pf", type=float, default=None, help="Power factor (default: from config)")
    ap.add_argument("--voltage", type=float, default=None, help="Line voltage (default: from config)")
    ap.add_argument("--calibration", type=float, default=None, help="Calibration factor (default: from config)")
    ap.add_argument("--price", type=float, default=None, help="Price per kWh (default: from config)")
    ap.add_argument("--bill-kwh", type=float, default=None, help="Utility bill energy (kWh) to compare against")
    ap.add_argument("--bill-amount", type=float, default=None, help="Utility bill cost to compare against")
    ap.add_argument("--config", default="visualization_config.json", help="Path to config file")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / args.config
    cfg = load_config(cfg_path if cfg_path.exists() else None)

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
        df = df[df["Timestamp"] <= pd.to_datetime(args.end + "T23:59:59" if len(args.end) == 10 else args.end, utc=True)]
    if df.empty:
        print("ERROR: no rows in the selected date range", file=sys.stderr)
        return 2

    dt_hours = df["Timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0
    panels = meter_columns(df.columns, exclude=excluded_columns(cfg))

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
    print(f"Panels summed into total ({len(panels)}): excluded={sorted(excluded_columns(cfg)) or '∅'}")
    print("-" * 64)
    print("Top columns by kWh share:")
    for col, kwh in shares[:10]:
        pct = (kwh / grand * 100) if grand else 0.0
        flag = "  <-- looks like an AGGREGATE/MAIN feed" if any(h in col.lower() for h in AGGREGATE_HINTS) else ""
        print(f"  {col:<28} {kwh:14,.1f} kWh  {pct:5.1f}%{flag}")
    if suspected:
        print("\n  WARNING: suspected aggregate column(s) are being SUMMED into the total:")
        print(f"           {suspected}")
        print("           If these already include branch panels, add them to")
        print("           'aggregate_columns' in the config to stop double-counting.")
        excl = excluded_columns(cfg)
        adj_total = sum(k for c, k in shares if c not in excl and c not in suspected)
        print(f"           total excluding suspected: {adj_total:,.1f} kWh "
              f"({adj_total / grand * 100:.1f}% of current total)")
    print("-" * 64)

    computed_kwh = grand
    computed_cost = computed_kwh * price
    print(f"COMPUTED   : {computed_kwh:,.1f} kWh   (${computed_cost:,.2f})")

    if args.bill_kwh:
        ratio = computed_kwh / args.bill_kwh if args.bill_kwh else float("nan")
        print(f"BILL (kWh) : {args.bill_kwh:,.1f} kWh")
        print(f"RATIO      : {ratio:.4f}  ({(ratio - 1) * 100:+.1f}% vs bill)")
        print(f"Suggested calibration_factor to match bill: {calibration / ratio:.4f}")
    if args.bill_amount:
        ratio = computed_cost / args.bill_amount if args.bill_amount else float("nan")
        print(f"BILL ($)   : ${args.bill_amount:,.2f}")
        print(f"RATIO      : {ratio:.4f}  ({(ratio - 1) * 100:+.1f}% vs bill)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
