# EMSlite Data Quality

_Last reviewed: 2026-02-20_

This document describes how EMSlite handles missing data, duplicate timestamps, irregular sampling cadence, and inferred values.

---

## Missing panel readings (NaN)

### Current behavior

All amps-to-kW conversions apply `.fillna(0)` before conversion:

```python
kw = amps_to_kw(df[col].fillna(0), line_voltage, power_factor)
```

**Locations:** `metrics.py:32,79,121`, `routes_data.py:80,88`, `behavior.py:307`, `trending.py:67,70,77`

### What NaN means

NaN in a panel column can mean either:
- **True zero current** — the circuit is energized but drawing no load
- **Missing telemetry** — the sensor was offline, the file had a gap, or the column didn't exist in a given panel dump

EMSlite does not currently distinguish between these cases. Both are treated as zero current.

### Impact

When a sensor goes offline and returns NaN readings, those time periods contribute `0 kWh` to all energy totals. This **understates consumption** during outages.

### How to identify missing data

The `/api/health` endpoint reports data completeness metrics including gaps and expected vs actual row counts. Use this to identify periods where sensor data may be missing.

### Future improvement

A NaN-handling strategy flag could differentiate:
- `fillna(0)` — current behavior, treats missing as zero (safe for display, understates energy)
- `dropna()` — exclude the panel from totals when data is missing (accurate per-panel, may produce gaps in total)
- `interpolate()` — linear fill between valid readings (smooth but introduces estimated values)

---

## Duplicate timestamps

### In ingestion

`merge_meter_data.py` and `ingest.py` both call `.drop_duplicates(subset=["Timestamp"], keep="last")` after parsing each file. The "last" policy means later-appearing rows for the same timestamp win.

The master CSV is append-only. If the same timestamp appears in both the master and a new panel file, the master row is kept (new rows are only appended for timestamps not already present).

### In analytics

`core.py:load_csv()` calls `.sort_values("Timestamp")` but does NOT deduplicate. If duplicates exist in the master CSV (should not happen under normal ingestion), they will produce double-counting in energy totals.

---

## Irregular sampling cadence

### Expected cadence

Panel CSV files typically have 15-minute intervals. The system makes no hard assumption about cadence; the integration formula adapts to any interval via `timestamps.diff()`.

### What changes with irregular cadence

**Energy (kWh):** Correctly computed regardless of cadence. Longer gaps produce more kWh from the same kW reading (the panel was running at that level for longer).

**Average kW:** Uses time-weighted mean since 2026-02-20 (`metrics.py`). Correctly handles irregular cadence. See [calculations.md §3](calculations.md#3-average-kw-time-weighted).

**Daily kWh in trending:** `trending.py:_daily_kwh()` uses left-point rectangular integration. With irregular cadence, a long gap straddling midnight will attribute the full interval's kWh to whichever day the end-of-interval sample falls on.

### Detecting irregular cadence

Check the interval distribution in your data:
```python
import pandas as pd
df = pd.read_csv("data/RawPanelUsageHistory.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
df["dt_min"] = df["Timestamp"].diff().dt.total_seconds() / 60
print(df["dt_min"].describe())
```

Significant standard deviation in `dt_min` indicates irregular sampling.

---

## Ingest log

Every ingestion attempt is recorded in the `ingest_log` SQLite table:

| Column | Description |
|--------|------------|
| `filename` | Source CSV filename |
| `status` | `"success"` or `"failed"` |
| `rows_added` | Number of new rows appended to master |
| `device_id` | Panel/meter ID |
| `error_message` | Error details if `status="failed"` |
| `processed_at` | Timestamp of ingestion (naive UTC) |

Query the log via `/api/health` or directly in SQLite (`data/emslite.db`).

---

## Data health endpoint

`GET /api/health` returns:
- Total row count in the master CSV
- Date range covered
- Per-panel expected vs actual row counts
- Gap detection results
- Any panels with >5% missing data

Use this to audit data quality before relying on dashboard metrics.
