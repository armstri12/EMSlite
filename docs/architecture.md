# EMSlite Architecture

## System overview

```
CSV drops (Meter*_SystemCurrent.csv)
        │
        ▼
  [ ingest.py ]  ←─ file watcher (watchdog)
        │  append-only merge
        ▼
  data/RawPanelUsageHistory.csv   (wide-format master, EST/EDT offset strings)
        │
        ▼
  [ core.py: load_csv() ]  ←─ normalizes timestamps to UTC
        │
        ▼
  [ metrics.py / behavior.py / trending.py ]  ←─ KPI, behavior, trend computation
        │
        ▼
  [ FastAPI routes (emslite/api/) ]  ←─ /api/* endpoints
        │  JSON responses
        ▼
  [ static/js/dashboard.js ]  ←─ Plotly charts, tab renderers
        │
        ▼
  Browser (http://localhost:8000)
```

SQLite (`data/emslite.db`) is used alongside the master CSV for:
- Device and department metadata
- Alert rules and events
- Weather cache (NOAA NCEI)
- Ingestion log

## Data flow in detail

### 1. Ingestion

`emslite/ingest.py` watches the `drops/` folder via `watchdog`. When a new `Meter*_SystemCurrent.csv` appears:

1. Reads the two-column panel file (`Timestamp`, `<MeterName>`)
2. Normalizes timestamps to `America/New_York` offset format
3. Merges the new rows into `data/RawPanelUsageHistory.csv` (append-only — no overwrites)
4. Logs the result to `ingest_log` in SQLite

The standalone `merge_meter_data.py` script performs the same merge operation in batch (designed for initial historical loads).

### 2. CSV to analytics

`core.py:load_csv()` is the single entry point for reading the master CSV:

- Parses `Timestamp` with `pd.to_datetime(..., utc=True)` — all offset strings (e.g., `2026-02-20 14:30:00-0500`) are converted to UTC-aware `datetime64[ns, UTC]`
- Coerces all meter columns to numeric
- Drops rows with unparseable timestamps and sorts by time

This means **all analytics operate on UTC timestamps**. The ingested file stores `America/New_York` offset strings; `load_csv()` bridges the representation.

### 3. Computation

Three computation modules operate on the UTC-normalized DataFrame:

| Module | Function | Output |
|--------|---------|--------|
| `metrics.py` | `compute_kpi()`, `compute_panel_rankings()`, `compute_department_breakdown()` | Total kWh, cost, peak, avg kW, rankings |
| `behavior.py` | `analyze_behavior()`, `compute_phantom_rankings()` | Shift/off-shift split, phantom draw, narratives |
| `trending.py` | `compute_trending_snapshot()`, `compute_trending_detail()` | Period-over-period comparison, daily sparklines |

### 4. API layer

`emslite/api/app.py` creates the FastAPI app with:
- All route modules registered under `/api` prefix
- Static file serving for the frontend SPA
- Lifespan handler for startup (DB table creation, file watcher start)

All endpoints use synchronous `def` (not `async def`) with `get_session()` from `database.py`.

### 5. Frontend

`static/js/dashboard.js` (~1600 lines) is a single-file vanilla JS application:
- Global state: `D` (raw data), `tabState` (per-tab filters), `ALL_PANELS`, `DEPARTMENTS`
- Tab renderers: `renderExecutive()`, `renderOverview()`, `renderAnalytics()`, `renderComparison()`, `renderDevices()`, etc.
- All charts use `Plotly.newPlot()` with shared helpers `pLayout()`, `xA()`, `yA()`, `pCfg`, `weekendShapes()`
- Theme colors from `T()` helper (returns `lightC` or `darkC`)

## Configuration

`visualization_config.json` is the primary configuration file. Loaded via `config.py:load_config()` which supports JS-style comments and trailing commas. Key fields:

| Field | Purpose |
|-------|---------|
| `line_voltage` | System voltage for amps→kW conversion (default: 480V three-phase) |
| `power_factor` | Power factor (default: 1.0) |
| `price_per_kwh` | Electricity cost rate |
| `carbon_kg_per_kwh` | Carbon intensity factor |
| `utility_meters` | Array of `{name, panels[]}` for meter grouping |
| `combo_columns` | Named groups of panels for aggregation (excluded from total_kw to avoid double-counting) |
| `weather.enabled` | Enable NOAA weather overlay |
| `weather.station_id` | NOAA NCEI station identifier |

## Key architectural invariants

1. **UTC at-rest**: All analytics use UTC timestamps. Local time display is frontend-only.
2. **Combo columns excluded from total_kw**: The `total_kw` field in `/api/data` sums only physical panel columns, not pre-aggregated group columns. Group columns are returned separately in `group_series`.
3. **Append-only master CSV**: The master CSV is never modified in place. New rows are appended; existing rows are never changed.
4. **SQLite for metadata only**: Energy time-series data lives in the CSV. SQLite stores device/department/alert metadata, weather cache, and ingestion logs only.
