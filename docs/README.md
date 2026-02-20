# EMSlite Documentation Index

## What is EMSlite?

EMSlite is an Energy Management System dashboard for monitoring electrical energy consumption across a facility. It ingests per-panel current readings from CSV files, computes energy KPIs, and serves an interactive web dashboard with analytics, trend detection, behavioral analysis, alerts, and device management.

## Who is this documentation for?

| Audience | Start here |
|---------|-----------|
| New developers / quick start | [operations.md](operations.md) |
| Understanding data flow and architecture | [architecture.md](architecture.md) |
| Energy calculation formulas and assumptions | [calculations.md](calculations.md) |
| Timezone and temporal behavior | [temporal-assumptions.md](temporal-assumptions.md) |
| API endpoint reference | [api/](api/) |
| Database entities and relationships | [data-model.md](data-model.md) |
| Telemetry quality and NaN handling | [data-quality.md](data-quality.md) |
| Dashboard structure and frontend internals | [frontend.md](frontend.md) |
| Contributing and code standards | [contributing.md](contributing.md) |

## Key concepts

- **Panel / Meter**: A current-sensing circuit breaker panel. One CSV column per panel, values in amps.
- **kW**: Panels report amps; kW = amps × √3 × voltage × power_factor / 1000 (three-phase).
- **kWh**: Energy = power × time. Computed as left-point rectangular integration over sample intervals.
- **Master CSV**: `data/RawPanelUsageHistory.csv` — the append-only wide-format master data store.
- **Combo columns**: Pre-aggregated columns in the CSV representing groups of panels (e.g., a whole floor). Excluded from `total_kw` sums to avoid double-counting.
- **UTC at-rest**: All timestamps are normalized to UTC when read into the analytics pipeline. Local display conversions happen in the frontend.

## Project structure

```
EMSlite/
  visualization_config.json   # Main config: voltage, pricing, meters, groups
  requirements.txt
  emslite/
    api/
      app.py                  # FastAPI factory, lifespan, router registration
      routes_data.py          # GET /api/data, /api/metrics
      routes_config.py        # GET/PUT /api/config
      routes_devices.py       # CRUD for devices (panels/meters)
      routes_departments.py   # CRUD for departments
      routes_floorplans.py    # Floor plan management
      routes_health.py        # Data health endpoint
      routes_alerts.py        # Alert rules and events
      routes_weather.py       # Weather data overlay (NOAA NCEI)
    config.py                 # Config loading (JSONC support, DEFAULT_CONFIG)
    core.py                   # Shared helpers: CSV loading, amps→kW, column utilities
    database.py               # SQLite engine + session factory
    ingest.py                 # CSV ingestion pipeline + file watcher
    metrics.py                # KPI computation (total kWh, peak, cost, carbon)
    behavior.py               # Shift vs off-shift energy split, phantom draw analysis
    trending.py               # Per-panel trend snapshot and period comparison
    models.py                 # SQLAlchemy ORM models
    weather.py                # NOAA NCEI weather data fetching and caching
    static/
      index.html              # SPA entry point (tab-based layout)
      js/
        api.js                # Fetch-based API client
        dashboard.js          # Main rendering logic, all tab renderers
      css/
        dashboard.css         # Responsive styling, dark/light themes
  docs/                       # This documentation directory
  drops/                      # Drop zone for incoming CSV files (auto-ingested)
  data/                       # Persistent data store (master CSV + SQLite DB)
```

## Running EMSlite

See [operations.md](operations.md) for full setup and operational runbook.

```bash
pip install -r requirements.txt
uvicorn emslite.api.app:app --reload
# Dashboard at http://localhost:8000
```

Drop CSV files matching `Meter*_SystemCurrent.csv` into the `drops/` folder — the file watcher auto-ingests them.
