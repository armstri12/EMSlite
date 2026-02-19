# EMSlite — Claude Code Guide

## What is this?

EMSlite is an Energy Management System dashboard for monitoring electrical energy consumption. It ingests CSV data from current-sensing panels, stores it in SQLite, and serves an interactive web dashboard with analytics, period comparisons, alerts, and device management.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy 2.0 (SQLite), Pandas
- **Frontend**: Vanilla JavaScript (no framework), Plotly.js 2.24.1 for charts, custom CSS
- **Data flow**: CSV files dropped into `drops/` folder → `ingest.py` processes and appends to master CSV + SQLite → FastAPI serves via `/api/*` → `dashboard.js` renders with Plotly

## How to Run

```bash
pip install -r requirements.txt
uvicorn emslite.api.app:app --reload
# Dashboard at http://localhost:8000
```

Drop CSV files matching the glob pattern (default: `Meter*_SystemCurrent.csv`) into the `drops/` folder. A file watcher auto-ingests them.

## Project Structure

```
EMSlite/
  visualization_config.json   # Main config (voltage, pricing, meters, groups)
  requirements.txt
  emslite/
    api/
      app.py                  # FastAPI factory, lifespan, router registration
      routes_data.py          # GET /api/data, /api/metrics — main data endpoints
      routes_config.py        # GET/PUT /api/config — system configuration
      routes_devices.py       # CRUD for devices (panels/meters)
      routes_departments.py   # CRUD for departments
      routes_floorplans.py    # Floor plan management with pins/zones
      routes_health.py        # Data health endpoint
      routes_alerts.py        # Alert rules and events
      routes_weather.py       # Weather data overlay (NOAA NCEI)
    config.py                 # Config loading with JSONC support, DEFAULT_CONFIG
    core.py                   # Shared helpers: CSV loading, amps→kW conversion
    database.py               # SQLite engine + session factory
    ingest.py                 # CSV ingestion pipeline + file watcher
    metrics.py                # KPI computation (total kWh, peak, cost, carbon)
    models.py                 # SQLAlchemy ORM models (Device, Department, AlertRule, etc.)
    weather.py                # NOAA NCEI weather data fetching and caching
    static/
      index.html              # SPA entry point with tab-based layout
      js/
        api.js                # Fetch-based API client (all /api/* calls)
        dashboard.js          # Main rendering logic (~1600 lines), all tab renderers
      css/
        dashboard.css         # Full responsive styling, dark/light themes
```

## Key Architecture Patterns

- **Config**: `visualization_config.json` is loaded with `config.py:load_config()` which supports JS-style comments (`//`, `/* */`) and trailing commas. `DEFAULT_CONFIG` in `config.py` provides fallback values. Config is merged via `merge_config()` (deep merge).
- **Models**: All SQLAlchemy models inherit from `Base` in `models.py` and include a `to_dict()` method for serialization. Tables are auto-created by `Base.metadata.create_all()` in `database.py`.
- **API routes**: Each route file defines an `APIRouter` with a tag. Routes are registered in `app.py:create_app()` with `prefix="/api"`. Endpoints use synchronous `def` (not `async def`).
- **Frontend state**: Global variables in `dashboard.js` — `D` (raw data), `tabState` (per-tab filters), `ALL_PANELS`, `DEPARTMENTS`, etc. No frontend framework.
- **Charts**: All charts use `Plotly.newPlot()` with helper functions `pLayout(ov)`, `xA(ov)`, `yA(ov)`, `pCfg`, `weekendShapes(ts)`. Theme colors from `T()` (returns `lightC` or `darkC`).
- **Dashboard tabs**: Executive, Overview/Operations, Analytics, Comparison, Data Table, Devices, Alerts Center, Data Health. Each has a `render*()` function in `dashboard.js`.

## Configuration Reference

Key fields in `visualization_config.json`:
- `line_voltage`, `power_factor`: Electrical conversion constants
- `price_per_kwh`, `carbon_kg_per_kwh`: Cost and emissions factors
- `utility_meters`: Array of `{name, panels[]}` for meter grouping
- `combo_columns`: Named groups of panels for aggregation
- `weather`: `{enabled, station_id, unit}` for NOAA weather overlay

## Conventions

- No TypeScript, no JSX, no build step — vanilla JS loaded directly
- Backend endpoints are synchronous; use `get_session()` from `database.py` with try/finally
- CSS uses custom properties for theming (`--ink`, `--card`, `--bg`, etc.)
- Plotly charts get theme-aware colors from `T()` helper
- File watcher (watchdog) auto-ingests CSVs from `drops/` folder
