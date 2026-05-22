"""FastAPI application factory for EMSlite."""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from ..config import load_config
from ..database import init_db
from ..ingest import ingest_existing, start_watcher, sync_devices_from_master
from . import app_state
from .routes_auth import router as auth_router
from .routes_config import router as config_router
from .routes_data import router as data_router
from .routes_departments import router as departments_router
from .routes_devices import router as devices_router
from .routes_floorplans import router as floorplans_router
from .routes_health import router as health_router
from .routes_alerts import router as alerts_router
from .routes_weather import router as weather_router
from .routes_hvac import router as hvac_router
from .routes_behavior import router as behavior_router
from .routes_trending import router as trending_router
from .routes_bills import router as bills_router
from .routes_production import router as production_router
from .routes_production import seed_default_metric_definitions
from .routes_reports import router as reports_router
from .routes_wireless import router as wireless_router
from .routes_demand_response import router as demand_response_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_observer = None


def get_app_config() -> dict:
    return app_state.get_app_config()


def get_project_root() -> Path:
    return app_state.get_project_root()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _observer

    root = get_project_root()
    config_path = root / "visualization_config.json"
    app_state._app_config = load_config(config_path if config_path.exists() else None)

    # Initialize database first so wireless listener can write on first connection
    db_path = root / "emslite.db"
    init_db(db_path)

    # Start wireless TCP listener if enabled (after DB is ready)
    _wireless_enabled = False
    wireless_cfg = app_state._app_config.get("wireless", {})
    if wireless_cfg.get("enabled", False):
        from ..wireless import start_listener
        _wireless_enabled = True
        await start_listener(
            int(wireless_cfg.get("tcp_port", 4950)),
            bool(wireless_cfg.get("auto_discover", True)),
        )

    # Seed default production metric definitions on first startup (idempotent).
    seed_default_metric_definitions(app_state._app_config)

    # Ensure directories exist
    drops_dir = root / app_state._app_config.get("drops_dir", "drops")
    data_dir = root / app_state._app_config.get("data_dir", "data")
    drops_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    master_path = data_dir / app_state._app_config.get("master_filename", "RawPanelUsageHistory.csv")
    glob_pattern = app_state._app_config.get("glob_pattern", "Meter*_SystemCurrent.csv")

    # Sync devices from existing master CSV (handles direct CSV placement)
    sync_devices_from_master(master_path)

    # Process any files already in the drops folder
    existing_count = ingest_existing(drops_dir, master_path, glob_pattern)
    if existing_count:
        logger.info("Processed %d existing files from drops folder", existing_count)

    # Start file watcher
    _observer = start_watcher(drops_dir, master_path, glob_pattern)
    logger.info("EMSlite started. Dashboard at http://localhost:8000")

    # Warn if admin password is still the insecure default
    admin_pw = os.environ.get("ADMIN_PASSWORD") or app_state._app_config.get("admin_password", "admin")
    if admin_pw == "admin":
        logger.warning(
            "SECURITY WARNING: Admin password is set to 'admin'. "
            "Set the ADMIN_PASSWORD environment variable or update "
            "'admin_password' in visualization_config.json before deploying."
        )

    yield

    # Shutdown wireless listener
    if _wireless_enabled:
        from ..wireless import stop_listener
        await stop_listener()

    # Shutdown file watcher
    if _observer:
        _observer.stop()
        _observer.join(timeout=5)
    logger.info("EMSlite stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EMSlite",
        description="Energy Management System Lite",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

    # API routes
    app.include_router(auth_router, prefix="/api")
    app.include_router(data_router, prefix="/api")
    app.include_router(devices_router, prefix="/api")
    app.include_router(departments_router, prefix="/api")
    app.include_router(config_router, prefix="/api")
    app.include_router(floorplans_router, prefix="/api")
    app.include_router(health_router, prefix="/api")
    app.include_router(alerts_router, prefix="/api")
    app.include_router(weather_router, prefix="/api")
    app.include_router(hvac_router, prefix="/api")
    app.include_router(behavior_router, prefix="/api")
    app.include_router(trending_router, prefix="/api")
    app.include_router(bills_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(production_router, prefix="/api")
    app.include_router(wireless_router, prefix="/api")
    app.include_router(demand_response_router, prefix="/api")

    # Serve static frontend files
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
