"""FastAPI application factory for EMSlite."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..config import load_config
from ..database import init_db
from ..ingest import ingest_existing, start_watcher
from .routes_config import router as config_router
from .routes_data import router as data_router
from .routes_departments import router as departments_router
from .routes_devices import router as devices_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Global config and watcher references
_app_config: dict = {}
_observer = None


def get_app_config() -> dict:
    return _app_config


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_config, _observer

    root = get_project_root()
    config_path = root / "visualization_config.json"
    _app_config = load_config(config_path if config_path.exists() else None)

    # Initialize database
    db_path = root / "emslite.db"
    init_db(db_path)

    # Ensure directories exist
    drops_dir = root / _app_config.get("drops_dir", "drops")
    data_dir = root / _app_config.get("data_dir", "data")
    drops_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    master_path = data_dir / _app_config.get("master_filename", "RawPanelUsageHistory.csv")
    glob_pattern = _app_config.get("glob_pattern", "Meter*_SystemCurrent.csv")

    # Process any files already in the drops folder
    existing_count = ingest_existing(drops_dir, master_path, glob_pattern)
    if existing_count:
        logger.info("Processed %d existing files from drops folder", existing_count)

    # Start file watcher
    _observer = start_watcher(drops_dir, master_path, glob_pattern)
    logger.info("EMSlite started. Dashboard at http://localhost:8000")

    yield

    # Shutdown
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

    # API routes
    app.include_router(data_router, prefix="/api")
    app.include_router(devices_router, prefix="/api")
    app.include_router(departments_router, prefix="/api")
    app.include_router(config_router, prefix="/api")

    # Serve static frontend files
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
