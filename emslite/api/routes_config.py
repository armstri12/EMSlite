"""System configuration API endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import load_config, save_config
from ..models import IngestLog

router = APIRouter(tags=["config"])


class ConfigUpdate(BaseModel):
    line_voltage: float | None = None
    power_factor: float | None = None
    price_per_kwh: float | None = None
    rolling_window: str | None = None
    glob_pattern: str | None = None


def _config_path() -> Path:
    from .app import get_project_root
    return get_project_root() / "visualization_config.json"


@router.get("/config")
def get_config() -> dict[str, Any]:
    """Return current system configuration."""
    from .app import get_app_config
    cfg = get_app_config()
    # Return only the user-relevant keys
    return {
        "line_voltage": cfg.get("line_voltage", 480.0),
        "power_factor": cfg.get("power_factor", 1.0),
        "price_per_kwh": cfg.get("price_per_kwh", 0.25),
        "rolling_window": cfg.get("rolling_window", "1h"),
        "glob_pattern": cfg.get("glob_pattern", "Meter*_SystemCurrent.csv"),
        "drops_dir": cfg.get("drops_dir", "drops"),
        "data_dir": cfg.get("data_dir", "data"),
    }


@router.put("/config")
def update_config(body: ConfigUpdate) -> dict[str, Any]:
    """Update system configuration."""
    from .app import _app_config

    path = _config_path()
    cfg = load_config(path if path.exists() else None)

    update_data = body.model_dump(exclude_none=True)
    for key, value in update_data.items():
        cfg[key] = value
        _app_config[key] = value

    save_config(cfg, path)
    return get_config()


@router.get("/ingest-log")
def get_ingest_log(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent ingestion log entries."""
    from ..database import get_session

    session = get_session()
    try:
        entries = (
            session.query(IngestLog)
            .order_by(IngestLog.processed_at.desc())
            .limit(limit)
            .all()
        )
        return [e.to_dict() for e in entries]
    finally:
        session.close()
