"""Data health API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..core import load_csv, meter_columns
from ..database import get_session
from ..models import IngestLog
from .routes_data import _get_master_path, _get_config

router = APIRouter(tags=["data-health"])


@router.get("/data-health")
def get_data_health() -> dict[str, Any]:
    """Return ingestion quality and freshness metrics."""
    session = get_session()
    try:
        logs = (
            session.query(IngestLog)
            .order_by(IngestLog.processed_at.desc())
            .limit(200)
            .all()
        )
    finally:
        session.close()

    recent = [l.to_dict() for l in logs]
    failed = [l for l in recent if l["status"] != "success"]

    master = _get_master_path()
    if not master.exists():
        return {
            "summary": {
                "total_ingest_events": len(recent),
                "failed_events": len(failed),
                "success_rate_pct": 0,
                "latest_timestamp": None,
                "avg_interval_minutes": None,
                "is_data_stale": True,
            },
            "device_freshness": [],
            "recent_failures": failed[:20],
            "recent_ingest": recent[:50],
        }

    import pandas as pd

    cfg = _get_config()
    df = load_csv(master)
    panels = meter_columns(df.columns, exclude=set(cfg.get("combo_columns", {}).keys()))

    latest_ts = df["Timestamp"].max() if not df.empty else None
    avg_interval = None
    is_stale = True
    if len(df.index) >= 2:
        diffs = df["Timestamp"].diff().dropna().dt.total_seconds() / 60.0
        if len(diffs):
            avg_interval = float(diffs.median())
            if latest_ts is not None:
                age_min = (pd.Timestamp.utcnow(tz="UTC") - latest_ts).total_seconds() / 60.0
                is_stale = age_min > (avg_interval * 3)

    device_freshness = []
    for panel in panels:
        col = df[panel]
        idx = col.last_valid_index()
        ts = df.loc[idx, "Timestamp"] if idx is not None else None
        last_val = col.loc[idx] if idx is not None else None
        device_freshness.append(
            {
                "device_id": panel,
                "last_seen": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts is not None else None,
                "latest_value": None if last_val is None else float(last_val),
            }
        )

    success_rate = (100.0 * (len(recent) - len(failed)) / len(recent)) if recent else 100.0

    return {
        "summary": {
            "total_ingest_events": len(recent),
            "failed_events": len(failed),
            "success_rate_pct": round(success_rate, 1),
            "latest_timestamp": latest_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if latest_ts is not None else None,
            "avg_interval_minutes": round(avg_interval, 2) if avg_interval is not None else None,
            "is_data_stale": is_stale,
        },
        "device_freshness": device_freshness,
        "recent_failures": failed[:20],
        "recent_ingest": recent[:50],
    }
