"""Alert and notification API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from .routes_auth import require_admin
from pydantic import BaseModel

from ..core import amps_to_kw, excluded_columns, load_csv, meter_columns
from ..database import get_session
from ..models import AlertEvent, Device
from .routes_data import _get_config, _get_master_path

router = APIRouter(tags=["alerts"], dependencies=[Depends(require_admin)])


class AlertAckBody(BaseModel):
    keys: list[str]


def _alert_key(device_id: str, severity: str, ts: pd.Timestamp) -> str:
    return f"{device_id}:{severity}:{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"


@router.get("/alerts")
def get_alerts(
    include_acknowledged: bool = Query(False),
    window_hours: int = Query(24, ge=1, le=168),
    latest_only: bool = Query(False),
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    """Return computed threshold alerts over a recent time window."""
    master = _get_master_path()
    if not master.exists():
        return {
            "alerts": [],
            "latest_timestamp": None,
            "summary": {
                "critical": 0,
                "warning": 0,
                "configured_devices": 0,
                "window_hours": window_hours,
            },
        }

    cfg = _get_config()
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    calibration_factor = float(cfg.get("calibration_factor", 1.0))

    df = load_csv(master)
    if df.empty:
        return {
            "alerts": [],
            "latest_timestamp": None,
            "summary": {
                "critical": 0,
                "warning": 0,
                "configured_devices": 0,
                "window_hours": window_hours,
            },
        }

    latest_ts = df["Timestamp"].max()
    window_start = latest_ts - pd.Timedelta(hours=window_hours)
    df_window = df[df["Timestamp"] >= window_start].copy()
    if latest_only:
        df_window = df_window[df_window["Timestamp"] == latest_ts]

    panels = meter_columns(df_window.columns, exclude=excluded_columns(cfg))

    session = get_session()
    try:
        devices = session.query(Device).filter(Device.enabled.is_(True)).all()
        by_id = {d.id: d for d in devices}
        events = {
            e.key: e
            for e in session.query(AlertEvent)
            .filter(AlertEvent.event_ts >= window_start.to_pydatetime())
            .all()
        }

        alerts: list[dict[str, Any]] = []
        configured_devices = 0

        for panel in panels:
            device = by_id.get(panel)
            if not device:
                continue
            if device.warning_kw is None and device.critical_kw is None:
                continue
            configured_devices += 1

            panel_voltage = device.voltage if device.voltage is not None else line_voltage
            kw_series = amps_to_kw(df_window[panel].fillna(0), panel_voltage, power_factor, calibration_factor).fillna(0)
            for idx, kw_val in kw_series.items():
                ts = df_window.loc[idx, "Timestamp"]
                amps_val = float(df_window.loc[idx, panel] or 0)

                severity = None
                threshold = None
                if device.critical_kw is not None and kw_val >= device.critical_kw:
                    severity = "critical"
                    threshold = device.critical_kw
                elif device.warning_kw is not None and kw_val >= device.warning_kw:
                    severity = "warning"
                    threshold = device.warning_kw

                if not severity:
                    continue

                key = _alert_key(panel, severity, ts)
                existing = events.get(key)
                acknowledged = bool(existing.acknowledged) if existing else False
                if (not include_acknowledged) and acknowledged:
                    continue

                alerts.append(
                    {
                        "key": key,
                        "device_id": panel,
                        "device_name": device.display_name,
                        "severity": severity,
                        "current_kw": round(float(kw_val), 3),
                        "threshold_kw": threshold,
                        "current_amps": round(amps_val, 3),
                        "department_id": device.department_id,
                        "acknowledged": acknowledged,
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )

        alerts.sort(
            key=lambda a: (
                1 if a["severity"] == "critical" else 0,
                1 if not a["acknowledged"] else 0,
                a["timestamp"],
            ),
            reverse=True,
        )

        total_count = len(alerts)
        critical_count = sum(1 for a in alerts if a["severity"] == "critical")
        warning_count = sum(1 for a in alerts if a["severity"] == "warning")
        truncated = total_count > limit
        alerts = alerts[:limit]
        return {
            "alerts": alerts,
            "latest_timestamp": latest_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": {
                "critical": critical_count,
                "warning": warning_count,
                "configured_devices": configured_devices,
                "window_hours": window_hours,
                "total_count": total_count,
                "truncated": truncated,
            },
        }
    finally:
        session.close()


@router.post("/alerts/ack")
def acknowledge_alerts(body: AlertAckBody) -> dict[str, Any]:
    """Acknowledge one or more alert keys."""
    session = get_session()
    try:
        updated = 0
        now = datetime.utcnow()
        for key in body.keys:
            parts = key.split(":", 2)
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail=f"Invalid alert key: {key}")
            device_id, severity, ts_raw = parts
            event_ts = datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%SZ")

            record = session.get(AlertEvent, key)
            if not record:
                record = AlertEvent(
                    key=key,
                    device_id=device_id,
                    severity=severity,
                    event_ts=event_ts,
                    acknowledged=True,
                    acknowledged_at=now,
                )
                session.add(record)
                updated += 1
            elif not record.acknowledged:
                record.acknowledged = True
                record.acknowledged_at = now
                updated += 1

        session.commit()
        return {"acknowledged": updated}
    finally:
        session.close()
