"""Alert and notification API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core import amps_to_kw, load_csv, meter_columns
from ..database import get_session
from ..models import AlertEvent, Device
from .routes_data import _get_master_path, _get_config

router = APIRouter(tags=["alerts"])


class AlertAckBody(BaseModel):
    keys: list[str]


@router.get("/alerts")
def get_alerts(include_acknowledged: bool = Query(False)) -> dict[str, Any]:
    """Return active threshold alerts for the latest timestamp."""
    master = _get_master_path()
    if not master.exists():
        return {"alerts": [], "latest_timestamp": None}

    cfg = _get_config()
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))

    df = load_csv(master)
    if df.empty:
        return {"alerts": [], "latest_timestamp": None}

    panels = meter_columns(df.columns, exclude=set(cfg.get("combo_columns", {}).keys()))
    latest_row = df.iloc[-1]
    latest_ts = latest_row["Timestamp"]

    session = get_session()
    try:
        devices = session.query(Device).filter(Device.enabled.is_(True)).all()
        by_id = {d.id: d for d in devices}
        alerts = []
        for panel in panels:
            if panel not in by_id:
                continue
            device = by_id[panel]
            amps_val = float(latest_row.get(panel, 0) or 0)
            kw_val = float(amps_to_kw(df[panel].tail(1).fillna(0), line_voltage, power_factor).iloc[0])

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

            key = f"{panel}:{severity}:{latest_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            existing = session.get(AlertEvent, key)
            acknowledged = bool(existing.acknowledged) if existing else False
            if (not include_acknowledged) and acknowledged:
                continue

            alerts.append(
                {
                    "key": key,
                    "device_id": panel,
                    "device_name": device.display_name,
                    "severity": severity,
                    "current_kw": round(kw_val, 3),
                    "threshold_kw": threshold,
                    "current_amps": round(amps_val, 3),
                    "department_id": device.department_id,
                    "acknowledged": acknowledged,
                    "timestamp": latest_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        alerts.sort(key=lambda a: (0 if a["severity"] == "critical" else 1, -a["current_kw"]))
        return {"alerts": alerts, "latest_timestamp": latest_ts.strftime("%Y-%m-%dT%H:%M:%SZ")}
    finally:
        session.close()


@router.post("/alerts/ack")
def acknowledge_alerts(body: AlertAckBody) -> dict[str, Any]:
    """Acknowledge one or more alert keys."""
    session = get_session()
    try:
        updated = 0
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
                    acknowledged_at=datetime.utcnow(),
                )
                session.add(record)
            elif not record.acknowledged:
                record.acknowledged = True
                record.acknowledged_at = datetime.utcnow()
            updated += 1

        session.commit()
        return {"acknowledged": updated}
    finally:
        session.close()
