"""Wireless sensor network API endpoints (Monnit Alta)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from .routes_auth import require_admin
from pydantic import BaseModel

from ..database import get_session
from ..models import SensorReading, WirelessGateway, WirelessSensor

router = APIRouter(tags=["wireless"], dependencies=[Depends(require_admin)])


class SensorUpdate(BaseModel):
    display_name: str | None = None
    unit: str | None = None
    enabled: bool | None = None
    notes: str | None = None
    tags: list[str] | None = None


@router.get("/wireless/status")
def get_wireless_status() -> dict[str, Any]:
    from ..wireless import get_status
    return get_status()


@router.get("/wireless/gateways")
def list_gateways() -> list[dict[str, Any]]:
    session = get_session()
    try:
        gws = session.query(WirelessGateway).order_by(WirelessGateway.display_name).all()
        return [gw.to_dict() for gw in gws]
    finally:
        session.close()


@router.get("/wireless/sensors")
def list_sensors(
    gateway_id: str | None = Query(None),
    enabled_only: bool = Query(False),
) -> list[dict[str, Any]]:
    session = get_session()
    try:
        q = session.query(WirelessSensor)
        if gateway_id:
            q = q.filter(WirelessSensor.gateway_id == gateway_id)
        if enabled_only:
            q = q.filter(WirelessSensor.enabled.is_(True))
        sensors = q.order_by(WirelessSensor.display_name).all()

        result = []
        for s in sensors:
            d = s.to_dict()
            latest = (
                session.query(SensorReading)
                .filter(SensorReading.sensor_id == s.id)
                .order_by(SensorReading.timestamp.desc())
                .first()
            )
            d["latest_reading"] = latest.to_dict() if latest else None
            result.append(d)
        return result
    finally:
        session.close()


@router.get("/wireless/sensors/{sensor_id}")
def get_sensor(sensor_id: str) -> dict[str, Any]:
    session = get_session()
    try:
        sensor = session.get(WirelessSensor, sensor_id)
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")
        d = sensor.to_dict()
        latest = (
            session.query(SensorReading)
            .filter(SensorReading.sensor_id == sensor_id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )
        d["latest_reading"] = latest.to_dict() if latest else None
        return d
    finally:
        session.close()


@router.put("/wireless/sensors/{sensor_id}")
def update_sensor(sensor_id: str, body: SensorUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        sensor = session.get(WirelessSensor, sensor_id)
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")

        update_data = body.model_dump(exclude_unset=True)
        if "tags" in update_data:
            update_data["tags"] = json.dumps(update_data["tags"])

        for key, value in update_data.items():
            setattr(sensor, key, value)

        session.commit()
        session.refresh(sensor)
        return sensor.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/wireless/readings")
def get_readings(
    sensor_id: str = Query(..., description="Sensor ID to query"),
    start: str | None = Query(None, description="Start ISO timestamp"),
    end: str | None = Query(None, description="End ISO timestamp"),
    limit: int = Query(1000, ge=1, le=10000),
) -> list[dict[str, Any]]:
    from datetime import datetime

    session = get_session()
    try:
        q = session.query(SensorReading).filter(SensorReading.sensor_id == sensor_id)
        if start:
            q = q.filter(SensorReading.timestamp >= datetime.fromisoformat(start.replace("Z", "+00:00")))
        if end:
            q = q.filter(SensorReading.timestamp <= datetime.fromisoformat(end.replace("Z", "+00:00")))
        readings = q.order_by(SensorReading.timestamp.asc()).limit(limit).all()
        return [r.to_dict() for r in readings]
    finally:
        session.close()
