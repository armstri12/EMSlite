"""Device CRUD API endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from .routes_auth import require_admin
from pydantic import BaseModel

from ..database import get_session
from ..models import Device

router = APIRouter(tags=["devices"], dependencies=[Depends(require_admin)])


class DeviceUpdate(BaseModel):
    display_name: str | None = None
    department_id: str | None = None
    meter_name: str | None = None
    location: str | None = None
    device_type: str | None = None
    rated_capacity: float | None = None
    voltage: float | None = None
    phase: str | None = None
    install_date: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    warning_kw: float | None = None
    critical_kw: float | None = None
    enabled: bool | None = None


class BulkAssign(BaseModel):
    device_ids: list[str]
    department_id: str | None = None
    meter_name: str | None = None
    location: str | None = None
    tags: list[str] | None = None


@router.get("/devices")
def list_devices(
    department: str | None = None,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    """List all devices with optional filtering."""
    session = get_session()
    try:
        query = session.query(Device)
        if department:
            query = query.filter(Device.department_id == department)
        if enabled_only:
            query = query.filter(Device.enabled.is_(True))
        devices = query.order_by(Device.display_name).all()
        return [d.to_dict() for d in devices]
    finally:
        session.close()


@router.get("/devices/{device_id}")
def get_device(device_id: str) -> dict[str, Any]:
    """Get a single device by ID."""
    session = get_session()
    try:
        device = session.get(Device, device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        return device.to_dict()
    finally:
        session.close()


@router.put("/devices/{device_id}")
def update_device(device_id: str, body: DeviceUpdate) -> dict[str, Any]:
    """Update device metadata."""
    session = get_session()
    try:
        device = session.get(Device, device_id)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Fields that can be explicitly set to None (reset to default)
        nullable_fields = {
            "voltage", "rated_capacity", "warning_kw", "critical_kw",
            "department_id", "meter_name", "location", "device_type",
            "phase", "install_date", "notes",
        }

        # Use exclude_unset so explicitly-sent null values are preserved
        update_data = body.model_dump(exclude_unset=True)
        if "tags" in update_data:
            update_data["tags"] = json.dumps(update_data["tags"])

        for key, value in update_data.items():
            if value is None and key not in nullable_fields:
                continue
            setattr(device, key, value)

        session.commit()
        session.refresh(device)
        return device.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/devices/bulk-assign")
def bulk_assign(body: BulkAssign) -> dict[str, Any]:
    """Bulk-assign department, location, or tags to multiple devices."""
    session = get_session()
    try:
        updated = 0
        for device_id in body.device_ids:
            device = session.get(Device, device_id)
            if not device:
                continue
            if body.department_id is not None:
                device.department_id = body.department_id if body.department_id else None
            if body.meter_name is not None:
                device.meter_name = body.meter_name if body.meter_name else None
            if body.location is not None:
                device.location = body.location if body.location else None
            if body.tags is not None:
                existing = json.loads(device.tags) if device.tags else []
                merged = list(set(existing + body.tags))
                device.tags = json.dumps(merged)
            updated += 1

        session.commit()
        return {"updated": updated}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/devices-export")
def export_devices() -> list[dict[str, Any]]:
    """Export all device config as JSON."""
    session = get_session()
    try:
        devices = session.query(Device).all()
        return [d.to_dict() for d in devices]
    finally:
        session.close()
