"""Floor plan API endpoints — upload images, manage device pins and zones."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..database import get_session
from ..models import FloorPlan, FloorPlanPin, FloorPlanZone

router = APIRouter(prefix="/floorplans", tags=["floorplans"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "static" / "uploads" / "floorplans"


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


# ── List / Get ──────────────────────────────────────────────

@router.get("")
def list_floorplans():
    session = get_session()
    try:
        plans = session.query(FloorPlan).order_by(FloorPlan.created_at.desc()).all()
        return [fp.to_dict() for fp in plans]
    finally:
        session.close()


@router.get("/{plan_id}")
def get_floorplan(plan_id: int):
    session = get_session()
    try:
        fp = session.query(FloorPlan).get(plan_id)
        if not fp:
            raise HTTPException(404, "Floor plan not found")
        data = fp.to_dict()
        data["pins"] = [p.to_dict() for p in fp.pins]
        data["zones"] = [z.to_dict() for z in fp.zones]
        return data
    finally:
        session.close()


# ── Create (upload) ─────────────────────────────────────────

@router.post("")
async def create_floorplan(
    name: str = Form(...),
    show_on_dashboard: bool = Form(False),
    image: UploadFile = File(...),
):
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
    ext = Path(image.filename).suffix.lower() if image.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported image type: {ext}")

    upload_dir = _ensure_upload_dir()
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = upload_dir / filename

    with dest.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    # Store path relative to static root for serving
    relative_path = f"/uploads/floorplans/{filename}"

    session = get_session()
    try:
        fp = FloorPlan(name=name, image_path=relative_path, show_on_dashboard=show_on_dashboard)
        session.add(fp)
        session.commit()
        session.refresh(fp)
        return fp.to_dict()
    except Exception:
        dest.unlink(missing_ok=True)
        session.rollback()
        raise
    finally:
        session.close()


# ── Update ──────────────────────────────────────────────────

class FloorPlanUpdate(BaseModel):
    name: str | None = None
    show_on_dashboard: bool | None = None


@router.put("/{plan_id}")
def update_floorplan(plan_id: int, body: FloorPlanUpdate):
    session = get_session()
    try:
        fp = session.query(FloorPlan).get(plan_id)
        if not fp:
            raise HTTPException(404, "Floor plan not found")
        if body.name is not None:
            fp.name = body.name
        if body.show_on_dashboard is not None:
            fp.show_on_dashboard = body.show_on_dashboard
        session.commit()
        session.refresh(fp)
        return fp.to_dict()
    finally:
        session.close()


# ── Delete ──────────────────────────────────────────────────

@router.delete("/{plan_id}")
def delete_floorplan(plan_id: int):
    session = get_session()
    try:
        fp = session.query(FloorPlan).get(plan_id)
        if not fp:
            raise HTTPException(404, "Floor plan not found")
        # Remove image file
        static_dir = Path(__file__).resolve().parent.parent / "static"
        img_path = static_dir / fp.image_path.lstrip("/")
        img_path.unlink(missing_ok=True)
        session.delete(fp)
        session.commit()
        return {"ok": True}
    finally:
        session.close()


# ── Pin management ──────────────────────────────────────────

class PinCreate(BaseModel):
    device_id: str
    x_pct: float
    y_pct: float
    label: str | None = None


class PinUpdate(BaseModel):
    x_pct: float | None = None
    y_pct: float | None = None
    label: str | None = None


@router.post("/{plan_id}/pins")
def add_pin(plan_id: int, body: PinCreate):
    session = get_session()
    try:
        fp = session.query(FloorPlan).get(plan_id)
        if not fp:
            raise HTTPException(404, "Floor plan not found")
        pin = FloorPlanPin(
            floor_plan_id=plan_id,
            device_id=body.device_id,
            x_pct=body.x_pct,
            y_pct=body.y_pct,
            label=body.label,
        )
        session.add(pin)
        session.commit()
        session.refresh(pin)
        return pin.to_dict()
    finally:
        session.close()


@router.put("/{plan_id}/pins/{pin_id}")
def update_pin(plan_id: int, pin_id: int, body: PinUpdate):
    session = get_session()
    try:
        pin = session.query(FloorPlanPin).filter(
            FloorPlanPin.id == pin_id, FloorPlanPin.floor_plan_id == plan_id
        ).first()
        if not pin:
            raise HTTPException(404, "Pin not found")
        if body.x_pct is not None:
            pin.x_pct = body.x_pct
        if body.y_pct is not None:
            pin.y_pct = body.y_pct
        if body.label is not None:
            pin.label = body.label
        session.commit()
        session.refresh(pin)
        return pin.to_dict()
    finally:
        session.close()


@router.delete("/{plan_id}/pins/{pin_id}")
def delete_pin(plan_id: int, pin_id: int):
    session = get_session()
    try:
        pin = session.query(FloorPlanPin).filter(
            FloorPlanPin.id == pin_id, FloorPlanPin.floor_plan_id == plan_id
        ).first()
        if not pin:
            raise HTTPException(404, "Pin not found")
        session.delete(pin)
        session.commit()
        return {"ok": True}
    finally:
        session.close()


# ── Zone management (polygon areas) ───────────────────────

class ZoneCreate(BaseModel):
    device_id: str
    points: list[dict]  # [{x: float, y: float}, ...]
    label: str | None = None


class ZoneUpdate(BaseModel):
    points: list[dict] | None = None
    label: str | None = None


@router.post("/{plan_id}/zones")
def add_zone(plan_id: int, body: ZoneCreate):
    session = get_session()
    try:
        fp = session.query(FloorPlan).get(plan_id)
        if not fp:
            raise HTTPException(404, "Floor plan not found")
        zone = FloorPlanZone(
            floor_plan_id=plan_id,
            device_id=body.device_id,
            points=json.dumps(body.points),
            label=body.label,
        )
        session.add(zone)
        session.commit()
        session.refresh(zone)
        return zone.to_dict()
    finally:
        session.close()


@router.put("/{plan_id}/zones/{zone_id}")
def update_zone(plan_id: int, zone_id: int, body: ZoneUpdate):
    session = get_session()
    try:
        zone = session.query(FloorPlanZone).filter(
            FloorPlanZone.id == zone_id, FloorPlanZone.floor_plan_id == plan_id
        ).first()
        if not zone:
            raise HTTPException(404, "Zone not found")
        if body.points is not None:
            zone.points = json.dumps(body.points)
        if body.label is not None:
            zone.label = body.label
        session.commit()
        session.refresh(zone)
        return zone.to_dict()
    finally:
        session.close()


@router.delete("/{plan_id}/zones/{zone_id}")
def delete_zone(plan_id: int, zone_id: int):
    session = get_session()
    try:
        zone = session.query(FloorPlanZone).filter(
            FloorPlanZone.id == zone_id, FloorPlanZone.floor_plan_id == plan_id
        ).first()
        if not zone:
            raise HTTPException(404, "Zone not found")
        session.delete(zone)
        session.commit()
        return {"ok": True}
    finally:
        session.close()
