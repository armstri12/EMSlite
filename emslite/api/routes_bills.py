"""Utility bill CRUD and cost comparison endpoints."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core import amps_to_kw, get_device_voltage_map, load_csv, meter_columns
from ..database import get_session
from ..metrics import integrate_kwh
from ..models import Device, UtilityBill

router = APIRouter(tags=["bills"])


class BillCreate(BaseModel):
    meter_name: str
    period_start: str  # ISO date
    period_end: str  # ISO date
    bill_date: str | None = None
    amount: float
    notes: str | None = None


class BillUpdate(BaseModel):
    meter_name: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    bill_date: str | None = None
    amount: float | None = None
    notes: str | None = None


@router.get("/bills")
def list_bills(meter: str | None = None) -> list[dict[str, Any]]:
    """List all utility bills, optionally filtered by meter name."""
    session = get_session()
    try:
        query = session.query(UtilityBill)
        if meter:
            query = query.filter(UtilityBill.meter_name == meter)
        bills = query.order_by(UtilityBill.period_end.desc()).all()
        return [b.to_dict() for b in bills]
    finally:
        session.close()


@router.post("/bills")
def create_bill(body: BillCreate) -> dict[str, Any]:
    """Create a new utility bill record."""
    session = get_session()
    try:
        bill = UtilityBill(
            meter_name=body.meter_name,
            period_start=date.fromisoformat(body.period_start),
            period_end=date.fromisoformat(body.period_end),
            bill_date=date.fromisoformat(body.bill_date) if body.bill_date else None,
            amount=body.amount,
            notes=body.notes,
        )
        session.add(bill)
        session.commit()
        session.refresh(bill)
        return bill.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/bills/{bill_id}")
def update_bill(bill_id: int, body: BillUpdate) -> dict[str, Any]:
    """Update an existing utility bill."""
    session = get_session()
    try:
        bill = session.get(UtilityBill, bill_id)
        if not bill:
            raise HTTPException(status_code=404, detail="Bill not found")

        if body.meter_name is not None:
            bill.meter_name = body.meter_name
        if body.period_start is not None:
            bill.period_start = date.fromisoformat(body.period_start)
        if body.period_end is not None:
            bill.period_end = date.fromisoformat(body.period_end)
        if body.bill_date is not None:
            bill.bill_date = date.fromisoformat(body.bill_date) if body.bill_date else None
        if body.amount is not None:
            bill.amount = body.amount
        if body.notes is not None:
            bill.notes = body.notes

        session.commit()
        session.refresh(bill)
        return bill.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/bills/{bill_id}")
def delete_bill(bill_id: int) -> dict[str, Any]:
    """Delete a utility bill."""
    session = get_session()
    try:
        bill = session.get(UtilityBill, bill_id)
        if not bill:
            raise HTTPException(status_code=404, detail="Bill not found")
        session.delete(bill)
        session.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/bills/{bill_id}/comparison")
def bill_comparison(bill_id: int) -> dict[str, Any]:
    """Compare a bill's actual amount with calculated cost from metered data."""
    from .app import get_app_config, get_project_root

    session = get_session()
    try:
        bill = session.get(UtilityBill, bill_id)
        if not bill:
            raise HTTPException(status_code=404, detail="Bill not found")

        # Find devices assigned to this meter
        devices = (
            session.query(Device)
            .filter(Device.meter_name == bill.meter_name, Device.enabled.is_(True))
            .all()
        )
        device_ids = [d.id for d in devices]

        cfg = get_app_config()
        root = get_project_root()
        data_dir = root / cfg.get("data_dir", "data")
        master = data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")

        if not master.exists() or not device_ids:
            return {
                "bill_id": bill.id,
                "meter_name": bill.meter_name,
                "bill_amount": bill.amount,
                "calculated_cost": 0.0,
                "difference": bill.amount,
                "total_kwh": 0.0,
                "device_count": len(device_ids),
            }

        line_voltage = float(cfg.get("line_voltage", 480.0))
        power_factor = float(cfg.get("power_factor", 1.0))
        calibration_factor = float(cfg.get("calibration_factor", 1.0))
        price_per_kwh = float(cfg.get("price_per_kwh", 0.25))
        voltage_map = get_device_voltage_map()

        df = load_csv(master)

        # Filter to bill period
        start_dt = pd.to_datetime(str(bill.period_start), utc=True)
        end_dt = pd.to_datetime(str(bill.period_end) + "T23:59:59", utc=True)
        df = df[(df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)]

        if df.empty:
            return {
                "bill_id": bill.id,
                "meter_name": bill.meter_name,
                "bill_amount": bill.amount,
                "calculated_cost": 0.0,
                "difference": bill.amount,
                "total_kwh": 0.0,
                "device_count": len(device_ids),
            }

        # Hours between readings, used for trapezoidal kWh integration.
        hours = df["Timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0

        # Calculate kWh for each device on this meter
        total_kwh = 0.0
        for dev_id in device_ids:
            if dev_id not in df.columns:
                continue
            v = voltage_map.get(dev_id, line_voltage)
            kw_series = amps_to_kw(df[dev_id].fillna(0), v, power_factor, calibration_factor).fillna(0)
            total_kwh += integrate_kwh(kw_series, hours)

        calculated_cost = round(total_kwh * price_per_kwh, 2)

        # Suggested calibration_factor to make computed energy match this bill.
        # Multiplying by the current factor keeps it idempotent (≈ current value
        # once already calibrated). bill kWh-equivalent = amount / price_per_kwh.
        suggested_calibration_factor = None
        if total_kwh > 0 and price_per_kwh > 0:
            bill_kwh_equiv = bill.amount / price_per_kwh
            suggested_calibration_factor = round(
                calibration_factor * (bill_kwh_equiv / total_kwh), 4
            )

        total_kwh = round(total_kwh, 2)

        return {
            "bill_id": bill.id,
            "meter_name": bill.meter_name,
            "bill_amount": bill.amount,
            "calculated_cost": calculated_cost,
            "difference": round(bill.amount - calculated_cost, 2),
            "total_kwh": total_kwh,
            "device_count": len(device_ids),
            "calibration_factor": calibration_factor,
            "suggested_calibration_factor": suggested_calibration_factor,
        }
    finally:
        session.close()
