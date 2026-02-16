"""Department CRUD API endpoints."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..database import get_session
from ..models import Department, Device

router = APIRouter(tags=["departments"])


class DepartmentCreate(BaseModel):
    display_name: str
    color: str = "#8BD435"
    description: str | None = None


class DepartmentUpdate(BaseModel):
    display_name: str | None = None
    color: str | None = None
    description: str | None = None


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


@router.get("/departments")
def list_departments() -> list[dict[str, Any]]:
    """List all departments with device counts."""
    session = get_session()
    try:
        departments = session.query(Department).order_by(Department.display_name).all()
        return [d.to_dict() for d in departments]
    finally:
        session.close()


@router.post("/departments")
def create_department(body: DepartmentCreate) -> dict[str, Any]:
    """Create a new department."""
    session = get_session()
    try:
        dept_id = _slugify(body.display_name)
        existing = session.get(Department, dept_id)
        if existing:
            raise HTTPException(status_code=409, detail=f"Department '{dept_id}' already exists")

        dept = Department(
            id=dept_id,
            display_name=body.display_name,
            color=body.color,
            description=body.description,
        )
        session.add(dept)
        session.commit()
        session.refresh(dept)
        return dept.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/departments/{dept_id}")
def update_department(dept_id: str, body: DepartmentUpdate) -> dict[str, Any]:
    """Update a department."""
    session = get_session()
    try:
        dept = session.get(Department, dept_id)
        if not dept:
            raise HTTPException(status_code=404, detail="Department not found")

        update_data = body.model_dump(exclude_none=True)
        for key, value in update_data.items():
            setattr(dept, key, value)

        session.commit()
        session.refresh(dept)
        return dept.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/departments/{dept_id}")
def delete_department(dept_id: str) -> dict[str, str]:
    """Delete a department. Devices are unassigned (SET NULL)."""
    session = get_session()
    try:
        dept = session.get(Department, dept_id)
        if not dept:
            raise HTTPException(status_code=404, detail="Department not found")

        # Unassign devices first (explicit for clarity)
        session.query(Device).filter(Device.department_id == dept_id).update(
            {"department_id": None}
        )
        session.delete(dept)
        session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
