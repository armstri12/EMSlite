"""SQLAlchemy models for device metadata, departments, and alerts."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    color: Mapped[str] = mapped_column(String(7), default="#8BD435")
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    devices: Mapped[list[Device]] = relationship(back_populates="department")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "color": self.color,
            "description": self.description,
            "device_count": len(self.devices) if self.devices else 0,
        }


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    department_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("departments.id", ondelete="SET NULL"), default=None
    )
    location: Mapped[str | None] = mapped_column(String(256), default=None)
    device_type: Mapped[str] = mapped_column(String(32), default="panel")
    rated_capacity: Mapped[float | None] = mapped_column(Float, default=None)
    voltage: Mapped[float | None] = mapped_column(Float, default=None)
    phase: Mapped[str] = mapped_column(String(16), default="3-phase")
    install_date: Mapped[date | None] = mapped_column(Date, default=None)
    tags: Mapped[str | None] = mapped_column(Text, default=None)  # JSON array string
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    warning_kw: Mapped[float | None] = mapped_column(Float, default=None)
    critical_kw: Mapped[float | None] = mapped_column(Float, default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    department: Mapped[Department | None] = relationship(back_populates="devices")

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "display_name": self.display_name,
            "department_id": self.department_id,
            "department_name": self.department.display_name if self.department else None,
            "location": self.location,
            "device_type": self.device_type,
            "rated_capacity": self.rated_capacity,
            "voltage": self.voltage,
            "phase": self.phase,
            "install_date": self.install_date.isoformat() if self.install_date else None,
            "tags": _json.loads(self.tags) if self.tags else [],
            "notes": self.notes,
            "warning_kw": self.warning_kw,
            "critical_kw": self.critical_kw,
            "enabled": self.enabled,
        }


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("devices.id", ondelete="CASCADE"), default=None
    )
    rule_type: Mapped[str] = mapped_column(String(32))  # threshold, offline, anomaly, spike
    threshold_value: Mapped[float | None] = mapped_column(Float, default=None)
    severity: Mapped[str] = mapped_column(String(16), default="warning")  # warning, critical
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "rule_type": self.rule_type,
            "threshold_value": self.threshold_value,
            "severity": self.severity,
            "enabled": self.enabled,
        }


class IngestLog(Base):
    __tablename__ = "ingest_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16))  # success, failed
    rows_added: Mapped[int] = mapped_column(default=0)
    device_id: Mapped[str | None] = mapped_column(String(128), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    processed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "rows_added": self.rows_added,
            "device_id": self.device_id,
            "error_message": self.error_message,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
