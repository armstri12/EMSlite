"""SQLAlchemy models for device metadata, departments, and alerts."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
    meter_name: Mapped[str | None] = mapped_column(String(128), default=None)
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
            "meter_name": self.meter_name,
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


class FloorPlan(Base):
    __tablename__ = "floor_plans"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    show_on_dashboard: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    pins: Mapped[list[FloorPlanPin]] = relationship(
        back_populates="floor_plan", cascade="all, delete-orphan"
    )
    zones: Mapped[list[FloorPlanZone]] = relationship(
        back_populates="floor_plan", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "image_path": self.image_path,
            "show_on_dashboard": self.show_on_dashboard,
            "pin_count": len(self.pins) if self.pins else 0,
            "zone_count": len(self.zones) if self.zones else 0,
        }


class FloorPlanPin(Base):
    __tablename__ = "floor_plan_pins"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    floor_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("floor_plans.id", ondelete="CASCADE"), nullable=False
    )
    device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    x_pct: Mapped[float] = mapped_column(Float, nullable=False)
    y_pct: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), default=None)

    floor_plan: Mapped[FloorPlan] = relationship(back_populates="pins")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "floor_plan_id": self.floor_plan_id,
            "device_id": self.device_id,
            "x_pct": self.x_pct,
            "y_pct": self.y_pct,
            "label": self.label,
        }


class FloorPlanZone(Base):
    """A polygon zone on a floor plan, linked to a device/panel."""

    __tablename__ = "floor_plan_zones"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    floor_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("floor_plans.id", ondelete="CASCADE"), nullable=False
    )
    device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), default=None)
    points: Mapped[str] = mapped_column(Text, nullable=False)  # JSON: [{x: float, y: float}, ...]

    floor_plan: Mapped[FloorPlan] = relationship(back_populates="zones")

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "floor_plan_id": self.floor_plan_id,
            "device_id": self.device_id,
            "label": self.label,
            "points": _json.loads(self.points) if self.points else [],
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


class WeatherCache(Base):
    """Cached hourly weather data from NOAA NCEI."""

    __tablename__ = "weather_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    temperature_c: Mapped[float | None] = mapped_column(Float, default=None)
    humidity_pct: Mapped[float | None] = mapped_column(Float, default=None)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
        }


class UtilityBill(Base):
    """Utility bill records for tracking actual costs per meter."""

    __tablename__ = "utility_bills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    meter_name: Mapped[str] = mapped_column(String(128), nullable=False)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    bill_date: Mapped[date | None] = mapped_column(Date, default=None)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "meter_name": self.meter_name,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "bill_date": self.bill_date.isoformat() if self.bill_date else None,
            "amount": self.amount,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MetricDefinition(Base):
    """A user-defined daily output metric (e.g. "final tests per day")."""

    __tablename__ = "metric_definitions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    unit: Mapped[str] = mapped_column(String(32), default="count")
    description: Mapped[str | None] = mapped_column(Text, default=None)
    color: Mapped[str] = mapped_column(String(7), default="#8BD435")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    entries: Mapped[list[DailyMetricEntry]] = relationship(
        back_populates="definition", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "unit": self.unit,
            "description": self.description,
            "color": self.color,
            "sort_order": self.sort_order,
        }


class DailyMetricEntry(Base):
    """A single per-day value for a MetricDefinition."""

    __tablename__ = "daily_metric_entries"
    __table_args__ = (
        UniqueConstraint("metric_def_id", "entry_date", name="uq_metric_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    metric_def_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("metric_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entry_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    definition: Mapped[MetricDefinition] = relationship(back_populates="entries")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "metric_def_id": self.metric_def_id,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "value": self.value,
            "notes": self.notes,
        }


class Workflow(Base):
    """A production workflow graph built in the drag-and-drop editor.

    The node/edge graph is stored as JSON in ``graph_json``. Canonical shape:

        {
          "nodes": [{"id", "label", "x", "y", "type",
                     "panel_ids": [...], "metric_def_ids": [...]}],
          "edges": [{"id", "from", "to"}]
        }
    """

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    graph_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self, include_graph: bool = True) -> dict:
        import json as _json

        out = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_graph:
            try:
                out["graph"] = _json.loads(self.graph_json) if self.graph_json else {"nodes": [], "edges": []}
            except Exception:
                out["graph"] = {"nodes": [], "edges": []}
        return out


class AlertEvent(Base):
    """Acknowledgement state for computed alert events."""

    __tablename__ = "alert_events"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    event_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "device_id": self.device_id,
            "severity": self.severity,
            "event_ts": self.event_ts.isoformat() if self.event_ts else None,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }
