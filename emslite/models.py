"""SQLAlchemy models for device metadata, departments, and alerts."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
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


class DailyTemperature(Base):
    """User-uploaded daily temperature observations for HVAC/weather analysis.

    Values are stored in Celsius regardless of the unit they were uploaded
    in, so analysis math has a single canonical unit. ``min``/``max`` are
    optional. Uploaded rows override NOAA cache data for the same date.
    """

    __tablename__ = "daily_temperatures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entry_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    avg_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    min_temp_c: Mapped[float | None] = mapped_column(Float, default=None)
    max_temp_c: Mapped[float | None] = mapped_column(Float, default=None)
    source: Mapped[str] = mapped_column(String(16), default="upload")
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "avg_temp_c": self.avg_temp_c,
            "min_temp_c": self.min_temp_c,
            "max_temp_c": self.max_temp_c,
            "source": self.source,
            "notes": self.notes,
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
    # On-site generation (kWh) during the period that offset this meter. The
    # utility bills NET energy (consumption − solar), so reconciliation compares
    # metered consumption against (billed energy + solar_kwh).
    solar_kwh: Mapped[float | None] = mapped_column(Float, default=None)
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
            "solar_kwh": self.solar_kwh,
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


class WirelessGateway(Base):
    """A Monnit Alta Ethernet Gateway that pushes sensor data over TCP."""

    __tablename__ = "wireless_gateways"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # gatewayID / networkID
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sensors: Mapped[list[WirelessSensor]] = relationship(
        back_populates="gateway", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WirelessSensor(Base):
    """A Monnit Alta wireless sensor registered via TCP auto-discovery."""

    __tablename__ = "wireless_sensors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sensorID as string
    gateway_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("wireless_gateways.id", ondelete="CASCADE"), nullable=False
    )
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    sensor_type: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. temperature, humidity
    unit: Mapped[str | None] = mapped_column(String(32), default=None)  # user-configured unit label
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    tags: Mapped[str | None] = mapped_column(Text, default=None)  # JSON array string
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    gateway: Mapped[WirelessGateway] = relationship(back_populates="sensors")
    readings: Mapped[list[SensorReading]] = relationship(
        back_populates="sensor", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "gateway_id": self.gateway_id,
            "display_name": self.display_name,
            "sensor_type": self.sensor_type,
            "unit": self.unit,
            "enabled": self.enabled,
            "notes": self.notes,
            "tags": _json.loads(self.tags) if self.tags else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DRProgram(Base):
    """A demand response program enrollment (e.g. Mass Save Connected Solutions)."""

    __tablename__ = "dr_programs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    utility: Mapped[str] = mapped_column(String(64), default="Mass Save")
    program_type: Mapped[str] = mapped_column(String(32), default="connected_solutions")
    season_year: Mapped[int] = mapped_column(Integer, nullable=False)
    season_start: Mapped[date | None] = mapped_column(Date, default=None)
    season_end: Mapped[date | None] = mapped_column(Date, default=None)
    event_window_start: Mapped[int] = mapped_column(Integer, default=14)  # hour (2 pm)
    event_window_end: Mapped[int] = mapped_column(Integer, default=19)    # hour (7 pm)
    committed_kw: Mapped[float | None] = mapped_column(Float, default=None)
    incentive_rate: Mapped[float | None] = mapped_column(Float, default=None)  # $/kW-season
    enrolled_panels: Mapped[str | None] = mapped_column(Text, default=None)    # JSON list of panel IDs
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    events: Mapped[list["DREvent"]] = relationship(
        back_populates="program", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "name": self.name,
            "utility": self.utility,
            "program_type": self.program_type,
            "season_year": self.season_year,
            "season_start": self.season_start.isoformat() if self.season_start else None,
            "season_end": self.season_end.isoformat() if self.season_end else None,
            "event_window_start": self.event_window_start,
            "event_window_end": self.event_window_end,
            "committed_kw": self.committed_kw,
            "incentive_rate": self.incentive_rate,
            "enrolled_panels": _json.loads(self.enrolled_panels) if self.enrolled_panels else [],
            "notes": self.notes,
            "active": self.active,
            "event_count": len(self.events) if self.events else 0,
        }


class DREvent(Base):
    """A single demand response curtailment event."""

    __tablename__ = "dr_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    program_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dr_programs.id", ondelete="CASCADE"), nullable=False
    )
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    end_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled, completed, cancelled
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    # Computed results stored after analysis
    baseline_kw: Mapped[float | None] = mapped_column(Float, default=None)
    actual_kw: Mapped[float | None] = mapped_column(Float, default=None)
    reduction_kw: Mapped[float | None] = mapped_column(Float, default=None)
    baseline_dates: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list of date strings
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    program: Mapped["DRProgram"] = relationship(back_populates="events")

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "program_id": self.program_id,
            "program_name": self.program.name if self.program else None,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "status": self.status,
            "notes": self.notes,
            "baseline_kw": self.baseline_kw,
            "actual_kw": self.actual_kw,
            "reduction_kw": self.reduction_kw,
            "baseline_dates": _json.loads(self.baseline_dates) if self.baseline_dates else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SensorReading(Base):
    """A single time-stamped reading from a wireless sensor."""

    __tablename__ = "sensor_readings"
    __table_args__ = (
        Index("ix_sensor_readings_sensor_ts", "sensor_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("wireless_sensors.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    value: Mapped[float | None] = mapped_column(Float, default=None)
    signal_strength: Mapped[int | None] = mapped_column(Integer, default=None)
    battery_level: Mapped[int | None] = mapped_column(Integer, default=None)

    sensor: Mapped[WirelessSensor] = relationship(back_populates="readings")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "value": self.value,
            "signal_strength": self.signal_strength,
            "battery_level": self.battery_level,
        }
