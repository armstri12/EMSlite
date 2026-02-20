# EMSlite Data Model

_Last reviewed: 2026-02-20_

This document describes the SQLAlchemy ORM models, their relationships, and storage conventions.

All models are defined in `emslite/models.py` and inherit from `Base` (a `DeclarativeBase`). Tables are auto-created by `Base.metadata.create_all()` in `database.py` on application startup.

---

## Entity relationship diagram

```
┌─────────────┐       ┌──────────────┐
│ Department   │──1:N──│    Device     │
│              │       │              │
│  id (PK)     │       │  id (PK)     │
│  display_name│       │  display_name│
│  color       │       │  department_id│──FK→ departments.id (SET NULL on delete)
│  description │       │  location    │
│  created_at  │       │  device_type │
└─────────────┘       │  rated_cap.  │
                       │  voltage     │
                       │  phase       │
                       │  install_date│
                       │  tags (JSON) │
                       │  notes       │
                       │  warning_kw  │
                       │  critical_kw │
                       │  enabled     │
                       │  created_at  │
                       │  updated_at  │
                       └──────┬───────┘
                              │
                     ┌────────┴────────┐
                     │                 │
              ┌──────┴──────┐   ┌──────┴──────┐
              │  AlertRule   │   │ FloorPlanPin │
              │             │   │              │
              │  id (PK)    │   │  id (PK)     │
              │  device_id  │   │  device_id   │
              │  rule_type  │   │  floor_plan_id│
              │  threshold  │   │  x_pct, y_pct│
              │  severity   │   │  label       │
              │  enabled    │   └──────────────┘
              │  created_at │
              └─────────────┘

┌───────────────┐       ┌───────────────┐
│  FloorPlan     │──1:N──│ FloorPlanPin   │
│               │       │               │
│  id (PK)      │──1:N──│ FloorPlanZone  │
│  name         │       │               │
│  image_path   │       │  id (PK)      │
│  show_on_dash │       │  floor_plan_id│
│  created_at   │       │  device_id    │
└───────────────┘       │  label        │
                        │  points (JSON)│
                        └───────────────┘

┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ WeatherCache   │    │  IngestLog     │    │  AlertEvent    │
│               │    │               │    │               │
│  id (PK)      │    │  id (PK)      │    │  key (PK)     │
│  station_id   │    │  filename     │    │  device_id    │
│  timestamp*   │    │  status       │    │  severity     │
│  temperature_c│    │  rows_added   │    │  event_ts*    │
│  humidity_pct │    │  device_id    │    │  acknowledged │
│  fetched_at   │    │  error_message│    │  ack_at*      │
└───────────────┘    │  processed_at │    └───────────────┘
                     └───────────────┘
* = DateTime(timezone=True)
```

---

## Table details

### Department

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | String(64) | PK | Slug-style identifier (e.g., `production`) |
| `display_name` | String(128) | NOT NULL | Human-readable name |
| `color` | String(7) | default `#8BD435` | Hex color for UI |
| `description` | Text | nullable | Optional description |
| `created_at` | DateTime | server_default=now() | Audit timestamp (naive) |

**Relationship**: `devices` → list of `Device` (back_populates)

### Device

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | String(128) | PK | Matches CSV column name (e.g., `MeterATS01_SystemCurrent`) |
| `display_name` | String(128) | NOT NULL | Human-readable name |
| `department_id` | String(64) | FK → departments.id, SET NULL on delete | nullable |
| `location` | String(256) | nullable | Physical location description |
| `device_type` | String(32) | default `panel` | `panel`, `meter`, etc. |
| `rated_capacity` | Float | nullable | Rated capacity in kW |
| `voltage` | Float | nullable | Per-device voltage override |
| `phase` | String(16) | default `3-phase` | Phase configuration |
| `install_date` | Date | nullable | Installation date |
| `tags` | Text | nullable | JSON array string (parsed in `to_dict()`) |
| `notes` | Text | nullable | Free-text notes |
| `warning_kw` | Float | nullable | kW threshold for warning alerts |
| `critical_kw` | Float | nullable | kW threshold for critical alerts |
| `enabled` | Boolean | default True | Whether to include in analytics |
| `created_at` | DateTime | server_default=now() | Audit timestamp (naive) |
| `updated_at` | DateTime | server_default=now(), onupdate=now() | Last modified (naive) |

**Critical convention**: The `id` field must exactly match the CSV column name for the panel. This is how API routes join device metadata to time-series data.

### AlertRule

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `device_id` | String(128) | FK → devices.id, CASCADE on delete | nullable (facility-wide rules) |
| `rule_type` | String(32) | NOT NULL | `threshold`, `offline`, `anomaly`, `spike` |
| `threshold_value` | Float | nullable | kW threshold for the rule |
| `severity` | String(16) | default `warning` | `warning` or `critical` |
| `enabled` | Boolean | default True | |
| `created_at` | DateTime | server_default=now() | Audit timestamp (naive) |

### AlertEvent

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `key` | String(256) | PK | Composite key (typically `{device_id}:{rule_type}:{timestamp}`) |
| `device_id` | String(128) | NOT NULL | |
| `severity` | String(16) | NOT NULL | |
| `event_ts` | DateTime(tz=True) | NOT NULL | When the alert condition was detected |
| `acknowledged` | Boolean | default False | |
| `acknowledged_at` | DateTime(tz=True) | nullable | When the alert was acknowledged |

### FloorPlan

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `name` | String(128) | NOT NULL | Display name |
| `image_path` | String(512) | NOT NULL | Path to floor plan image |
| `show_on_dashboard` | Boolean | default False | Show on executive dashboard |
| `created_at` | DateTime | server_default=now() | Audit timestamp (naive) |

**Relationships**: `pins` → list of `FloorPlanPin`, `zones` → list of `FloorPlanZone` (both cascade delete)

### FloorPlanPin

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `floor_plan_id` | Integer | FK → floor_plans.id, CASCADE | |
| `device_id` | String(128) | NOT NULL | Panel to display at this pin |
| `x_pct` | Float | NOT NULL | X position as percentage (0-100) |
| `y_pct` | Float | NOT NULL | Y position as percentage (0-100) |
| `label` | String(128) | nullable | Override label |

### FloorPlanZone

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `floor_plan_id` | Integer | FK → floor_plans.id, CASCADE | |
| `device_id` | String(128) | NOT NULL | Panel this zone represents |
| `label` | String(128) | nullable | Override label |
| `points` | Text | NOT NULL | JSON array of `{x, y}` coordinates (percentages) |

### WeatherCache

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `station_id` | String(32) | NOT NULL, indexed | NOAA station ID |
| `timestamp` | DateTime(tz=True) | NOT NULL, indexed | Observation time (UTC) |
| `temperature_c` | Float | nullable | Temperature in Celsius |
| `humidity_pct` | Float | nullable | Relative humidity (0-100) |
| `fetched_at` | DateTime | server_default=now() | When cached (naive) |

### IngestLog

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | Integer | PK, autoincrement | |
| `filename` | String(256) | NOT NULL | Source CSV filename |
| `status` | String(16) | | `success` or `failed` |
| `rows_added` | Integer | default 0 | New rows appended |
| `device_id` | String(128) | nullable | Panel/meter ID |
| `error_message` | Text | nullable | Error details if failed |
| `processed_at` | DateTime | server_default=now() | Ingestion time (naive) |

---

## Serialization convention

All models implement a `to_dict()` method that returns a plain dict suitable for JSON serialization. These methods handle:
- Date/datetime → ISO 8601 string conversion
- JSON string columns (tags, points) → parsed Python objects
- Relationship counts (e.g., `device_count`, `pin_count`)

API routes call `to_dict()` on query results before returning them as JSON responses.

---

## DateTime timezone conventions

| Column | Timezone | Rationale |
|--------|---------|-----------|
| `WeatherCache.timestamp` | `DateTime(timezone=True)` | Written with UTC-aware values from NOAA |
| `AlertEvent.event_ts` | `DateTime(timezone=True)` | Alert detection times are UTC-aware |
| `AlertEvent.acknowledged_at` | `DateTime(timezone=True)` | Acknowledgment times are UTC-aware |
| All other DateTime columns | `DateTime` (naive) | Audit timestamps using `server_default=func.now()` |

**Convention**: Treat all naive DateTime values as UTC. SQLite does not enforce timezone semantics, so timezone information may be stripped on roundtrip. See [temporal-assumptions.md](temporal-assumptions.md#database-datetime-columns) for details.
