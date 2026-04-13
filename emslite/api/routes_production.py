"""Production metrics and workflow builder API.

Exposes:
  - CRUD for user-defined daily output metric definitions
  - CRUD for daily metric entries (+ bulk upload)
  - CRUD for drag-and-drop workflow graphs
  - Correlation endpoint joining daily metrics against per-panel kWh
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date as _date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core import amps_to_kw, get_device_voltage_map, load_csv, meter_columns
from ..database import get_session
from ..models import DailyMetricEntry, MetricDefinition, Workflow

logger = logging.getLogger(__name__)

router = APIRouter(tags=["production"])


# ─────────────────────────── Shared helpers ────────────────────────────────


def _get_master_path() -> Path:
    from .app import get_app_config, get_project_root

    cfg = get_app_config()
    root = get_project_root()
    data_dir = root / cfg.get("data_dir", "data")
    return data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")


def _get_config() -> dict:
    from .app import get_app_config

    return get_app_config()


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "metric"


def _parse_date(value: str | _date) -> _date:
    if isinstance(value, _date):
        return value
    return _date.fromisoformat(str(value)[:10])


# ─────────────────────────── Metric definitions ────────────────────────────


class MetricDefinitionCreate(BaseModel):
    id: str | None = None
    display_name: str
    unit: str = "count"
    description: str | None = None
    color: str = "#8BD435"
    sort_order: int = 0


class MetricDefinitionUpdate(BaseModel):
    display_name: str | None = None
    unit: str | None = None
    description: str | None = None
    color: str | None = None
    sort_order: int | None = None


@router.get("/metric-definitions")
def list_metric_definitions() -> list[dict[str, Any]]:
    session = get_session()
    try:
        defs = (
            session.query(MetricDefinition)
            .order_by(MetricDefinition.sort_order, MetricDefinition.display_name)
            .all()
        )
        return [d.to_dict() for d in defs]
    finally:
        session.close()


@router.post("/metric-definitions")
def create_metric_definition(body: MetricDefinitionCreate) -> dict[str, Any]:
    session = get_session()
    try:
        metric_id = body.id or _slugify(body.display_name)
        if session.get(MetricDefinition, metric_id):
            raise HTTPException(status_code=409, detail=f"Metric '{metric_id}' already exists")

        obj = MetricDefinition(
            id=metric_id,
            display_name=body.display_name,
            unit=body.unit,
            description=body.description,
            color=body.color,
            sort_order=body.sort_order,
        )
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/metric-definitions/{metric_id}")
def update_metric_definition(metric_id: str, body: MetricDefinitionUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        obj = session.get(MetricDefinition, metric_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Metric definition not found")

        for key, value in body.model_dump(exclude_none=True).items():
            setattr(obj, key, value)
        session.commit()
        session.refresh(obj)
        return obj.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/metric-definitions/{metric_id}")
def delete_metric_definition(metric_id: str) -> dict[str, str]:
    session = get_session()
    try:
        obj = session.get(MetricDefinition, metric_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Metric definition not found")
        session.delete(obj)
        session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────── Daily metric entries ──────────────────────────


class DailyMetricCreate(BaseModel):
    metric_def_id: str
    entry_date: str  # ISO date YYYY-MM-DD
    value: float
    notes: str | None = None


class DailyMetricUpdate(BaseModel):
    value: float | None = None
    notes: str | None = None
    entry_date: str | None = None


class BulkRow(BaseModel):
    date: str
    value: float


class DailyMetricBulkUpload(BaseModel):
    metric_def_id: str
    rows: list[BulkRow] | None = None
    csv: str | None = None  # raw text "date,value\n..."


@router.get("/daily-metrics")
def list_daily_metrics(
    metric_def_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
) -> list[dict[str, Any]]:
    session = get_session()
    try:
        q = session.query(DailyMetricEntry)
        if metric_def_id:
            q = q.filter(DailyMetricEntry.metric_def_id == metric_def_id)
        if start:
            q = q.filter(DailyMetricEntry.entry_date >= _parse_date(start))
        if end:
            q = q.filter(DailyMetricEntry.entry_date <= _parse_date(end))
        entries = q.order_by(DailyMetricEntry.entry_date).all()
        return [e.to_dict() for e in entries]
    finally:
        session.close()


def _upsert_entry(
    session, metric_def_id: str, entry_date: _date, value: float, notes: str | None = None
) -> tuple[DailyMetricEntry, bool]:
    """Insert or update a (metric, date) entry. Returns (entry, created)."""
    existing = (
        session.query(DailyMetricEntry)
        .filter(
            DailyMetricEntry.metric_def_id == metric_def_id,
            DailyMetricEntry.entry_date == entry_date,
        )
        .one_or_none()
    )
    if existing:
        existing.value = value
        if notes is not None:
            existing.notes = notes
        return existing, False
    new = DailyMetricEntry(
        metric_def_id=metric_def_id,
        entry_date=entry_date,
        value=value,
        notes=notes,
    )
    session.add(new)
    return new, True


@router.post("/daily-metrics")
def create_daily_metric(body: DailyMetricCreate) -> dict[str, Any]:
    session = get_session()
    try:
        if not session.get(MetricDefinition, body.metric_def_id):
            raise HTTPException(status_code=404, detail="Metric definition not found")
        entry, _created = _upsert_entry(
            session, body.metric_def_id, _parse_date(body.entry_date), body.value, body.notes
        )
        session.commit()
        session.refresh(entry)
        return entry.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/daily-metrics/{entry_id}")
def update_daily_metric(entry_id: int, body: DailyMetricUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        entry = session.get(DailyMetricEntry, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        data = body.model_dump(exclude_none=True)
        if "entry_date" in data:
            entry.entry_date = _parse_date(data.pop("entry_date"))
        for key, value in data.items():
            setattr(entry, key, value)
        session.commit()
        session.refresh(entry)
        return entry.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/daily-metrics/{entry_id}")
def delete_daily_metric(entry_id: int) -> dict[str, str]:
    session = get_session()
    try:
        entry = session.get(DailyMetricEntry, entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        session.delete(entry)
        session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _parse_csv_rows(csv_text: str) -> tuple[list[tuple[_date, float]], list[str]]:
    """Parse ``date,value`` CSV text (header optional). Returns (rows, errors)."""
    rows: list[tuple[_date, float]] = []
    errors: list[str] = []
    for line_num, raw in enumerate(csv_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            errors.append(f"line {line_num}: not enough columns")
            continue
        date_str, value_str = parts[0], parts[1]
        if line_num == 1 and not re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
            # Header row — skip.
            continue
        try:
            d = _parse_date(date_str)
        except Exception:
            errors.append(f"line {line_num}: bad date '{date_str}'")
            continue
        try:
            v = float(value_str)
        except ValueError:
            errors.append(f"line {line_num}: bad value '{value_str}'")
            continue
        rows.append((d, v))
    return rows, errors


@router.post("/daily-metrics/bulk-upload")
def bulk_upload_daily_metrics(body: DailyMetricBulkUpload) -> dict[str, Any]:
    session = get_session()
    try:
        if not session.get(MetricDefinition, body.metric_def_id):
            raise HTTPException(status_code=404, detail="Metric definition not found")

        pairs: list[tuple[_date, float]] = []
        errors: list[str] = []

        if body.rows:
            for r in body.rows:
                try:
                    pairs.append((_parse_date(r.date), float(r.value)))
                except Exception as exc:
                    errors.append(f"row {r.date}: {exc}")
        elif body.csv:
            parsed, errs = _parse_csv_rows(body.csv)
            pairs.extend(parsed)
            errors.extend(errs)
        else:
            raise HTTPException(status_code=400, detail="Provide either 'rows' or 'csv'")

        inserted = 0
        updated = 0
        for entry_date, value in pairs:
            _entry, created = _upsert_entry(session, body.metric_def_id, entry_date, value)
            if created:
                inserted += 1
            else:
                updated += 1
        session.commit()
        return {"inserted": inserted, "updated": updated, "errors": errors}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────── Workflows ─────────────────────────────────


class WorkflowCreate(BaseModel):
    name: str
    description: str | None = None
    graph: dict[str, Any] | None = None
    is_active: bool = True


class WorkflowUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    graph: dict[str, Any] | None = None
    is_active: bool | None = None


@router.get("/workflows")
def list_workflows() -> list[dict[str, Any]]:
    session = get_session()
    try:
        flows = session.query(Workflow).order_by(Workflow.name).all()
        return [w.to_dict(include_graph=False) for w in flows]
    finally:
        session.close()


@router.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: int) -> dict[str, Any]:
    session = get_session()
    try:
        wf = session.get(Workflow, workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return wf.to_dict(include_graph=True)
    finally:
        session.close()


@router.post("/workflows")
def create_workflow(body: WorkflowCreate) -> dict[str, Any]:
    session = get_session()
    try:
        wf = Workflow(
            name=body.name,
            description=body.description,
            graph_json=json.dumps(body.graph or {"nodes": [], "edges": []}),
            is_active=body.is_active,
        )
        session.add(wf)
        session.commit()
        session.refresh(wf)
        return wf.to_dict(include_graph=True)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/workflows/{workflow_id}")
def update_workflow(workflow_id: int, body: WorkflowUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        wf = session.get(Workflow, workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        data = body.model_dump(exclude_none=True)
        if "graph" in data:
            wf.graph_json = json.dumps(data.pop("graph"))
        for key, value in data.items():
            setattr(wf, key, value)
        session.commit()
        session.refresh(wf)
        return wf.to_dict(include_graph=True)
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: int) -> dict[str, str]:
    session = get_session()
    try:
        wf = session.get(Workflow, workflow_id)
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        session.delete(wf)
        session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ────────────────────────────── Correlation ────────────────────────────────


def _clean_float(x: float) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(float(x), 3)


@router.get("/production/correlation")
def production_correlation(
    start: str | None = Query(None),
    end: str | None = Query(None),
    metric_def_ids: str | None = Query(None, description="Comma-separated metric ids"),
    panels: str | None = Query(None, description="Comma-separated panel ids"),
) -> dict[str, Any]:
    """Return daily-aligned kWh and metric series for a correlation chart."""
    master = _get_master_path()
    cfg = _get_config()
    line_voltage = float(cfg.get("line_voltage", 480.0))
    power_factor = float(cfg.get("power_factor", 1.0))
    voltage_map = get_device_voltage_map()

    # ── Pick panels ──
    requested_panels: list[str] = []
    if panels:
        requested_panels = [p.strip() for p in panels.split(",") if p.strip()]

    kwh_by_panel: dict[str, list[float | None]] = {}
    kwh_total: list[float | None] = []
    dates: list[str] = []

    if master.exists():
        df = load_csv(master)
        if start:
            df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)]
        if end:
            df = df[df["Timestamp"] <= pd.to_datetime(end, utc=True) + pd.Timedelta(days=1)]

        all_panels = meter_columns(df.columns, exclude=set(cfg.get("combo_columns", {}).keys()))
        if not requested_panels:
            requested_panels = all_panels[:]
        else:
            requested_panels = [p for p in requested_panels if p in all_panels]

        if not df.empty and requested_panels:
            df = df.set_index("Timestamp")
            # Convert each panel's amps series to kW, then resample to daily kWh
            # (mean kW over the day × 24 h).
            daily_kwh_frame = pd.DataFrame(index=df.index)
            for col in requested_panels:
                if col not in df.columns:
                    continue
                v = voltage_map.get(col, line_voltage)
                kw = amps_to_kw(df[col].fillna(0), v, power_factor).fillna(0)
                daily_kwh_frame[col] = kw
            daily = daily_kwh_frame.resample("1D").mean() * 24.0
            daily = daily.fillna(0.0)
            dates = [d.strftime("%Y-%m-%d") for d in daily.index]
            for col in requested_panels:
                if col in daily.columns:
                    kwh_by_panel[col] = [_clean_float(v) for v in daily[col].tolist()]
            if not daily.empty:
                kwh_total = [_clean_float(v) for v in daily.sum(axis=1).tolist()]

    # ── Pull metric entries and align to date index ──
    metrics_out: dict[str, dict[str, Any]] = {}
    session = get_session()
    try:
        metric_ids: list[str] = []
        if metric_def_ids:
            metric_ids = [m.strip() for m in metric_def_ids.split(",") if m.strip()]

        if metric_ids:
            defs = (
                session.query(MetricDefinition)
                .filter(MetricDefinition.id.in_(metric_ids))
                .all()
            )
        else:
            defs = (
                session.query(MetricDefinition)
                .order_by(MetricDefinition.sort_order, MetricDefinition.display_name)
                .all()
            )

        # If there are no panel-driven dates yet, fall back to the date range
        # spanned by the metric entries themselves so the user still sees data.
        if not dates and defs:
            q = session.query(DailyMetricEntry).filter(
                DailyMetricEntry.metric_def_id.in_([d.id for d in defs])
            )
            if start:
                q = q.filter(DailyMetricEntry.entry_date >= _parse_date(start))
            if end:
                q = q.filter(DailyMetricEntry.entry_date <= _parse_date(end))
            all_entries = q.order_by(DailyMetricEntry.entry_date).all()
            if all_entries:
                unique_dates = sorted({e.entry_date for e in all_entries})
                dates = [d.isoformat() for d in unique_dates]

        date_index = pd.to_datetime(dates) if dates else pd.DatetimeIndex([])

        for mdef in defs:
            q = session.query(DailyMetricEntry).filter(
                DailyMetricEntry.metric_def_id == mdef.id
            )
            if start:
                q = q.filter(DailyMetricEntry.entry_date >= _parse_date(start))
            if end:
                q = q.filter(DailyMetricEntry.entry_date <= _parse_date(end))
            entries = q.all()
            if entries:
                s = pd.Series(
                    {pd.Timestamp(e.entry_date): e.value for e in entries},
                    dtype="float64",
                )
            else:
                s = pd.Series(dtype="float64")
            if len(date_index):
                s = s.reindex(date_index)
            values = [None if pd.isna(v) else _clean_float(v) for v in s.tolist()]
            metrics_out[mdef.id] = {
                "display_name": mdef.display_name,
                "unit": mdef.unit,
                "color": mdef.color,
                "values": values,
            }
    finally:
        session.close()

    return {
        "dates": dates,
        "kwh_by_panel": kwh_by_panel,
        "kwh_total": kwh_total,
        "metrics": metrics_out,
    }


# ────────────────────────────── Seeding ────────────────────────────────────


def seed_default_metric_definitions(app_config: dict) -> None:
    """Insert starter metric definitions on first startup.

    Only runs when the metric_definitions table is empty; users can freely
    delete seeds afterward without them being re-added.
    """
    seeds = (app_config.get("production") or {}).get("default_metrics") or []
    if not seeds:
        return
    session = get_session()
    try:
        if session.query(MetricDefinition).count() > 0:
            return
        for row in seeds:
            session.add(
                MetricDefinition(
                    id=row["id"],
                    display_name=row["display_name"],
                    unit=row.get("unit", "count"),
                    color=row.get("color", "#8BD435"),
                    sort_order=row.get("sort_order", 0),
                    description=row.get("description"),
                )
            )
        session.commit()
        logger.info("Seeded %d default metric definitions", len(seeds))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
