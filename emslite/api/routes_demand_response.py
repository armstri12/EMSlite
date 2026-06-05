"""Demand response program management and performance tracking."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core import amps_to_kw, get_device_voltage_map, load_csv, meter_columns
from ..database import get_session
from ..models import DREvent, DRProgram

router = APIRouter(tags=["demand_response"])

_ET = ZoneInfo("America/New_York")


# ── Pydantic schemas ────────────────────────────────────────────────────────

class ProgramCreate(BaseModel):
    name: str
    utility: str = "Mass Save"
    program_type: str = "connected_solutions"
    season_year: int
    season_start: str | None = None
    season_end: str | None = None
    event_window_start: int = 14
    event_window_end: int = 19
    committed_kw: float | None = None
    incentive_rate: float | None = None
    enrolled_panels: list[str] = []
    notes: str | None = None
    active: bool = True


class ProgramUpdate(BaseModel):
    name: str | None = None
    utility: str | None = None
    program_type: str | None = None
    season_year: int | None = None
    season_start: str | None = None
    season_end: str | None = None
    event_window_start: int | None = None
    event_window_end: int | None = None
    committed_kw: float | None = None
    incentive_rate: float | None = None
    enrolled_panels: list[str] | None = None
    notes: str | None = None
    active: bool | None = None


class EventCreate(BaseModel):
    program_id: int
    event_date: str  # ISO date
    start_hour: int | None = None   # defaults to program window
    end_hour: int | None = None
    status: str = "scheduled"
    notes: str | None = None


class EventUpdate(BaseModel):
    event_date: str | None = None
    start_hour: int | None = None
    end_hour: int | None = None
    status: str | None = None
    notes: str | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _master_path():
    from .app import get_app_config, get_project_root
    cfg = get_app_config()
    root = get_project_root()
    data_dir = root / cfg.get("data_dir", "data")
    return data_dir / cfg.get("master_filename", "RawPanelUsageHistory.csv")


def _app_cfg():
    from .app import get_app_config
    return get_app_config()


def _build_kw_series(df: pd.DataFrame, panels: list[str], voltage_map: dict,
                     line_voltage: float, power_factor: float,
                     calibration_factor: float = 1.0) -> pd.Series:
    """Sum panel amps → total kW for the enrolled panels."""
    total = pd.Series(0.0, index=df.index)
    for col in panels:
        if col in df.columns:
            v = voltage_map.get(col, line_voltage)
            total += amps_to_kw(df[col].fillna(0), v, power_factor, calibration_factor).fillna(0)
    return total


def _compute_baseline(
    df: pd.DataFrame,
    panels: list[str],
    event_date: date,
    start_hour: int,
    end_hour: int,
    line_voltage: float,
    power_factor: float,
    voltage_map: dict,
    other_event_dates: set[date],
    calibration_factor: float = 1.0,
) -> tuple[float | None, list[str], float]:
    """
    10-of-10 baseline with same-day adjustment (Mass Save Connected Solutions method).

    Returns (adjusted_baseline_kw, baseline_date_strings, adjustment_factor).
    """
    total_kw = _build_kw_series(df, panels, voltage_map, line_voltage, power_factor, calibration_factor)

    df_et = df[["Timestamp"]].copy()
    df_et["kw"] = total_kw.values
    local_ts = df_et["Timestamp"].dt.tz_convert(_ET)
    df_et["et_date"] = local_ts.dt.date
    df_et["et_hour"] = local_ts.dt.hour

    # Collect up to 10 eligible weekdays before event_date (up to 45 calendar days back)
    candidate_days: list[tuple[date, float]] = []
    check = event_date - timedelta(days=1)
    while len(candidate_days) < 10 and (event_date - check).days <= 45:
        if check.weekday() < 5 and check not in other_event_dates:
            mask = (
                (df_et["et_date"] == check) &
                (df_et["et_hour"] >= start_hour) &
                (df_et["et_hour"] < end_hour)
            )
            day_kw = df_et.loc[mask, "kw"]
            if len(day_kw) > 0:
                candidate_days.append((check, float(day_kw.mean())))
        check -= timedelta(days=1)

    if not candidate_days:
        return None, [], 1.0

    raw_baseline = sum(avg for _, avg in candidate_days) / len(candidate_days)
    baseline_date_strs = [d.isoformat() for d, _ in candidate_days]

    # Same-day adjustment: compare actual vs. baseline kW in 2-hour pre-event window
    pre_start = max(start_hour - 2, 0)
    pre_end = start_hour

    event_pre = df_et.loc[
        (df_et["et_date"] == event_date) &
        (df_et["et_hour"] >= pre_start) &
        (df_et["et_hour"] < pre_end),
        "kw",
    ]
    if event_pre.empty:
        return round(raw_baseline, 2), baseline_date_strs, 1.0

    actual_pre = float(event_pre.mean())

    baseline_pre_vals = []
    for d, _ in candidate_days:
        pre_mask = (
            (df_et["et_date"] == d) &
            (df_et["et_hour"] >= pre_start) &
            (df_et["et_hour"] < pre_end)
        )
        day_pre = df_et.loc[pre_mask, "kw"]
        if not day_pre.empty:
            baseline_pre_vals.append(float(day_pre.mean()))

    if not baseline_pre_vals:
        return round(raw_baseline, 2), baseline_date_strs, 1.0

    baseline_pre = sum(baseline_pre_vals) / len(baseline_pre_vals)
    adj_factor = (actual_pre / baseline_pre) if baseline_pre > 0 else 1.0
    adj_factor = max(0.6, min(1.4, adj_factor))  # cap at ±40%

    return round(raw_baseline * adj_factor, 2), baseline_date_strs, round(adj_factor, 3)


def _actual_event_kw(
    df: pd.DataFrame,
    panels: list[str],
    event_date: date,
    start_hour: int,
    end_hour: int,
    line_voltage: float,
    power_factor: float,
    voltage_map: dict,
    calibration_factor: float = 1.0,
) -> float | None:
    total_kw = _build_kw_series(df, panels, voltage_map, line_voltage, power_factor, calibration_factor)
    df_et = df[["Timestamp"]].copy()
    df_et["kw"] = total_kw.values
    local_ts = df_et["Timestamp"].dt.tz_convert(_ET)
    df_et["et_date"] = local_ts.dt.date
    df_et["et_hour"] = local_ts.dt.hour
    mask = (
        (df_et["et_date"] == event_date) &
        (df_et["et_hour"] >= start_hour) &
        (df_et["et_hour"] < end_hour)
    )
    event_kw = df_et.loc[mask, "kw"]
    return round(float(event_kw.mean()), 2) if not event_kw.empty else None


# ── Program endpoints ────────────────────────────────────────────────────────

@router.get("/demand-response/programs")
def list_programs() -> list[dict[str, Any]]:
    session = get_session()
    try:
        programs = session.query(DRProgram).order_by(DRProgram.season_year.desc(), DRProgram.id).all()
        return [p.to_dict() for p in programs]
    finally:
        session.close()


@router.post("/demand-response/programs")
def create_program(body: ProgramCreate) -> dict[str, Any]:
    session = get_session()
    try:
        prog = DRProgram(
            name=body.name,
            utility=body.utility,
            program_type=body.program_type,
            season_year=body.season_year,
            season_start=date.fromisoformat(body.season_start) if body.season_start else None,
            season_end=date.fromisoformat(body.season_end) if body.season_end else None,
            event_window_start=body.event_window_start,
            event_window_end=body.event_window_end,
            committed_kw=body.committed_kw,
            incentive_rate=body.incentive_rate,
            enrolled_panels=json.dumps(body.enrolled_panels),
            notes=body.notes,
            active=body.active,
        )
        session.add(prog)
        session.commit()
        session.refresh(prog)
        return prog.to_dict()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/demand-response/programs/{program_id}")
def update_program(program_id: int, body: ProgramUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        prog = session.get(DRProgram, program_id)
        if not prog:
            raise HTTPException(status_code=404, detail="Program not found")
        if body.name is not None:
            prog.name = body.name
        if body.utility is not None:
            prog.utility = body.utility
        if body.program_type is not None:
            prog.program_type = body.program_type
        if body.season_year is not None:
            prog.season_year = body.season_year
        if body.season_start is not None:
            prog.season_start = date.fromisoformat(body.season_start) if body.season_start else None
        if body.season_end is not None:
            prog.season_end = date.fromisoformat(body.season_end) if body.season_end else None
        if body.event_window_start is not None:
            prog.event_window_start = body.event_window_start
        if body.event_window_end is not None:
            prog.event_window_end = body.event_window_end
        if body.committed_kw is not None:
            prog.committed_kw = body.committed_kw
        if body.incentive_rate is not None:
            prog.incentive_rate = body.incentive_rate
        if body.enrolled_panels is not None:
            prog.enrolled_panels = json.dumps(body.enrolled_panels)
        if body.notes is not None:
            prog.notes = body.notes
        if body.active is not None:
            prog.active = body.active
        session.commit()
        session.refresh(prog)
        return prog.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/demand-response/programs/{program_id}")
def delete_program(program_id: int) -> dict[str, Any]:
    session = get_session()
    try:
        prog = session.get(DRProgram, program_id)
        if not prog:
            raise HTTPException(status_code=404, detail="Program not found")
        session.delete(prog)
        session.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Event endpoints ──────────────────────────────────────────────────────────

@router.get("/demand-response/events")
def list_events(
    program_id: int | None = Query(None),
    status: str | None = Query(None),
) -> list[dict[str, Any]]:
    session = get_session()
    try:
        q = session.query(DREvent)
        if program_id:
            q = q.filter(DREvent.program_id == program_id)
        if status:
            q = q.filter(DREvent.status == status)
        events = q.order_by(DREvent.event_date.desc()).all()
        return [e.to_dict() for e in events]
    finally:
        session.close()


@router.post("/demand-response/events")
def create_event(body: EventCreate) -> dict[str, Any]:
    session = get_session()
    try:
        prog = session.get(DRProgram, body.program_id)
        if not prog:
            raise HTTPException(status_code=404, detail="Program not found")
        ev = DREvent(
            program_id=body.program_id,
            event_date=date.fromisoformat(body.event_date),
            start_hour=body.start_hour if body.start_hour is not None else prog.event_window_start,
            end_hour=body.end_hour if body.end_hour is not None else prog.event_window_end,
            status=body.status,
            notes=body.notes,
        )
        session.add(ev)
        session.commit()
        session.refresh(ev)
        return ev.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.put("/demand-response/events/{event_id}")
def update_event(event_id: int, body: EventUpdate) -> dict[str, Any]:
    session = get_session()
    try:
        ev = session.get(DREvent, event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")
        if body.event_date is not None:
            ev.event_date = date.fromisoformat(body.event_date)
        if body.start_hour is not None:
            ev.start_hour = body.start_hour
        if body.end_hour is not None:
            ev.end_hour = body.end_hour
        if body.status is not None:
            ev.status = body.status
        if body.notes is not None:
            ev.notes = body.notes
        session.commit()
        session.refresh(ev)
        return ev.to_dict()
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.delete("/demand-response/events/{event_id}")
def delete_event(event_id: int) -> dict[str, Any]:
    session = get_session()
    try:
        ev = session.get(DREvent, event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")
        session.delete(ev)
        session.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.post("/demand-response/events/{event_id}/compute")
def compute_event(event_id: int) -> dict[str, Any]:
    """Compute 10-of-10 baseline and actual performance for an event, storing the results."""
    session = get_session()
    try:
        ev = session.get(DREvent, event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")
        prog = ev.program

        master = _master_path()
        if not master.exists():
            raise HTTPException(status_code=422, detail="No energy data available")

        cfg = _app_cfg()
        line_voltage = float(cfg.get("line_voltage", 480.0))
        power_factor = float(cfg.get("power_factor", 1.0))
        calibration_factor = float(cfg.get("calibration_factor", 1.0))
        voltage_map = get_device_voltage_map()

        panels = json.loads(prog.enrolled_panels) if prog.enrolled_panels else []

        df = load_csv(master)

        if not panels:
            panels = meter_columns(df.columns)

        # Other event dates for this program (exclude from baseline candidates)
        other_events = (
            session.query(DREvent.event_date)
            .filter(DREvent.program_id == prog.id, DREvent.id != ev.id)
            .all()
        )
        other_event_dates: set[date] = {r[0] for r in other_events}

        baseline_kw, baseline_dates, adj_factor = _compute_baseline(
            df, panels, ev.event_date, ev.start_hour, ev.end_hour,
            line_voltage, power_factor, voltage_map, other_event_dates,
            calibration_factor,
        )

        actual_kw = _actual_event_kw(
            df, panels, ev.event_date, ev.start_hour, ev.end_hour,
            line_voltage, power_factor, voltage_map, calibration_factor,
        )

        ev.baseline_kw = baseline_kw
        ev.actual_kw = actual_kw
        ev.reduction_kw = round(baseline_kw - actual_kw, 2) if (baseline_kw is not None and actual_kw is not None) else None
        ev.baseline_dates = json.dumps(baseline_dates)
        if actual_kw is not None:
            ev.status = "completed"
        session.commit()
        session.refresh(ev)

        return {
            **ev.to_dict(),
            "adjustment_factor": adj_factor,
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.get("/demand-response/events/{event_id}/profile")
def event_profile(event_id: int) -> dict[str, Any]:
    """Return minute-level kW data for the event window, for charting."""
    session = get_session()
    try:
        ev = session.get(DREvent, event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Event not found")
        prog = ev.program

        master = _master_path()
        if not master.exists():
            return {"timestamps": [], "actual_kw": [], "baseline_kw": ev.baseline_kw}

        cfg = _app_cfg()
        line_voltage = float(cfg.get("line_voltage", 480.0))
        power_factor = float(cfg.get("power_factor", 1.0))
        calibration_factor = float(cfg.get("calibration_factor", 1.0))
        voltage_map = get_device_voltage_map()
        panels = json.loads(prog.enrolled_panels) if prog.enrolled_panels else []

        df = load_csv(master)

        if not panels:
            panels = meter_columns(df.columns)

        total_kw = _build_kw_series(df, panels, voltage_map, line_voltage, power_factor, calibration_factor)

        df_et = df[["Timestamp"]].copy()
        df_et["kw"] = total_kw.values
        local_ts = df_et["Timestamp"].dt.tz_convert(_ET)
        df_et["et_date"] = local_ts.dt.date
        df_et["et_hour"] = local_ts.dt.hour

        mask = (
            (df_et["et_date"] == ev.event_date) &
            (df_et["et_hour"] >= ev.start_hour) &
            (df_et["et_hour"] < ev.end_hour)
        )
        window = df_et.loc[mask]

        return {
            "timestamps": df.loc[mask, "Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
            "actual_kw": [round(v, 2) for v in window["kw"].tolist()],
            "baseline_kw": ev.baseline_kw,
        }
    finally:
        session.close()


@router.get("/demand-response/season-summary/{program_id}")
def season_summary(program_id: int) -> dict[str, Any]:
    """Aggregate performance stats for a program's season."""
    session = get_session()
    try:
        prog = session.get(DRProgram, program_id)
        if not prog:
            raise HTTPException(status_code=404, detail="Program not found")

        events = (
            session.query(DREvent)
            .filter(DREvent.program_id == program_id)
            .order_by(DREvent.event_date)
            .all()
        )

        completed = [e for e in events if e.reduction_kw is not None]
        avg_reduction = (
            round(sum(e.reduction_kw for e in completed) / len(completed), 2)
            if completed else None
        )
        estimated_payment = (
            round(avg_reduction * prog.incentive_rate, 2)
            if (avg_reduction is not None and prog.incentive_rate)
            else None
        )
        pct_of_commitment = (
            round(avg_reduction / prog.committed_kw * 100, 1)
            if (avg_reduction is not None and prog.committed_kw)
            else None
        )

        return {
            "program": prog.to_dict(),
            "total_events": len(events),
            "completed_events": len(completed),
            "avg_reduction_kw": avg_reduction,
            "estimated_payment": estimated_payment,
            "pct_of_commitment": pct_of_commitment,
            "events": [e.to_dict() for e in events],
        }
    finally:
        session.close()
