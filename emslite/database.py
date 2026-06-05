"""SQLite database setup and session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine = None
_SessionLocal = None


def init_db(db_path: str | Path = "emslite.db") -> None:
    """Initialize the database engine and create tables."""
    global _engine, _SessionLocal
    p = Path(db_path).resolve()
    _engine = create_engine(f"sqlite:///{p}", echo=False)
    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)
    _migrate(_engine)


def _migrate(engine) -> None:
    """Add columns that may be missing from older database schemas."""
    insp = inspect(engine)
    # Add meter_name to devices table if missing
    if "devices" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("devices")}
        if "meter_name" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE devices ADD COLUMN meter_name VARCHAR(128)"))
    # Add solar_kwh to utility_bills table if missing
    if "utility_bills" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("utility_bills")}
        if "solar_kwh" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE utility_bills ADD COLUMN solar_kwh FLOAT"))


def get_session() -> Session:
    """Return a new database session."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def get_engine():
    if _engine is None:
        init_db()
    return _engine
