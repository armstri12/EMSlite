"""SQLite database setup and session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
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


def get_session() -> Session:
    """Return a new database session."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def get_engine():
    if _engine is None:
        init_db()
    return _engine
