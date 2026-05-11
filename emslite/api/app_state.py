"""Shared application state to avoid circular imports."""

from __future__ import annotations

from pathlib import Path

_app_config: dict = {}


def get_app_config() -> dict:
    return _app_config


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent
