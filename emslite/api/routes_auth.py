"""Admin authentication endpoints."""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .app import get_app_config

router = APIRouter(tags=["auth"])


def _get_admin_password() -> str:
    env_pw = os.environ.get("ADMIN_PASSWORD")
    if env_pw:
        return env_pw
    cfg = get_app_config()
    return cfg.get("admin_password", "admin")


def require_admin(request: Request) -> None:
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")


class LoginRequest(BaseModel):
    password: str


@router.get("/auth/me")
def auth_me(request: Request):
    return {"authenticated": bool(request.session.get("is_admin"))}


@router.post("/auth/login")
def auth_login(request: Request, body: LoginRequest):
    expected = _get_admin_password()
    if not secrets.compare_digest(body.password, expected):
        raise HTTPException(status_code=401, detail="Invalid password")
    request.session["is_admin"] = True
    return {"authenticated": True}


@router.post("/auth/logout")
def auth_logout(request: Request):
    request.session.clear()
    return {"authenticated": False}
