"""Weekly report API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, Response

from ..report import generate_report_data, render_report_html

router = APIRouter(tags=["reports"])


@router.get("/reports/preview")
def preview_report(
    period_days: int = Query(7, ge=1, le=90, description="Report period in days"),
) -> HTMLResponse:
    """Generate and return the weekly report as HTML for browser preview."""
    data = generate_report_data(period_days=period_days)
    html = render_report_html(data)
    return HTMLResponse(content=html)


@router.get("/reports/download")
def download_report(
    period_days: int = Query(7, ge=1, le=90, description="Report period in days"),
) -> Response:
    """Generate and return the weekly report as a downloadable HTML file."""
    data = generate_report_data(period_days=period_days)
    html = render_report_html(data)

    import pandas as pd
    end_dt = pd.Timestamp(data.get("period", {}).get("end", ""))
    start_dt = pd.Timestamp(data.get("period", {}).get("start", ""))
    filename = f"energy_report_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.html"

    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reports/data")
def report_data(
    period_days: int = Query(7, ge=1, le=90, description="Report period in days"),
) -> dict:
    """Return raw report data as JSON (for debugging or custom integrations)."""
    return generate_report_data(period_days=period_days)
