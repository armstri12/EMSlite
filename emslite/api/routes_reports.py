"""Weekly report API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, Response

from ..report import (
    generate_email_summary_data,
    generate_report_data,
    generate_ytd_report_data,
    render_email_summary_html,
    render_report_html,
    render_ytd_report_html,
)

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


# ---------------------------------------------------------------------------
# YTD Cost Allocation Report
# ---------------------------------------------------------------------------


@router.get("/reports/ytd/preview")
def preview_ytd_report(
    year: int | None = Query(None, description="Report year (defaults to current year)"),
) -> HTMLResponse:
    """Generate and return the YTD cost allocation report as HTML for browser preview."""
    data = generate_ytd_report_data(year=year)
    html = render_ytd_report_html(data)
    return HTMLResponse(content=html)


@router.get("/reports/ytd/download")
def download_ytd_report(
    year: int | None = Query(None, description="Report year (defaults to current year)"),
) -> Response:
    """Generate and return the YTD cost allocation report as a downloadable HTML file."""
    data = generate_ytd_report_data(year=year)
    html = render_ytd_report_html(data)
    report_year = data.get("period", {}).get("year", "unknown")
    filename = f"ytd_cost_allocation_{report_year}.html"
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/reports/ytd/data")
def ytd_report_data(
    year: int | None = Query(None, description="Report year (defaults to current year)"),
) -> dict:
    """Return raw YTD report data as JSON."""
    return generate_ytd_report_data(year=year)


# ---------------------------------------------------------------------------
# Email Summary Report
# ---------------------------------------------------------------------------


@router.get("/reports/email-summary")
def email_summary_report(
    panels: str | None = Query(None, description="Comma-separated panel IDs; omit for all panels"),
    start: str | None = Query(None, description="Start date (ISO, e.g. 2024-01-01)"),
    end: str | None = Query(None, description="End date (ISO, e.g. 2024-01-07)"),
    download: bool = Query(False, description="If true, return as downloadable attachment"),
) -> Response:
    """Generate an Outlook-ready email summary HTML for selected panels."""
    panel_list = [p.strip() for p in panels.split(",") if p.strip()] if panels else None
    data = generate_email_summary_data(panels=panel_list, start=start, end=end)
    html = render_email_summary_html(data)

    if download:
        period = data.get("period", {})
        try:
            import pandas as pd
            start_dt = pd.Timestamp(period.get("start", ""))
            end_dt = pd.Timestamp(period.get("end", ""))
            filename = f"email_summary_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.html"
        except Exception:
            filename = "email_summary.html"
        return Response(
            content=html,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return HTMLResponse(content=html)
