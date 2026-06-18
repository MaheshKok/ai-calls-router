"""Dashboard and metrics HTTP handlers for the proxy.

Keeping these handlers outside server.py leaves the app module focused on route
wiring and lifecycle. The functions are pure Starlette endpoints over the live
metrics singleton and packaged dashboard asset.
"""

from __future__ import annotations

import importlib.resources
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse, Response

from ai_calls_router.accounting import metrics

if TYPE_CHECKING:
    from starlette.requests import Request


async def metrics_endpoint(request: Request) -> JSONResponse:
    """Return live in-memory counters for dashboard/API consumers."""
    del request
    return JSONResponse(metrics.get_metrics().snapshot())


async def dashboard(request: Request) -> Response:
    """Serve the live dashboard single-page app."""
    del request
    return _serve_dashboard()


def _serve_dashboard() -> Response:
    """Return the packaged dashboard HTML with no-cache headers."""
    body = importlib.resources.read_text("ai_calls_router.proxy", "dashboard.html")
    return Response(
        body,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
