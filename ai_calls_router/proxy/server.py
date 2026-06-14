"""Starlette application wiring the routing decision to the two call paths.

POST /v1/messages reads the body once and decides per request: turns that
process tool results mapped to a configured cheap tier with a resolvable key
are served by routed_call; everything else (turn openers, premium or unmapped
tools, missing keys, malformed JSON, decision errors, routed_call declining)
streams through to the premium upstream with the client's headers intact.
GET /health answers locally, GET /metrics exposes live counters, and every
other path proxies unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.resources
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_calls_router._lib import config
from ai_calls_router.accounting import metrics, savings
from ai_calls_router.proxy import passthrough
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
from ai_calls_router.routing import synthesis

logger = logging.getLogger("acr.server")

PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


async def health(request: Request) -> JSONResponse:
    """Answer the daemon health probe without touching the upstream.

    Args:
        request: Incoming health-check request.

    Returns:
        A local 200 status response.
    """
    return JSONResponse({"status": "ok"})


async def metrics_endpoint(request: Request) -> JSONResponse:
    """Return live in-memory counters for dashboard/API consumers."""
    del request
    return JSONResponse(metrics.get_metrics().snapshot())


async def dashboard(request: Request) -> Response:
    """Serve the live dashboard single-page app."""
    del request
    return _serve_dashboard()


def _serve_dashboard() -> Response:
    body = importlib.resources.read_text("ai_calls_router.proxy", "dashboard.html")
    return Response(
        body,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


async def _serve_passthrough(request: Request, body_bytes: bytes) -> Response:
    """Relay a request to the premium upstream unchanged.

    Args:
        request: Incoming client request.
        body_bytes: Raw request body, already read.

    Returns:
        The streamed upstream response.
    """
    routes = routing.load_routes()
    settings = config.server_settings(routes)
    return await passthrough.forward(
        client=request.app.state.client,
        upstream=settings.upstream,
        method=request.method,
        path=request.url.path,
        headers=request.headers,
        body=body_bytes,
        query=request.url.query,
    )


def _client_ip(request: Request) -> str:
    """Extract the client IP from the request."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _user_agent(request: Request) -> str:
    """Extract the User-Agent header."""
    return request.headers.get("user-agent", "")


async def _try_route(
    body_bytes: bytes,
    *,
    user_agent: str = "",
    agent: str = "",
    session: str | None = None,
) -> Response | None:
    """Attempt to serve a /v1/messages request on a cheap tier.

    Any error anywhere in the decision or the routed call resolves to None
    so the caller replays the turn on premium passthrough (invariant 3:
    routing never breaks a turn).

    Args:
        body_bytes: Raw request body bytes.

    Returns:
        The routed response, or None when the turn must pass through.
    """
    try:
        body = json.loads(body_bytes)
        if not isinstance(body, dict):
            return None
        names = routing.pending_tool_names(body)
        if not names:
            return None
        routes = routing.load_routes()
        tier = routing.tier_for_tools(names, routes)
        if tier == "premium":
            return None
        tiers = routes.get("tiers") or {}
        tier_cfg = tiers.get(tier)
        if not isinstance(tier_cfg, dict):
            return None
        settings_cfg = routes.get("settings") or {}
        api_key = routing.resolve_api_key(tier_cfg, settings_cfg)
        if not api_key:
            logger.info("acr: tier=%s has no API key; passing through", tier)
            return None
        savings.register_tier_prices(routes)
        result = await routed_call.routed_call(
            body=body,
            tier_name=tier,
            tier_cfg=tier_cfg,
            api_key=api_key,
            settings=settings_cfg,
            tool_names=names,
            user_agent=user_agent,
            agent=agent,
            session_id=session or "",
        )
        if result is None:
            return None
        if body.get("stream"):
            return Response(
                synthesis.synthesize_sse(result.body),
                media_type="text/event-stream",
            )
        return JSONResponse(result.body)
    except Exception as exc:
        logger.warning("acr: routing decision failed (%s); passing through", exc)
        return None


async def messages(request: Request) -> Response:
    """Decide and serve one /v1/messages request.

    Records routing metrics as a side-effect.

    Args:
        request: Incoming Anthropic Messages API request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    body_bytes = await request.body()
    agent = metrics.identify_agent(_user_agent(request))
    try:
        payload: Any = json.loads(body_bytes)
        body_dict: Any = payload if isinstance(payload, dict) else None
    except Exception:
        body_dict = None
    session = metrics.session_fingerprint(
        body_dict.get("messages") if isinstance(body_dict, dict) else None
    )

    m = metrics.get_metrics()
    m.incr_total()

    routed = await _try_route(
        body_bytes,
        user_agent=_user_agent(request),
        agent=agent,
        session=session,
    )
    if routed is not None:
        m.incr_routed()
        return routed

    m.incr_passthrough()
    m.record_request(
        method="POST",
        path="/v1/messages",
        status=0,
        tier="premium",
        route="passthrough",
        model="",
        user_agent=_user_agent(request),
        client_ip=_client_ip(request),
        tool_names=[],
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_creation=0,
        duration=0,
        agent=agent,
        session_id=session or "",
    )
    return await _serve_passthrough(request, body_bytes)


async def proxy(request: Request) -> Response:
    """Proxy any non-messages endpoint to the premium upstream.

    Args:
        request: Incoming client request for any other path.

    Returns:
        The streamed upstream response.
    """
    body_bytes = await request.body()
    return await _serve_passthrough(request, body_bytes)


@contextlib.asynccontextmanager
async def _lifespan(
    app: Starlette, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncIterator[None]:
    """Own the shared upstream HTTP client for the app's lifetime.

    Args:
        app: The Starlette application being started.
        transport: Optional httpx transport override (tests inject mocks).

    Yields:
        None while the application serves requests.
    """
    mtr = metrics.get_metrics()
    mtr.bootstrap(ledger_path=config.ledger_path(), max_recent=100)
    app.state.client = httpx.AsyncClient(transport=transport, timeout=passthrough.UPSTREAM_TIMEOUT)
    try:
        yield
    finally:
        await app.state.client.aclose()


def create_app(transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """Build the proxy application.

    Args:
        transport: Optional httpx transport override (tests inject mocks).

    Returns:
        A Starlette app serving /health, /metrics, /v1/messages, and a catch-all proxy.
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    lifespan: Any = functools.partial(_lifespan, transport=transport)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/metrics", metrics_endpoint, methods=["GET"]),
            Route("/dashboard", dashboard, methods=["GET"]),
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/{path:path}", proxy, methods=PROXY_METHODS),
        ],
        lifespan=lifespan,
    )
