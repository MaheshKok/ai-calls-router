"""Starlette application wiring the routing decision to the two call paths.

POST /v1/messages reads the body once and decides per request: turns that
process tool results mapped to a configured cheap tier with a resolvable key
are served by routed_call; everything else (turn openers, premium or unmapped
tools, missing keys, malformed JSON, decision errors, routed_call declining)
streams through to the premium upstream with the client's headers intact.
GET /health answers locally and every other path proxies unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_calls_router import config, passthrough, routed_call, routing, savings

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
        request.app.state.client,
        settings.upstream,
        request.method,
        request.url.path,
        request.headers,
        body_bytes,
        query=request.url.query,
    )


async def _try_route(body_bytes: bytes) -> Response | None:
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
            body, tier, tier_cfg, api_key, settings_cfg
        )
        if result is None:
            return None
        if body.get("stream"):
            return Response(
                routed_call.synthesize_sse(result.body),
                media_type="text/event-stream",
            )
        return JSONResponse(result.body)
    except Exception as exc:
        logger.warning("acr: routing decision failed (%s); passing through", exc)
        return None


async def messages(request: Request) -> Response:
    """Decide and serve one /v1/messages request.

    Args:
        request: Incoming Anthropic Messages API request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    body_bytes = await request.body()
    routed = await _try_route(body_bytes)
    if routed is not None:
        return routed
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
    app.state.client = httpx.AsyncClient(
        transport=transport, timeout=passthrough.UPSTREAM_TIMEOUT
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


def create_app(transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """Build the proxy application.

    Args:
        transport: Optional httpx transport override (tests inject mocks).

    Returns:
        A Starlette app serving /health, /v1/messages, and a catch-all proxy.
    """
    lifespan: Any = functools.partial(_lifespan, transport=transport)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/{path:path}", proxy, methods=PROXY_METHODS),
        ],
        lifespan=lifespan,
    )
