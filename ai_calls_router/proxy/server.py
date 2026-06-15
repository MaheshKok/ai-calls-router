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
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_calls_router._lib import config, logging_setup
from ai_calls_router.accounting import metrics, savings
from ai_calls_router.proxy import passthrough
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
from ai_calls_router.routing import synthesis

logger = logging.getLogger("acr.server")

LOG_REVISION = "2026-06-15-premium-guard-v2"


@dataclass(frozen=True)
class _RouteAttempt:
    response: Response | None = None
    tier: str = "premium"
    reason: str = "passthrough"
    model: str = ""
    tool_names: list[str] = field(default_factory=list)


def _request_summary(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    msg_count = len(messages) if isinstance(messages, list) else 0
    model = body.get("model")
    stream = body.get("stream")
    return f"model={model!r} stream={stream!r} messages={msg_count}"


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


def _resolve_tier_config(
    names: list[str],
) -> tuple[str, dict[str, Any] | None, str | None, dict[str, Any]]:
    """Resolve the tier, its config, and API key from loaded routes.

    Returns (tier_name, tier_cfg, api_key, raw_routes). When tier_cfg or
    api_key is None the caller must pass through to premium.
    """
    routes = routing.load_routes()
    tier = routing.tier_for_tools(names, routes)
    if tier == "premium":
        return "premium", None, None, routes
    tier_cfg = (routes.get("tiers") or {}).get(tier)
    if not isinstance(tier_cfg, dict):
        return tier, None, None, routes
    settings_cfg = routes.get("settings") or {}
    api_key = routing.resolve_api_key(tier_cfg, settings_cfg)
    if not api_key:
        logger.info("acr: tier=%s has no API key; passing through", tier)
        return tier, None, None, routes
    return tier, tier_cfg, api_key, routes


async def _try_route(
    body_bytes: bytes,
    *,
    user_agent: str = "",
    agent: str = "",
    session: str | None = None,
) -> _RouteAttempt:
    """Attempt to serve a /v1/messages request on a cheap tier.

    Any error anywhere in the decision or the routed call resolves to None
    so the caller replays the turn on premium passthrough (invariant 3:
    routing never breaks a turn).

    Args:
        body_bytes: Raw request body bytes.
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label.
        session: Session fingerprint hex string.

    Returns:
        A route attempt carrying either the routed response or the passthrough reason.
    """
    try:
        body = json.loads(body_bytes)
        if not isinstance(body, dict):
            return _RouteAttempt(reason="non_object_body")
        requested_model = str(body.get("model") or "")
        names = routing.pending_tool_names(body)
        if not names:
            logger.debug("no pending tool results; passing through")
            return _RouteAttempt(reason="no_pending_tools", model=requested_model)
        logger.debug("pending tools=%s", names)
        tier, tier_cfg, api_key, routes = _resolve_tier_config(names)
        logger.debug("resolved tier=%s routable=%s", tier, tier_cfg is not None)
        if tier_cfg is None:
            reason = "request_premium_guard" if tier == "premium" else "tier_unavailable"
            logger.info(
                "acr: premium guard decision reason=%s model=%r tools=%s tier=%s",
                reason,
                requested_model,
                names,
                tier,
            )
            return _RouteAttempt(
                tier=tier,
                reason=reason,
                model=requested_model,
                tool_names=names,
            )
        if api_key is None:
            logger.info(
                "acr: premium guard decision reason=tier_unavailable model=%r tools=%s tier=%s",
                requested_model,
                names,
                tier,
            )
            return _RouteAttempt(
                tier=tier,
                reason="tier_unavailable",
                model=requested_model,
                tool_names=names,
            )
        savings.register_tier_prices(routes)
        settings_cfg = routes.get("settings") or {}
        response_guard_tools: list[str] = []

        def _mark_response_guard(tool_names: list[str]) -> None:
            response_guard_tools.extend(tool_names)

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
            on_premium_guard=_mark_response_guard,
        )
        if result is None:
            reason = "response_premium_guard" if response_guard_tools else "routed_fallback"
            if response_guard_tools:
                logger.info(
                    "acr: premium guard decision reason=%s model=%r tools=%s "
                    "response_tools=%s tier=%s",
                    reason,
                    requested_model,
                    names,
                    response_guard_tools,
                    tier,
                )
            return _RouteAttempt(
                tier=tier,
                reason=reason,
                model=requested_model,
                tool_names=names,
            )
        if body.get("stream"):
            return _RouteAttempt(
                response=Response(
                    synthesis.synthesize_sse(result.body),
                    media_type="text/event-stream",
                ),
                tier=tier,
                reason="routed",
                model=requested_model,
                tool_names=names,
            )
        return _RouteAttempt(
            response=JSONResponse(result.body),
            tier=tier,
            reason="routed",
            model=requested_model,
            tool_names=names,
        )
    except Exception as exc:
        logger.warning("acr: routing decision failed (%s); passing through", exc, exc_info=True)
        return _RouteAttempt(reason="routing_error")


async def messages(request: Request) -> Response:
    """Decide and serve one /v1/messages request.

    Records routing metrics as a side-effect.

    Args:
        request: Incoming Anthropic Messages API request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    with logging_setup.request_context():
        return await _handle_messages(request)


async def _handle_messages(request: Request) -> Response:
    """Serve one /v1/messages request inside an active request context."""
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
    if isinstance(body_dict, dict):
        logger.info("inbound /v1/messages %s agent=%s", _request_summary(body_dict), agent)
    else:
        logger.info("inbound /v1/messages unparsed-body bytes=%d agent=%s", len(body_bytes), agent)

    m = metrics.get_metrics()
    m.incr_total()

    attempt = await _try_route(
        body_bytes,
        user_agent=_user_agent(request),
        agent=agent,
        session=session,
    )
    if attempt.response is not None:
        m.incr_routed()
        logger.info("outcome=routed /v1/messages agent=%s tier=%s", agent, attempt.tier)
        return attempt.response

    m.incr_passthrough()
    if attempt.reason == "response_premium_guard":
        m.incr_escalated()
    elif attempt.reason in {"routing_error", "routed_fallback", "tier_unavailable"}:
        m.incr_fallback()
    logger.info(
        "outcome=passthrough /v1/messages agent=%s reason=%s tier=%s model=%r tools=%s",
        agent,
        attempt.reason,
        attempt.tier,
        attempt.model,
        attempt.tool_names,
    )
    m.record_request(
        method="POST",
        path="/v1/messages",
        status=0,
        tier=attempt.tier,
        route="premium_guard"
        if attempt.reason in {"request_premium_guard", "response_premium_guard"}
        else "passthrough",
        model=attempt.model,
        user_agent=_user_agent(request),
        client_ip=_client_ip(request),
        tool_names=attempt.tool_names,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_creation=0,
        duration=0,
        premium_model=attempt.model,
        agent=agent,
        session_id=session or "",
        decision_reason=attempt.reason,
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
    logging_setup.setup_logging()
    logger.info(
        "startup revision=%s pid=%s cwd=%s log=%s",
        LOG_REVISION,
        os.getpid(),
        Path.cwd(),
        config.log_path(),
    )
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
