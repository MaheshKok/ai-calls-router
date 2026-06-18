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
import threading
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from ai_calls_router._lib import config, logging_setup
from ai_calls_router.accounting import metrics, savings
from ai_calls_router.ops import bootstrap
from ai_calls_router.proxy import passthrough, websocket_passthrough
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
from ai_calls_router.routing import provider_config
from ai_calls_router.routing.adapters import adapter_for_path
from ai_calls_router.routing.adapters.base import KNOWN_GROUPS

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue
    from ai_calls_router.routing.adapters.base import ClientAdapter

logger = logging.getLogger("acr.server")

LOG_REVISION = "2026-06-15-premium-guard-v2"


@dataclass(frozen=True)
class _RouteAttempt:
    response: Response | None = None
    tier: str = "premium"
    reason: str = "passthrough"
    model: str = ""
    tool_names: list[str] = field(default_factory=lambda: [])


@dataclass
class _RoutesCache:
    signature: tuple[tuple[str, float], ...] | None = None
    routes: JsonObject | None = None


_ROUTES_CACHE_LOCK = threading.Lock()
_ROUTES_CACHE = _RoutesCache()


def _request_summary(body: JsonObject) -> str:
    messages = body.get("messages")
    msg_count = len(messages) if isinstance(messages, list) else 0
    model = body.get("model")
    stream = body.get("stream")
    return f"model={model!r} stream={stream!r} messages={msg_count}"


PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def _mtime_or_zero(path: Path) -> float:
    """Return a config file mtime, treating missing files as absent."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _assembled_routes_signature() -> tuple[tuple[str, float], ...]:
    """Build the mtime signature for the global and provider config files."""
    paths = [config.config_path()]
    paths.extend(config.provider_config_path(group) for group in sorted(KNOWN_GROUPS))
    return tuple((str(path), _mtime_or_zero(path)) for path in paths)


def _assemble_routes_fail_open(
    base: JsonObject, provider_files: dict[str, JsonObject]
) -> JsonObject:
    """Assemble routes, dropping invalid provider payloads one at a time."""
    remaining = dict(provider_files)
    while True:
        try:
            return provider_config.assemble_routes(base, provider_files=remaining)
        except provider_config.ProviderConfigError as exc:
            logger.warning(
                "acr: provider config assembly failed (%s); skipping provider file",
                exc,
                exc_info=True,
            )
            if exc.group is not None and exc.group in remaining:
                remaining = {
                    group: payload for group, payload in remaining.items() if group != exc.group
                }
            elif remaining:
                remaining = {}
            else:
                return provider_config.assemble_routes(base, provider_files={})


def _load_assembled_routes() -> JsonObject:
    """Load the canonical routes dict assembled from global and provider YAML."""
    signature = _assembled_routes_signature()
    with _ROUTES_CACHE_LOCK:
        if _ROUTES_CACHE.signature == signature and _ROUTES_CACHE.routes is not None:
            return _ROUTES_CACHE.routes

    assembled = _assemble_routes_fail_open(
        routing.load_routes(),
        provider_config.load_provider_files(),
    )
    with _ROUTES_CACHE_LOCK:
        _ROUTES_CACHE.signature = signature
        _ROUTES_CACHE.routes = assembled
    return assembled


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


async def _serve_passthrough(
    request: Request,
    body_bytes: bytes,
    *,
    group: str | None = None,
    on_complete: passthrough.ResponseComplete | None = None,
) -> Response:
    """Relay a request to an upstream unchanged.

    When an agent group is available, passthrough targets that group's own
    upstream. Without a group, it uses the premium default. Body bytes and
    client headers are relayed without conversion or injected tier keys.

    Args:
        request: Incoming client request.
        body_bytes: Raw request body, already read.
        group: Optional agent group for per-agent upstream selection.
        on_complete: Optional callback invoked after the premium response is
            fully relayed.

    Returns:
        The streamed upstream response.
    """
    routes = _load_assembled_routes()
    upstream = (
        routing.agent_upstream(routes, group)
        if group is not None
        else config.server_settings(routes).upstream
    )
    return await passthrough.forward(
        client=request.app.state.client,
        upstream=upstream,
        method=request.method,
        path=request.url.path,
        headers=request.headers,
        body=body_bytes,
        query=request.url.query,
        on_complete=on_complete,
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


def _premium_usage_callback(
    *,
    m: metrics.Metrics,
    request_id: str,
) -> Callable[[int, dict[str, int], float], None]:
    """Build a callback that updates metrics after premium passthrough completes."""

    def _record(status: int, usage: dict[str, int], duration: float) -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        if input_tokens or output_tokens or cache_read or cache_creation:
            m.add_premium_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_creation=cache_creation,
            )
        m.update_request_usage(
            request_id=request_id,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_creation=cache_creation,
            duration=duration,
        )

    return _record


def _resolve_tier_config(
    names: list[str],
    *,
    group: str,
) -> tuple[str, JsonObject | None, str | None, JsonObject]:
    """Resolve the tier, its config, and API key from loaded routes.

    Returns (tier_name, tier_cfg, api_key, raw_routes). When tier_cfg or
    api_key is None the caller must pass through to premium.
    """
    routes = _load_assembled_routes()
    tier = routing.tier_for_tools(names, routes, group=group)
    if tier == "premium":
        return "premium", None, None, routes
    tiers = routes.get("tiers")
    tier_cfg = tiers.get(tier) if isinstance(tiers, dict) else None
    if not isinstance(tier_cfg, dict):
        return tier, None, None, routes
    settings_value = routes.get("settings")
    settings_cfg = settings_value if isinstance(settings_value, dict) else {}
    api_key = routing.resolve_api_key(tier_cfg, settings_cfg)
    if not api_key:
        logger.info("acr: tier=%s has no API key; passing through", tier)
        return tier, None, None, routes
    return tier, tier_cfg, api_key, routes


def _premium_guard_attempt(
    *, reason: str, requested_model: str, names: list[str], tier: str
) -> _RouteAttempt:
    """Log a premium-guard passthrough decision and build its route attempt.

    Args:
        reason: Passthrough reason recorded on the attempt and in the log line.
        requested_model: Model the client asked for.
        names: Pending tool-result names that triggered the decision.
        tier: Resolved tier name.

    Returns:
        A passthrough route attempt carrying the reason and request context.
    """
    logger.info(
        "acr: premium guard decision reason=%s model=%r tools=%s tier=%s",
        reason,
        requested_model,
        names,
        tier,
    )
    return _RouteAttempt(tier=tier, reason=reason, model=requested_model, tool_names=names)


def _routed_fallback_attempt(
    *, response_guard_tools: list[str], requested_model: str, names: list[str], tier: str
) -> _RouteAttempt:
    """Build the passthrough attempt for a routed call that returned no body.

    Args:
        response_guard_tools: Tool names the tier flagged for premium handling
            mid-response; empty means an ordinary routed fallback.
        requested_model: Model the client asked for.
        names: Pending tool-result names for this turn.
        tier: Resolved tier name.

    Returns:
        A passthrough route attempt distinguishing a response-side premium
        guard from a generic routed fallback.
    """
    reason = "response_premium_guard" if response_guard_tools else "routed_fallback"
    if response_guard_tools:
        logger.info(
            "acr: premium guard decision reason=%s model=%r tools=%s response_tools=%s tier=%s",
            reason,
            requested_model,
            names,
            response_guard_tools,
            tier,
        )
    return _RouteAttempt(tier=tier, reason=reason, model=requested_model, tool_names=names)


async def _try_route(
    body_bytes: bytes,
    *,
    adapter: ClientAdapter,
    group: str,
    request_path: str,
    user_agent: str = "",
    agent: str = "",
    session: str | None = None,
) -> _RouteAttempt:
    """Attempt to serve a /v1/messages request on a cheap tier.

    Any error anywhere in the decision or the routed call resolves to None
    so the caller replays the turn on premium passthrough (invariant 3:
    routing never breaks a turn).

    Args:
        adapter: Client adapter for the request path.
        group: Agent group to use for routing.
        request_path: Client-facing request path for logging and metrics.
        body_bytes: Raw request body bytes.
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label.
        session: Session fingerprint hex string.

    Returns:
        A route attempt carrying either the routed response or the passthrough reason.
    """
    try:
        body = cast("JsonValue", json.loads(body_bytes))
        if not isinstance(body, dict):
            return _RouteAttempt(reason="non_object_body")
        anthropic_body = adapter.to_anthropic_request(body)
        requested_model = str(anthropic_body.get("model") or "")
        streaming = _wants_stream(body, anthropic_body)
        names = adapter.extract_pending_tools(body)
        if not names:
            logger.debug("no pending tool results; passing through")
            return _RouteAttempt(reason="no_pending_tools", model=requested_model)
        logger.debug("pending tools=%s", names)
        tier, tier_cfg, api_key, routes = _resolve_tier_config(names, group=group)
        logger.debug("resolved tier=%s routable=%s", tier, tier_cfg is not None)
        if tier_cfg is None:
            reason = "request_premium_guard" if tier == "premium" else "tier_unavailable"
            return _premium_guard_attempt(
                reason=reason, requested_model=requested_model, names=names, tier=tier
            )
        if api_key is None:
            return _premium_guard_attempt(
                reason="tier_unavailable", requested_model=requested_model, names=names, tier=tier
            )
        savings.register_tier_prices(routes)
        settings_value = routes.get("settings")
        settings_cfg = settings_value if isinstance(settings_value, dict) else {}
        premium_tools = routing.agent_premium_tools(routes, group)
        response_guard_tools: list[str] = []

        def _mark_response_guard(tool_names: list[str]) -> None:
            response_guard_tools.extend(tool_names)

        result = await routed_call.routed_call(
            body=anthropic_body,
            tier_name=tier,
            tier_cfg=tier_cfg,
            api_key=api_key,
            settings=settings_cfg,
            tool_names=names,
            premium_tools=premium_tools,
            request_path=request_path,
            user_agent=user_agent,
            agent=group,
            session_id=session or "",
            on_premium_guard=_mark_response_guard,
        )
        if result is None:
            return _routed_fallback_attempt(
                response_guard_tools=response_guard_tools,
                requested_model=requested_model,
                names=names,
                tier=tier,
            )
        client_body = adapter.to_client_response(result.body)
        if streaming:
            return _RouteAttempt(
                response=Response(
                    b"".join(adapter.to_client_sse(result.body)),
                    media_type="text/event-stream",
                ),
                tier=tier,
                reason="routed",
                model=requested_model,
                tool_names=names,
            )
        return _RouteAttempt(
            response=JSONResponse(client_body),
            tier=tier,
            reason="routed",
            model=requested_model,
            tool_names=names,
        )
    except Exception as exc:
        logger.warning("acr: routing decision failed (%s); passing through", exc, exc_info=True)
        return _RouteAttempt(reason="routing_error")


def _wants_stream(client_body: JsonObject, anthropic_body: JsonObject) -> bool:
    """Return whether the client requested a streaming response."""
    stream = client_body.get("stream")
    if isinstance(stream, bool):
        return stream
    return bool(anthropic_body.get("stream"))


async def messages(request: Request) -> Response:
    """Decide and serve one /v1/messages request.

    Records routing metrics as a side-effect.

    Args:
        request: Incoming Anthropic Messages API request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    with logging_setup.request_context():
        return await _handle_routed_request(request)


async def chat_completions(request: Request) -> Response:
    """Decide and serve one /v1/chat/completions request.

    Args:
        request: Incoming OpenAI Chat Completions request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    with logging_setup.request_context():
        return await _handle_routed_request(request)


async def responses(request: Request) -> Response:
    """Decide and serve one /v1/responses request.

    Args:
        request: Incoming OpenAI Responses request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    with logging_setup.request_context():
        return await _handle_routed_request(request)


async def responses_ws(websocket: WebSocket) -> None:
    """Relay Codex ChatGPT-auth Responses WebSockets."""
    await websocket_passthrough.forward_codex_chatgpt(
        websocket, route_first_frame=_try_route_ws_response_create
    )


async def responses_ws_sub(websocket: WebSocket) -> None:
    """Relay Codex ChatGPT-auth Responses WebSocket subpaths."""
    raw_sub_path = websocket.path_params.get("sub_path", "")
    sub_path = raw_sub_path if isinstance(raw_sub_path, str) else ""
    await websocket_passthrough.forward_codex_chatgpt(
        websocket, sub_path=sub_path, route_first_frame=_try_route_ws_response_create
    )


async def _try_route_ws_response_create(
    first_msg_raw: str, headers: Mapping[str, str]
) -> list[str] | None:
    """Try to serve one Codex WS `response.create` through the routed core."""
    try:
        body = websocket_passthrough.response_create_to_http_body(first_msg_raw)
        if body is None:
            return None
        adapter = adapter_for_path("/v1/responses")
        if adapter is None:
            return None
        routes = _load_assembled_routes()
        group = provider_config.resolve_agent_group(
            path="/v1/responses",
            headers=headers,
            routes=routes,
            adapter_default=adapter.default_agent_group,
        )
        if group is None:
            return None
        body_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        user_agent = headers.get("user-agent", "")
        attempt = await _try_route(
            body_bytes,
            adapter=adapter,
            group=group,
            request_path="/v1/responses",
            user_agent=user_agent,
            agent=metrics.identify_agent(user_agent),
            session=None,
        )
        if attempt.response is None:
            return None
        return websocket_passthrough.sse_to_ws_messages(bytes(attempt.response.body))
    except Exception as exc:
        logger.warning("acr: websocket routing decision failed (%s); passing through", exc)
        return None


async def _handle_routed_request(request: Request) -> Response:
    """Serve one adapter-backed request inside an active request context."""
    path = request.url.path
    body_bytes = await request.body()
    agent = metrics.identify_agent(_user_agent(request))
    try:
        payload = cast("JsonValue", json.loads(body_bytes))
        body_dict = payload if isinstance(payload, dict) else None
    except Exception:
        body_dict = None
    session = metrics.session_fingerprint(
        body_dict.get("messages") if isinstance(body_dict, dict) else None
    )
    if isinstance(body_dict, dict):
        logger.info("inbound %s %s agent=%s", path, _request_summary(body_dict), agent)
    else:
        logger.info("inbound %s unparsed-body bytes=%d agent=%s", path, len(body_bytes), agent)

    m = metrics.get_metrics()
    m.incr_total()

    adapter = adapter_for_path(path)
    if adapter is None:
        # No adapter means no trusted agent group; keep the premium default.
        return await _serve_passthrough(request, body_bytes)
    routes = _load_assembled_routes()
    group = provider_config.resolve_agent_group(
        path=path,
        headers=request.headers,
        routes=routes,
        adapter_default=adapter.default_agent_group,
    )
    if group is None:
        return JSONResponse({"error": "unresolved agent identity"}, status_code=400)
    attempt = await _try_route(
        body_bytes,
        adapter=adapter,
        group=group,
        request_path=path,
        user_agent=_user_agent(request),
        agent=agent,
        session=session,
    )
    if attempt.response is not None:
        m.incr_routed()
        logger.info("outcome=routed %s agent=%s tier=%s", path, agent, attempt.tier)
        return attempt.response

    m.incr_passthrough()
    if attempt.reason == "response_premium_guard":
        m.incr_escalated()
    elif attempt.reason in {"routing_error", "routed_fallback", "tier_unavailable"}:
        m.incr_fallback()
    logger.info(
        "outcome=passthrough %s agent=%s reason=%s tier=%s model=%r tools=%s",
        path,
        agent,
        attempt.reason,
        attempt.tier,
        attempt.model,
        attempt.tool_names,
    )
    m.record_request(
        method="POST",
        path=path,
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
        request_id=logging_setup.current_request_id(),
    )
    return await _serve_passthrough(
        request,
        body_bytes,
        group=group,
        on_complete=_premium_usage_callback(
            m=m,
            request_id=logging_setup.current_request_id(),
        ),
    )


async def proxy(request: Request) -> Response:
    """Proxy any non-messages endpoint to the premium upstream.

    Args:
        request: Incoming client request for any other path.

    Returns:
        The streamed upstream response.
    """
    body_bytes = await request.body()
    # Catch-all paths have no adapter-derived agent group; keep the premium default.
    return await _serve_passthrough(request, body_bytes)


@contextlib.asynccontextmanager
async def _lifespan(
    app: Starlette, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncGenerator[None, None]:
    """Own the shared upstream HTTP client for the app's lifetime.

    Args:
        app: The Starlette application being started.
        transport: Optional httpx transport override (tests inject mocks).

    Yields:
        None while the application serves requests.
    """
    try:
        bootstrap.ensure_provider_configs()
    except Exception as exc:
        logger.warning("acr: provider config bootstrap failed (%s); continuing", exc, exc_info=True)
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
    lifespan = functools.partial(_lifespan, transport=transport)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/metrics", metrics_endpoint, methods=["GET"]),
            Route("/dashboard", dashboard, methods=["GET"]),
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/responses", responses, methods=["POST"]),
            WebSocketRoute("/v1/responses", responses_ws),
            WebSocketRoute("/v1/responses/{sub_path:path}", responses_ws_sub),
            WebSocketRoute("/v1/codex/responses", responses_ws),
            WebSocketRoute("/v1/codex/responses/{sub_path:path}", responses_ws_sub),
            Route("/{path:path}", proxy, methods=PROXY_METHODS),
        ],
        lifespan=lifespan,
    )
