"""Request orchestration between the proxy transport and the routing layer.

The Starlette layer parses a request into a :class:`RequestContext` and calls
:func:`handle`; this module owns everything from there: agent identity and
metrics, adapter selection, agent-group resolution, the routed-tier attempt, and
-- when routing declines -- forward-body compression, ledger recording, and the
premium passthrough. It depends on the routing layer and the upstream relay but
never on Starlette request/response plumbing beyond the response objects it
returns, so the serving policy stays testable without a live server. The
assembled-routes loader is injected so route caching remains an app-layer
concern.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from starlette.responses import JSONResponse, Response

from ai_calls_router._lib import config, logging_setup
from ai_calls_router.accounting import metrics
from ai_calls_router.proxy import passthrough, route_dispatch
from ai_calls_router.routing import (
    anthropic_oauth,
    content_sanitize,
    context_budget,
    forward_compression,
    provider_config,
)
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing.adapters import adapter_for_path

if TYPE_CHECKING:
    import httpx

    from ai_calls_router._lib.types import JsonObject, JsonValue

logger = logging.getLogger("acr.server")

RoutesLoader = Callable[[], "JsonObject"]


@dataclass(frozen=True)
class RequestContext:
    """Transport-agnostic snapshot of one incoming request.

    Carries everything the orchestrator needs to decide and serve a request
    without reaching back into the Starlette request object, so serving policy
    can be exercised directly in tests.
    """

    method: str
    path: str
    query: str
    headers: Mapping[str, str]
    body_bytes: bytes
    client: httpx.AsyncClient
    user_agent: str
    client_ip: str


def _request_summary(body: JsonObject) -> str:
    """Render a compact one-line summary of a parsed request body."""
    messages = body.get("messages")
    msg_count = len(messages) if isinstance(messages, list) else 0
    model = body.get("model")
    stream = body.get("stream")
    return f"model={model!r} stream={stream!r} messages={msg_count}"


def _parse_body_dict(body_bytes: bytes) -> JsonObject | None:
    """Parse a request body to a mapping, or None when not object-shaped."""
    try:
        payload = cast("JsonValue", json.loads(body_bytes))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _repair_inbound_body(ctx: RequestContext, body: JsonObject, *, agent: str) -> RequestContext:
    """Strip blank text blocks from the inbound body, returning a repaired context.

    Anthropic rejects blank text content blocks, so a history poisoned by one (an
    empty text block a routed model emitted before a tool_use) 400s on both the
    routed call and its byte-identical premium passthrough. Returns the original
    context untouched when nothing was blank, so valid bodies stay byte-identical
    and keep the upstream prompt cache.
    """
    cleaned = content_sanitize.clean_request_messages(body)
    if cleaned is None:
        return ctx
    logger.warning("acr: stripped blank text block(s) from %s request agent=%s", ctx.path, agent)
    return replace(ctx, body_bytes=json.dumps(cleaned, ensure_ascii=False).encode("utf-8"))


async def serve_passthrough(
    ctx: RequestContext,
    body_bytes: bytes,
    *,
    routes_loader: RoutesLoader,
    group: str | None = None,
    on_complete: passthrough.ResponseComplete | None = None,
) -> Response:
    """Relay a request to an upstream unchanged.

    When an agent group is available, passthrough targets that group's own
    upstream. Without a group, it uses the premium default. Body bytes and
    client headers are relayed without conversion or injected tier keys.

    Args:
        ctx: The request context for the incoming call.
        body_bytes: Request body to relay (may differ from ctx.body_bytes when
            the forward body was compressed).
        routes_loader: Loader for the assembled routes mapping.
        group: Optional agent group for per-agent upstream selection.
        on_complete: Optional callback invoked after the response is relayed.

    Returns:
        The streamed upstream response.
    """
    routes = routes_loader()
    upstream = (
        routing.agent_upstream(routes, group)
        if group is not None
        else config.server_settings(routes).upstream
    )
    return await passthrough.forward(
        client=ctx.client,
        upstream=upstream,
        method=ctx.method,
        path=ctx.path,
        headers=ctx.headers,
        body=body_bytes,
        query=ctx.query,
        on_complete=on_complete,
    )


def models_passthrough_group(ctx: RequestContext, *, routes_loader: RoutesLoader) -> str | None:
    """Resolve a per-agent upstream for model-list requests when identity is clear."""
    if ctx.path != "/v1/models":
        return None
    try:
        return provider_config.resolve_agent_group(
            path=ctx.path,
            headers=ctx.headers,
            routes=routes_loader(),
            adapter_default="",
        )
    except Exception as exc:
        logger.warning("acr: model-list identity lookup failed (%s); using premium default", exc)
        return None


def _premium_usage_callback(
    *,
    m: metrics.Metrics,
    request_id: str,
    session: str | None,
) -> Callable[[int, dict[str, int], float], None]:
    """Build a callback that updates metrics after premium passthrough completes."""

    def _record(status: int, usage: dict[str, int], duration: float) -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        # Teach the context-window guard the real premium size so the next turn
        # of an overflowing session skips routing instead of failing open again.
        context_budget.record_context_size(session, input_tokens, output_tokens)
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


async def _serve_premium_passthrough(
    ctx: RequestContext,
    *,
    routes: JsonObject,
    group: str,
    agent: str,
    session: str | None,
    attempt: route_dispatch.RouteAttempt,
    m: metrics.Metrics,
    routes_loader: RoutesLoader,
) -> Response:
    """Record a declined route and relay the (optionally compressed) body upstream."""
    m.incr_passthrough()
    if attempt.reason == "response_premium_guard":
        m.incr_escalated()
    elif attempt.reason in {"routing_error", "routed_fallback", "tier_unavailable"}:
        m.incr_fallback()
    logger.info(
        "outcome=passthrough %s agent=%s reason=%s tier=%s model=%r tools=%s",
        ctx.path,
        agent,
        attempt.reason,
        attempt.tier,
        attempt.model,
        attempt.tool_names,
    )
    # Drop the 1M long-context opt-in (anthropic-beta: context-1m + opus[1m] model
    # suffix) before relaying: the routed path strips it, but an OAuth subscription
    # without long-context credits 429s a passthrough turn that opts in.
    sanitized = anthropic_oauth.strip_long_context_passthrough(ctx.body_bytes, ctx.headers)
    if sanitized is not None:
        long_context_body, long_context_headers = sanitized
        logger.warning(
            "acr: stripped 1M long-context opt-in from passthrough %s agent=%s", ctx.path, agent
        )
        ctx = replace(ctx, body_bytes=long_context_body, headers=long_context_headers)
    # Compress the forwarded body for non-DeepSeek premium turns and record the
    # realized tool-output shrink (per-row column plus the aggregate tile). The
    # compressor relays byte-identical when nothing shrank, so short decision
    # turns keep the upstream prompt cache intact.
    upstream = routing.agent_upstream(routes, group)
    forward_bytes, shrink = forward_compression.compress_forward_body(
        ctx.body_bytes, request_path=ctx.path, upstream=upstream
    )
    m.add_shrink(chars_before=shrink.chars_before, chars_after=shrink.chars_after)
    m.record_request(
        method="POST",
        path=ctx.path,
        status=0,
        tier=attempt.tier,
        route="premium_guard"
        if attempt.reason
        in {"request_premium_guard", "response_premium_guard", "context_window_guard"}
        else "passthrough",
        model=attempt.model,
        user_agent=ctx.user_agent,
        client_ip=ctx.client_ip,
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
        shrink_chars_before=shrink.chars_before,
        shrink_chars_after=shrink.chars_after,
        tool_output_type=shrink.content_type_label,
    )
    return await serve_passthrough(
        ctx,
        forward_bytes,
        routes_loader=routes_loader,
        group=group,
        on_complete=_premium_usage_callback(
            m=m,
            request_id=logging_setup.current_request_id(),
            session=session,
        ),
    )


async def handle(ctx: RequestContext, *, routes_loader: RoutesLoader) -> Response:
    """Decide and serve one adapter-backed request.

    Records routing metrics as a side-effect. Returns the routed response when a
    cheap tier serves the turn, otherwise the streamed premium passthrough.

    Args:
        ctx: Transport-agnostic snapshot of the incoming request.
        routes_loader: Loader for the assembled routes mapping.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    agent = metrics.identify_agent(ctx.user_agent)
    body_dict = _parse_body_dict(ctx.body_bytes)
    session = metrics.session_fingerprint(
        body_dict.get("messages") if isinstance(body_dict, dict) else None
    )
    if isinstance(body_dict, dict):
        logger.info("inbound %s %s agent=%s", ctx.path, _request_summary(body_dict), agent)
        ctx = _repair_inbound_body(ctx, body_dict, agent=agent)
    else:
        logger.info(
            "inbound %s unparsed-body bytes=%d agent=%s", ctx.path, len(ctx.body_bytes), agent
        )

    m = metrics.get_metrics()
    m.incr_total()

    adapter = adapter_for_path(ctx.path)
    if adapter is None:
        # No adapter means no trusted agent group; keep the premium default.
        return await serve_passthrough(ctx, ctx.body_bytes, routes_loader=routes_loader)
    routes = routes_loader()
    group = provider_config.resolve_agent_group(
        path=ctx.path,
        headers=ctx.headers,
        routes=routes,
        adapter_default=adapter.default_agent_group,
    )
    if group is None:
        m.incr_error()
        logger.info("outcome=error %s agent=%s reason=unresolved_agent_identity", ctx.path, agent)
        return JSONResponse({"error": "unresolved agent identity"}, status_code=400)
    attempt = await route_dispatch.try_route(
        ctx.body_bytes,
        adapter=adapter,
        group=group,
        request_path=ctx.path,
        request_headers=ctx.headers,
        routes_loader=routes_loader,
        client=ctx.client,
        user_agent=ctx.user_agent,
        agent=agent,
        session=session,
    )
    if attempt.response is not None:
        m.incr_routed()
        context_budget.record_context_size(session, attempt.input_tokens, attempt.output_tokens)
        logger.info("outcome=routed %s agent=%s tier=%s", ctx.path, agent, attempt.tier)
        return attempt.response

    return await _serve_premium_passthrough(
        ctx,
        routes=routes,
        group=group,
        agent=agent,
        session=session,
        attempt=attempt,
        m=m,
        routes_loader=routes_loader,
    )
