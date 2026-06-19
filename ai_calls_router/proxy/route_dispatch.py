"""Routing orchestration for adapter-backed proxy requests.

This module sits above the routing engine and adapters but below the Starlette
route table. It owns tier resolution, routed-call dispatch, and premium-guard
attempt construction without importing the app module.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from starlette.responses import JSONResponse, Response

from ai_calls_router._lib import logging_setup
from ai_calls_router.accounting import metrics, savings, shrink_stats
from ai_calls_router.proxy import websocket_passthrough
from ai_calls_router.routing import codex_direct, provider_config
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
from ai_calls_router.routing.adapters import adapter_for_path
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)
from ai_calls_router.routing.synthesis_responses import synthesize_response_object_sse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

    from ai_calls_router._lib.types import JsonObject, JsonValue
    from ai_calls_router.routing.adapters.base import ClientAdapter

logger = logging.getLogger("acr.server")


@dataclass(frozen=True)
class RouteAttempt:
    """Result of trying to route one request."""

    response: Response | None = None
    tier: str = "premium"
    reason: str = "passthrough"
    model: str = ""
    premium_model: str = ""
    tool_names: list[str] = field(default_factory=lambda: [])
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    duration: float = 0.0
    shrink: shrink_stats.ShrinkStats | None = None


def record_ws_route_attempt(headers: Mapping[str, str], attempt: RouteAttempt) -> None:
    """Record one WebSocket response.create route attempt for dashboard visibility."""
    try:
        m = metrics.get_metrics()
        shrink = attempt.shrink or shrink_stats.compute_shrink(path="none", before={}, after={})
        user_agent = headers.get("user-agent", "")
        routed = attempt.response is not None
        m.incr_total()
        if routed:
            m.incr_routed()
            route = "routed"
            status = 101
        else:
            m.incr_passthrough()
            if attempt.reason == "response_premium_guard":
                m.incr_escalated()
            elif attempt.reason in {"routing_error", "routed_fallback", "tier_unavailable"}:
                m.incr_fallback()
            route = (
                "premium_guard"
                if attempt.reason in {"request_premium_guard", "response_premium_guard"}
                else "passthrough"
            )
            status = 0
        m.record_request(
            method="WS",
            path="/v1/responses",
            status=status,
            tier=attempt.tier,
            route=route,
            model=attempt.model,
            user_agent=user_agent,
            client_ip="",
            tool_names=attempt.tool_names,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            cache_read=attempt.cache_read_tokens,
            cache_creation=attempt.cache_creation_tokens,
            duration=attempt.duration,
            shrink_chars_before=shrink.chars_before,
            shrink_chars_after=shrink.chars_after,
            premium_model="" if routed else attempt.model,
            agent=metrics.identify_agent(user_agent),
            session_id="",
            decision_reason=attempt.reason,
            request_id=logging_setup.current_request_id(),
        )
    except Exception as exc:
        logger.warning("acr: websocket metrics update failed (%s); continuing", exc, exc_info=True)


def resolve_tier_config(
    names: list[str],
    *,
    group: str,
    routes_loader: Callable[[], JsonObject],
) -> tuple[str, JsonObject | None, routing.TierCredential | None, JsonObject]:
    """Resolve the tier, its config, and API key from loaded routes."""
    routes = routes_loader()
    tier = routing.tier_for_tools(names, routes, group=group)
    if tier == "premium":
        return "premium", None, None, routes
    tiers = routes.get("tiers")
    tier_cfg = tiers.get(tier) if isinstance(tiers, dict) else None
    if not isinstance(tier_cfg, dict):
        return tier, None, None, routes
    settings_value = routes.get("settings")
    settings_cfg: JsonObject = settings_value if isinstance(settings_value, dict) else {}
    credential = routing.resolve_tier_credential(tier_cfg, settings_cfg)
    if credential is None:
        logger.info("acr: tier=%s has no API key; passing through", tier)
        return tier, None, None, routes
    return tier, tier_cfg, credential, routes


def premium_guard_attempt(
    *, reason: str, requested_model: str, names: list[str], tier: str
) -> RouteAttempt:
    """Log a premium-guard passthrough decision and build its route attempt."""
    logger.info(
        "premium guard reason=%s model=%r tools=%s tier=%s",
        reason,
        requested_model,
        names,
        tier,
    )
    return RouteAttempt(
        reason=reason,
        model=requested_model,
        tool_names=names,
        tier=tier,
    )


def routed_fallback_attempt(
    *, response_guard_tools: list[str], requested_model: str, names: list[str], tier: str
) -> RouteAttempt:
    """Build the passthrough attempt for a routed call that returned no body."""
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
    return RouteAttempt(
        reason=reason,
        model=requested_model,
        tool_names=names,
        tier=tier,
    )


async def try_codex_direct_route(
    *,
    body: JsonObject,
    tier: str,
    tier_cfg: JsonObject,
    credential: routing.TierCredential,
    request_headers: Mapping[str, str],
    streaming: bool,
    requested_model: str,
    names: list[str],
    premium_tools: list[str],
    request_path: str,
    user_agent: str,
    agent: str,
    session: str,
    client: httpx.AsyncClient | None = None,
) -> RouteAttempt | None:
    """Serve a Codex Responses request through the direct Codex provider path."""
    if request_path != "/v1/responses":
        return None
    try:
        tier_model = parse_tier_config(tier_cfg).model
        if not is_codex_tier(tier_cfg):
            return None
    except ConfigSchemaError:
        return None
    started = time.monotonic()
    result = await codex_direct.responses_call(
        body=body,
        tier_cfg=tier_cfg,
        credential=credential.value,
        auth_mode=credential.auth_mode,
        chatgpt_headers=websocket_passthrough.codex_chatgpt_headers(request_headers),
        client=client,
    )
    duration = time.monotonic() - started
    if result is None:
        return routed_fallback_attempt(
            response_guard_tools=[],
            requested_model=requested_model,
            names=names,
            tier=tier,
        )
    response_body, usage_values, shrink = result
    response_guard_tools = routed_call.premium_tool_names_from_responses(
        response_body, {}, premium_tools=premium_tools
    )
    if response_guard_tools:
        return routed_fallback_attempt(
            response_guard_tools=response_guard_tools,
            requested_model=requested_model,
            names=names,
            tier=tier,
        )
    usage = routed_call.RouteUsage(
        input_tokens=usage_values[0],
        output_tokens=usage_values[1],
        cache_read_tokens=usage_values[2],
        cache_creation_tokens=usage_values[3],
    )
    await routed_call.record_route_outcome(
        routed_call.RouteOutcome(
            premium_model=requested_model,
            routed_model=tier_model,
            tier_name=tier,
            tier_cfg=tier_cfg,
            tool_names=names,
            usage=usage,
            request_path=request_path,
            route="direct",
            user_agent=user_agent,
            agent=agent,
            session_id=session,
            elapsed=duration,
            shrink=shrink,
            request_id=logging_setup.current_request_id(),
        )
    )
    client_body = {**response_body, "model": requested_model} if requested_model else response_body
    response = (
        Response(
            b"".join(synthesize_response_object_sse(client_body)),
            media_type="text/event-stream",
        )
        if streaming
        else JSONResponse(client_body)
    )
    return RouteAttempt(
        response=response,
        tier=tier,
        reason="routed",
        model=tier_model,
        premium_model=requested_model,
        tool_names=names,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        duration=duration,
        shrink=shrink,
    )


async def try_route(  # noqa: PLR0911 - pure move of fail-open routing branches.
    body_bytes: bytes,
    *,
    adapter: ClientAdapter,
    group: str,
    request_path: str,
    request_headers: Mapping[str, str],
    routes_loader: Callable[[], JsonObject],
    client: httpx.AsyncClient | None = None,
    user_agent: str = "",
    agent: str = "",
    session: str | None = None,
) -> RouteAttempt:
    """Attempt to serve an adapter request on a routed tier."""
    requested_model = ""
    names: list[str] = []
    try:
        body = cast("JsonValue", json.loads(body_bytes))
        if not isinstance(body, dict):
            return RouteAttempt(reason="non_object_body")
        requested_model = str(body.get("model") or "")
        anthropic_body = adapter.to_anthropic_request(body)
        requested_model = str(anthropic_body.get("model") or requested_model)
        streaming = wants_stream(body, anthropic_body)
        names = adapter.extract_pending_tools(body)
        if not names:
            logger.debug("no pending tool results; passing through")
            return RouteAttempt(reason="no_pending_tools", model=requested_model)
        logger.debug("pending tools=%s", names)
        tier, tier_cfg, credential, routes = resolve_tier_config(
            names, group=group, routes_loader=routes_loader
        )
        logger.debug("resolved tier=%s routable=%s", tier, tier_cfg is not None)
        if tier_cfg is None:
            reason = "request_premium_guard" if tier == "premium" else "tier_unavailable"
            return premium_guard_attempt(
                reason=reason, requested_model=requested_model, names=names, tier=tier
            )
        if credential is None:
            return premium_guard_attempt(
                reason="tier_unavailable", requested_model=requested_model, names=names, tier=tier
            )
        savings.register_tier_prices(routes)
        settings_value = routes.get("settings")
        settings_cfg: JsonObject = settings_value if isinstance(settings_value, dict) else {}
        premium_tools = routing.agent_premium_tools(routes, group)
        codex_attempt = await try_codex_direct_route(
            body=body,
            tier=tier,
            tier_cfg=tier_cfg,
            credential=credential,
            request_headers=request_headers,
            streaming=streaming,
            requested_model=requested_model,
            names=names,
            premium_tools=premium_tools,
            request_path=request_path,
            user_agent=user_agent,
            agent=agent,
            session=session or "",
            client=client,
        )
        if codex_attempt is not None:
            return codex_attempt
        response_guard_tools: list[str] = []

        result = await routed_call.routed_call(
            body=anthropic_body,
            tier_name=tier,
            tier_cfg=tier_cfg,
            api_key=credential.value,
            settings=settings_cfg,
            tool_names=names,
            premium_tools=premium_tools,
            request_path=request_path,
            user_agent=user_agent,
            agent=group,
            session_id=session or "",
            on_premium_guard=response_guard_tools.extend,
            client=client,
        )
        if result is None:
            return routed_fallback_attempt(
                response_guard_tools=response_guard_tools,
                requested_model=requested_model,
                names=names,
                tier=tier,
            )
        client_body = adapter.to_client_response(result.body)
        if streaming:
            return RouteAttempt(
                response=Response(
                    b"".join(adapter.to_client_sse(result.body)),
                    media_type="text/event-stream",
                ),
                tier=tier,
                reason="routed",
                model=requested_model,
                tool_names=names,
            )
        return RouteAttempt(
            response=JSONResponse(client_body),
            tier=tier,
            reason="routed",
            model=requested_model,
            tool_names=names,
        )
    except Exception as exc:
        logger.warning("acr: routing decision failed (%s); passing through", exc, exc_info=True)
        return RouteAttempt(reason="routing_error", model=requested_model, tool_names=names)


async def try_route_ws_response_create(
    first_msg_raw: str,
    headers: Mapping[str, str],
    *,
    routes_loader: Callable[[], JsonObject],
) -> list[str] | None:
    """Try to serve one Codex WS `response.create` through the routed core."""
    try:
        body = websocket_passthrough.response_create_to_http_body(first_msg_raw)
        if body is None:
            return None
        adapter = adapter_for_path("/v1/responses")
        if adapter is None:
            return None
        routes = routes_loader()
        group = provider_config.resolve_agent_group(
            path="/v1/responses",
            headers=headers,
            routes=routes,
            adapter_default=adapter.default_agent_group,
        )
        if group is None:
            return None
        body_bytes = json.dumps(
            body, separators=(",", ":"), ensure_ascii=False, sort_keys=True
        ).encode()
        user_agent = headers.get("user-agent", "")
        attempt = await try_route(
            body_bytes,
            adapter=adapter,
            group=group,
            request_path="/v1/responses",
            request_headers=headers,
            user_agent=user_agent,
            agent=metrics.identify_agent(user_agent),
            session=None,
            routes_loader=routes_loader,
        )
        record_ws_route_attempt(headers, attempt)
        if attempt.response is None:
            return None
        previous_response_id = body.get("previous_response_id")
        response_id = previous_response_id if isinstance(previous_response_id, str) else None
        return websocket_passthrough.sse_to_ws_messages(
            bytes(attempt.response.body), response_id=response_id
        )
    except Exception as exc:
        logger.warning("acr: websocket routing decision failed (%s); passing through", exc)
        return None


def wants_stream(client_body: JsonObject, anthropic_body: JsonObject) -> bool:
    """Return whether the client requested a streaming response."""
    stream = client_body.get("stream")
    if isinstance(stream, bool):
        return stream
    anthropic_stream = anthropic_body.get("stream")
    if isinstance(anthropic_stream, bool):
        return anthropic_stream
    return False
