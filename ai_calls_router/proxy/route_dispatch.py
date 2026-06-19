"""Routing orchestration for adapter-backed proxy requests.

This module sits above the routing engine and adapters but below the Starlette
route table. It owns tier resolution, routed-call dispatch, and premium-guard
attempt construction without importing the app module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from starlette.responses import JSONResponse, Response

from ai_calls_router.accounting import savings, shrink_stats
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call

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
    tier_cfg = routing.agent_tier_config(routes, group, tier)
    if tier_cfg is None:
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
    session: str | None = None,
) -> RouteAttempt:
    """Attempt to serve an adapter request on a routed tier."""
    del request_headers
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


def wants_stream(client_body: JsonObject, anthropic_body: JsonObject) -> bool:
    """Return whether the client requested a streaming response."""
    stream = client_body.get("stream")
    if isinstance(stream, bool):
        return stream
    anthropic_stream = anthropic_body.get("stream")
    if isinstance(anthropic_stream, bool):
        return anthropic_stream
    return False
