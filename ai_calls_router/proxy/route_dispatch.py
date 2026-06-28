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
from ai_calls_router._lib.responses_inbound import (
    anthropic_request_to_responses,
    responses_to_anthropic_response,
)
from ai_calls_router.accounting import savings, shrink_stats
from ai_calls_router.proxy import chatgpt_oauth, passthrough
from ai_calls_router.routing import anthropic_oauth, codex_direct, content_sanitize
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
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


@dataclass(frozen=True)
class RouteDecision:
    """Resolved routing decision for a routable request.

    Carries the parsed request bodies, serve flags, and the chosen tier with its
    credential so the dispatch stage can serve the call without re-deriving any
    of it. Only built when the request is routable; otherwise the decision stage
    returns a :class:`RouteAttempt` describing the passthrough reason.
    """

    body: JsonObject
    anthropic_body: JsonObject
    requested_model: str
    streaming: bool
    names: list[str]
    tier: str
    tier_cfg: JsonObject
    credential: routing.TierCredential
    settings: JsonObject
    premium_tools: list[str]


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


def _routing_error_attempt(
    exc: Exception, *, requested_model: str, names: list[str]
) -> RouteAttempt:
    """Log a fail-open routing failure and build its passthrough attempt."""
    logger.warning("acr: routing decision failed (%s); passing through", exc, exc_info=True)
    return RouteAttempt(reason="routing_error", model=requested_model, tool_names=names)


def _adapter_response(
    adapter: ClientAdapter, anthropic_response: JsonObject, *, streaming: bool
) -> Response:
    """Shape an Anthropic-shaped response into the client's wire format.

    Blank text blocks are stripped first so the proxy never emits one. A routed
    tier model can return an empty text block ahead of a tool_use; persisted by
    the client it would 400 ("text content blocks must be non-empty") on every
    later turn that replays it.
    """
    anthropic_response = content_sanitize.clean_response_content(anthropic_response)
    if streaming:
        return Response(
            b"".join(adapter.to_client_sse(anthropic_response)),
            media_type="text/event-stream",
        )
    return JSONResponse(adapter.to_client_response(anthropic_response))


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
        chatgpt_headers=chatgpt_oauth.codex_chatgpt_headers(request_headers),
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


async def try_oauth_responses_route(
    *,
    anthropic_body: JsonObject,
    adapter: ClientAdapter,
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
    """Serve any agent group through a ChatGPT OAuth Responses tier."""
    if credential.auth_mode != "oauth":
        return None
    started = time.monotonic()
    responses_body = anthropic_request_to_responses(anthropic_body)
    result = await codex_direct.responses_call(
        body=responses_body,
        tier_cfg=tier_cfg,
        credential=credential.value,
        auth_mode=credential.auth_mode,
        chatgpt_headers=chatgpt_oauth.codex_chatgpt_headers(request_headers),
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
    tier_model = codex_direct.native_model_id(tier_cfg)
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
            route="oauth_responses",
            user_agent=user_agent,
            agent=agent,
            session_id=session,
            elapsed=duration,
            shrink=shrink,
            request_id=logging_setup.current_request_id(),
        )
    )
    anthropic_response = responses_to_anthropic_response(response_body, requested_model)
    return RouteAttempt(
        response=_adapter_response(adapter, anthropic_response, streaming=streaming),
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


async def try_anthropic_oauth_route(
    decision: RouteDecision,
    *,
    adapter: ClientAdapter,
    request_headers: Mapping[str, str],
    request_path: str,
    user_agent: str,
    agent: str,
    session: str,
    client: httpx.AsyncClient | None = None,
) -> RouteAttempt | None:
    """Serve a claude_code cheap turn on Anthropic subscription OAuth.

    Forwards the model-swapped Anthropic body to api.anthropic.com with the
    client's inbound OAuth bearer so the turn draws the subscription's quota for
    the tier model (e.g. Sonnet) instead of paying per token. Returns None when
    the tier is not an Anthropic-OAuth Messages tier so non-matching tiers fall
    through to the Codex paths and the LiteLLM engine.
    """
    if decision.credential.auth_mode != "oauth" or request_path != "/v1/messages":
        return None
    if not anthropic_oauth.is_anthropic_oauth_tier(decision.tier_cfg):
        return None
    started = time.monotonic()
    result = await anthropic_oauth.messages_call(
        body=decision.anthropic_body,
        tier_cfg=decision.tier_cfg,
        oauth_headers=passthrough.filter_request_headers(request_headers),
        compress=bool(decision.settings.get("compress_routed", True)),
        enable_text_ml=bool(decision.tier_cfg.get("text_ml_compression")),
        prompt_cache=bool(decision.settings.get("anthropic_prompt_cache", False)),
        client=client,
    )
    duration = time.monotonic() - started
    if result is None:
        return routed_fallback_attempt(
            response_guard_tools=[],
            requested_model=decision.requested_model,
            names=decision.names,
            tier=decision.tier,
        )
    response_body, usage_values, shrink = result
    response_guard_tools = routed_call.premium_tool_names_from_anthropic(
        response_body, decision.settings, premium_tools=decision.premium_tools
    )
    if response_guard_tools:
        return routed_fallback_attempt(
            response_guard_tools=response_guard_tools,
            requested_model=decision.requested_model,
            names=decision.names,
            tier=decision.tier,
        )
    tier_model = anthropic_oauth.native_model_id(decision.tier_cfg)
    usage = routed_call.RouteUsage(
        input_tokens=usage_values[0],
        output_tokens=usage_values[1],
        cache_read_tokens=usage_values[2],
        cache_creation_tokens=usage_values[3],
    )
    await routed_call.record_route_outcome(
        routed_call.RouteOutcome(
            premium_model=decision.requested_model,
            routed_model=tier_model,
            tier_name=decision.tier,
            tier_cfg=decision.tier_cfg,
            tool_names=decision.names,
            usage=usage,
            request_path=request_path,
            route="anthropic_oauth",
            user_agent=user_agent,
            agent=agent,
            session_id=session,
            elapsed=duration,
            shrink=shrink,
            request_id=logging_setup.current_request_id(),
        )
    )
    client_body = (
        {**response_body, "model": decision.requested_model}
        if decision.requested_model
        else response_body
    )
    return RouteAttempt(
        response=_adapter_response(adapter, client_body, streaming=decision.streaming),
        tier=decision.tier,
        reason="routed",
        model=tier_model,
        premium_model=decision.requested_model,
        tool_names=decision.names,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        duration=duration,
        shrink=shrink,
    )


async def try_native_or_oauth_route(
    decision: RouteDecision,
    *,
    adapter: ClientAdapter,
    request_headers: Mapping[str, str],
    request_path: str,
    user_agent: str,
    agent: str,
    session: str,
    client: httpx.AsyncClient | None = None,
) -> RouteAttempt | None:
    """Try Anthropic OAuth, native Codex, then generic OAuth Responses routing."""
    anthropic_attempt = await try_anthropic_oauth_route(
        decision,
        adapter=adapter,
        request_headers=request_headers,
        request_path=request_path,
        user_agent=user_agent,
        agent=agent,
        session=session,
        client=client,
    )
    if anthropic_attempt is not None:
        return anthropic_attempt
    codex_attempt = await try_codex_direct_route(
        body=decision.body,
        tier=decision.tier,
        tier_cfg=decision.tier_cfg,
        credential=decision.credential,
        request_headers=request_headers,
        streaming=decision.streaming,
        requested_model=decision.requested_model,
        names=decision.names,
        premium_tools=decision.premium_tools,
        request_path=request_path,
        user_agent=user_agent,
        agent=agent,
        session=session,
        client=client,
    )
    if codex_attempt is not None:
        return codex_attempt
    return await try_oauth_responses_route(
        anthropic_body=decision.anthropic_body,
        adapter=adapter,
        tier=decision.tier,
        tier_cfg=decision.tier_cfg,
        credential=decision.credential,
        request_headers=request_headers,
        streaming=decision.streaming,
        requested_model=decision.requested_model,
        names=decision.names,
        premium_tools=decision.premium_tools,
        request_path=request_path,
        user_agent=user_agent,
        agent=agent,
        session=session,
        client=client,
    )


def prepare_route(
    body_bytes: bytes,
    *,
    adapter: ClientAdapter,
    group: str,
    routes_loader: Callable[[], JsonObject],
) -> RouteDecision | RouteAttempt:
    """Resolve the routing decision for a request.

    Returns a :class:`RouteDecision` when the request is routable, otherwise a
    :class:`RouteAttempt` describing why it falls back to premium passthrough.
    Every path is fail-open: a parse/resolution failure surfaces the partial
    requested model so the caller records it, and routing never breaks traffic
    that would otherwise succeed.
    """
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
        return RouteDecision(
            body=body,
            anthropic_body=anthropic_body,
            requested_model=requested_model,
            streaming=streaming,
            names=names,
            tier=tier,
            tier_cfg=tier_cfg,
            credential=credential,
            settings=settings_cfg,
            premium_tools=routing.agent_premium_tools(routes, group),
        )
    except Exception as exc:
        return _routing_error_attempt(exc, requested_model=requested_model, names=names)


async def _serve_routed_litellm(
    decision: RouteDecision,
    *,
    adapter: ClientAdapter,
    request_path: str,
    user_agent: str,
    agent: str,
    session: str,
    client: httpx.AsyncClient | None = None,
) -> RouteAttempt:
    """Serve a routed request through the LiteLLM engine and shape the response."""
    response_guard_tools: list[str] = []
    result = await routed_call.routed_call(
        body=decision.anthropic_body,
        tier_name=decision.tier,
        tier_cfg=decision.tier_cfg,
        api_key=decision.credential.value,
        settings=decision.settings,
        tool_names=decision.names,
        premium_tools=decision.premium_tools,
        request_path=request_path,
        user_agent=user_agent,
        agent=agent,
        session_id=session,
        on_premium_guard=response_guard_tools.extend,
        client=client,
    )
    if result is None:
        return routed_fallback_attempt(
            response_guard_tools=response_guard_tools,
            requested_model=decision.requested_model,
            names=decision.names,
            tier=decision.tier,
        )
    return RouteAttempt(
        response=_adapter_response(adapter, result.body, streaming=decision.streaming),
        tier=decision.tier,
        reason="routed",
        model=decision.requested_model,
        tool_names=decision.names,
    )


async def try_route(
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
    """Attempt to serve an adapter request on a routed tier.

    Resolves the decision (:func:`prepare_route`), then dispatches to the native
    Codex / OAuth Responses path or the LiteLLM engine. A failure during dispatch
    falls back to premium passthrough while preserving the partial model.
    """
    decision = prepare_route(body_bytes, adapter=adapter, group=group, routes_loader=routes_loader)
    if isinstance(decision, RouteAttempt):
        return decision
    session_id = session or ""
    try:
        attempt = await try_native_or_oauth_route(
            decision,
            adapter=adapter,
            request_headers=request_headers,
            request_path=request_path,
            user_agent=user_agent,
            agent=agent,
            session=session_id,
            client=client,
        )
        if attempt is not None:
            return attempt
        return await _serve_routed_litellm(
            decision,
            adapter=adapter,
            request_path=request_path,
            user_agent=user_agent,
            agent=agent,
            session=session_id,
            client=client,
        )
    except Exception as exc:
        return _routing_error_attempt(
            exc, requested_model=decision.requested_model, names=decision.names
        )


def wants_stream(client_body: JsonObject, anthropic_body: JsonObject) -> bool:
    """Return whether the client requested a streaming response."""
    stream = client_body.get("stream")
    if isinstance(stream, bool):
        return stream
    anthropic_stream = anthropic_body.get("stream")
    if isinstance(anthropic_stream, bool):
        return anthropic_stream
    return False
