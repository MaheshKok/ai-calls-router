"""Routed call engine: serve a tool-result turn on a cheap tier model.

routed_call rewrites the client's Anthropic request for the tier model
(prepare_routed_body) and serves it on one of two paths: providers with a
native Anthropic endpoint (DeepSeek) receive the body directly -- no LiteLLM
conversion -- so consecutive tool-result turns keep byte-identical prefixes for
the provider's prefix cache; every other provider goes through LiteLLM. The
direct path applies no compression (the byte-identical prefix is what the cache
needs); the LiteLLM path optionally runs the converted messages through headroom
when "compress_routed" is set. Responses that invoke a premium
tool are discarded (escalates) so the caller replays the turn on premium
passthrough, and savings are recorded under the true routed model before the
served body is masked to the client-requested model. Every failure path returns
None so routing never breaks a turn. SSE rendering of the buffered response
lives in the sibling synthesis module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

from ai_calls_router._lib import jsonnum
from ai_calls_router._lib.conversion import (
    BackendResponse,
    LiteLLMResponse,
    completion_kwargs,
    to_anthropic_response,
)
from ai_calls_router._lib.litellm_guard import load_litellm
from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.accounting import savings, shrink_stats
from ai_calls_router.routing import direct as anthropic_direct
from ai_calls_router.routing.compression import compress_litellm_messages

logger = logging.getLogger("acr.routed_call")


class _LiteLLMCompletion(Protocol):
    """LiteLLM async completion surface used by the routed engine."""

    async def acompletion(self, **kwargs: JsonValue) -> LiteLLMResponse: ...


@dataclass(frozen=True)
class RouteUsage:
    """Normalized routed token buckets."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass(frozen=True)
class RouteOutcome:
    """Transport-agnostic routed outcome for shared accounting."""

    premium_model: str | None
    routed_model: str
    tier_name: str
    tier_cfg: JsonObject
    tool_names: list[str]
    usage: RouteUsage
    request_path: str
    route: str
    user_agent: str = ""
    agent: str = ""
    session_id: str = ""
    elapsed: float = 0.0
    shrink: shrink_stats.ShrinkStats | None = None
    method: str = "POST"
    status: int = 200
    request_id: str = ""


def _strip_thinking_from_messages(body: JsonObject, routed: JsonObject) -> None:
    """Strip thinking blocks from messages, dropping emptied assistant messages."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    cleaned: JsonArray = []
    for msg in cast("JsonArray", messages):
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        blocks = [
            b
            for b in cast("JsonArray", content)
            if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
        ]
        if not blocks and msg.get("role") == "assistant":
            continue
        cleaned.append(cast("JsonObject", {**msg, "content": blocks}))
    routed["messages"] = cleaned


def _clamp_max_tokens(body: JsonObject, tier_cfg: JsonObject, routed: JsonObject) -> None:
    """Clamp max_tokens to the tier limit when needed."""
    tier_max = tier_cfg.get("max_tokens")
    if isinstance(tier_max, int) and not isinstance(tier_max, bool) and tier_max > 0:
        requested = body.get("max_tokens")
        if not isinstance(requested, int) or requested > tier_max:
            routed["max_tokens"] = tier_max


# Opus accepts output_config.effort='xhigh'; Sonnet and other routed Anthropic
# models reject it (HTTP 400, supported: high/low/max/medium). A tier may pin its
# own routed reasoning level via tier_cfg "effort"; absent that, only the Opus-
# only 'xhigh' is downgraded so a routed turn never fails open on effort alone.
# Each tier decides independently; premium passthrough keeps its own level.
_UNSUPPORTED_ROUTED_EFFORT = "xhigh"
_ROUTED_EFFORT_FALLBACK = "high"


def _routed_effort(tier_cfg: JsonObject, current_effort: object) -> str | None:
    """Return the effort to force on a routed turn, or None to leave it as-is.

    A configured tier ``effort`` always wins (returning None only when it already
    matches the request, to avoid a needless copy). Without a tier override, the
    sole adjustment is the ``xhigh`` safety downgrade; any already-supported level
    is left untouched.

    Args:
        tier_cfg: Tier config; reads an optional ``effort`` override.
        current_effort: The effort the client requested, if any.

    Returns:
        The effort level to write onto the routed body, or None for no change.
    """
    configured = tier_cfg.get("effort")
    if isinstance(configured, str) and configured:
        return configured if configured != current_effort else None
    if current_effort == _UNSUPPORTED_ROUTED_EFFORT:
        return _ROUTED_EFFORT_FALLBACK
    return None


def _normalize_effort(body: JsonObject, tier_cfg: JsonObject, routed: JsonObject) -> None:
    """Apply the tier's routed effort, or the xhigh safety downgrade, to the body."""
    output_config = body.get("output_config")
    if not isinstance(output_config, dict):
        return
    target = _routed_effort(tier_cfg, output_config.get("effort"))
    if target is None:
        return
    routed["output_config"] = {**output_config, "effort": target}


def prepare_routed_body(body: JsonObject, tier_cfg: JsonObject) -> JsonObject:
    """Rewrite a client request body for the tier model.

    Swaps in the tier model, removes the stream flag (routed calls are
    buffered), clamps max_tokens to the tier limit when the requested value
    is missing, non-int, or larger, strips thinking/redacted_thinking blocks
    the routed provider cannot interpret, and sets the reasoning effort to the
    tier's configured ``effort`` (falling back to downgrading the Opus-only
    'xhigh' the routed model rejects). Assistant messages emptied by the
    stripping are dropped. The input body is never mutated.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model", optional "max_tokens", and optional
            "effort" reasoning level.

    Returns:
        A new request body ready for conversion to the routed provider.
    """
    routed = dict(body)
    routed["model"] = tier_cfg["model"]
    routed.pop("stream", None)
    _clamp_max_tokens(body, tier_cfg, routed)
    _strip_thinking_from_messages(body, routed)
    _normalize_effort(body, tier_cfg, routed)
    return routed


def _usage_summary(response_body: JsonObject) -> str:
    """Return a compact token/cache summary for routing logs."""
    raw_usage = response_body.get("usage")
    usage: JsonObject = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = jsonnum.int_value(usage.get("input_tokens", 0), minimum=0)
    output_tokens = jsonnum.int_value(usage.get("output_tokens", 0), minimum=0)
    cache_read = jsonnum.int_value(usage.get("cache_read_input_tokens", 0), minimum=0)
    cache_creation = jsonnum.int_value(usage.get("cache_creation_input_tokens", 0), minimum=0)
    return (
        f"in={input_tokens} out={output_tokens} cache_hit={cache_read} cache_miss={cache_creation}"
    )


def _shrink_summary(stats: shrink_stats.ShrinkStats) -> str:
    """Return a compact tool_result compression summary for routing logs."""
    return (
        f"shrink={stats.path} "
        f"chars={stats.chars_before}->{stats.chars_after} "
        f"saved={stats.chars_saved}({stats.ratio:.0%}) "
        f"est_tok=-{stats.est_tokens_saved()}"
    )


def escalates(
    response_body: JsonObject,
    settings: JsonObject,
    *,
    premium_tools: list[str] | None = None,
) -> bool:
    """Check whether a routed response invokes a premium tool.

    Args:
        response_body: Anthropic-format response body from the routed call.
        settings: The config "settings" section; reads
            escalate_on_premium_tools (default True).
        premium_tools: Agent-specific premium tool names. When omitted, falls
            back to legacy settings.premium_tools.

    Returns:
        True when the guard is enabled and any tool_use block names a
        premium tool, otherwise False.
    """
    return bool(
        premium_tool_names_from_anthropic(response_body, settings, premium_tools=premium_tools)
    )


def _response_guard_tool_names(
    response_tool_names: list[str],
    settings: JsonObject,
    *,
    premium_tools: list[str] | None = None,
) -> list[str]:
    """Return guarded premium tool names from response tool names."""
    if not settings.get("escalate_on_premium_tools", True):
        return []
    raw_tools = premium_tools if premium_tools is not None else settings.get("premium_tools")
    guard_tools: set[str] = (
        {tool for tool in raw_tools if isinstance(tool, str)}
        if isinstance(raw_tools, list)
        else set()
    )
    if not guard_tools:
        return []
    names: list[str] = []
    for name in response_tool_names:
        if name in guard_tools and name not in names:
            names.append(name)
    return names


def _anthropic_response_tool_names(response_body: JsonObject) -> list[str]:
    """Return tool names invoked by an Anthropic response body."""
    content = response_body.get("content")
    if not isinstance(content, list):
        return []
    return [
        str(block["name"])
        for block in cast("JsonArray", content)
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("name"), str)
    ]


def _responses_output_tool_names(response_body: JsonObject) -> list[str]:
    """Return tool names invoked by an OpenAI Responses body."""
    output = response_body.get("output")
    if not isinstance(output, list):
        return []
    names: list[str] = []
    for item in cast("JsonArray", output):
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        name = item.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def premium_tool_names_from_anthropic(
    response_body: JsonObject,
    settings: JsonObject,
    *,
    premium_tools: list[str] | None = None,
) -> list[str]:
    """Return premium tool names invoked by an Anthropic response."""
    return _response_guard_tool_names(
        _anthropic_response_tool_names(response_body),
        settings,
        premium_tools=premium_tools,
    )


def premium_tool_names_from_responses(
    response_body: JsonObject,
    settings: JsonObject,
    *,
    premium_tools: list[str] | None = None,
) -> list[str]:
    """Return premium tool names invoked by an OpenAI Responses body."""
    return _response_guard_tool_names(
        _responses_output_tool_names(response_body),
        settings,
        premium_tools=premium_tools,
    )


async def _serve_via_litellm(
    *,
    body: JsonObject,
    tier_cfg: JsonObject,
    api_key: str,
    compress: bool,
) -> tuple[JsonObject, shrink_stats.ShrinkStats]:
    """Serve a turn through LiteLLM, optionally compressing the routed messages.

    Rewrites the body for the tier model, converts it to OpenAI format, calls
    the provider through litellm with only the tier key, and converts the
    result back to Anthropic format. When ``compress`` is set, the converted
    OpenAI messages are passed through headroom before the provider call: that
    is the wire shape headroom is effective on (role="tool" messages with string
    content), and headroom's default exclusions still leave coding-agent output
    verbatim. This is the provider-agnostic path for every provider without a
    native Anthropic endpoint.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key.
        compress: When True, compress the converted messages with headroom; when
            False, the messages reach the provider unmodified.

    Returns:
        A pair of the routed response in Anthropic format (tagged with the tier
        model) and a read-only ShrinkStats. The stats report the headroom
        character delta when compression runs, or a no-op (path "none", zero
        chars saved) when it is disabled or unavailable.
    """
    routed_body = prepare_routed_body(body, tier_cfg)
    kwargs = completion_kwargs(routed_body, api_key)
    if compress:
        messages = kwargs.get("messages")
        model = tier_cfg.get("model")
        if not isinstance(messages, list) or not isinstance(model, str):
            stats = shrink_stats.compute_shrink(path="none", before=body, after=body)
        else:
            kwargs["messages"], stats = compress_litellm_messages(
                cast("JsonArray", messages),
                model=model,
                enable_text_ml=bool(tier_cfg.get("text_ml_compression")),
            )
    else:
        stats = shrink_stats.compute_shrink(path="none", before=body, after=body)
    litellm = cast("_LiteLLMCompletion", load_litellm())
    raw = await litellm.acompletion(**kwargs)
    model = tier_cfg.get("model")
    return to_anthropic_response(raw, model if isinstance(model, str) else ""), stats


async def _serve_via_direct(
    *,
    body: JsonObject,
    tier_cfg: JsonObject,
    api_key: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[JsonObject | None, shrink_stats.ShrinkStats]:
    """Serve a turn directly on the provider's native Anthropic endpoint.

    Skips LiteLLM conversion. The body is prepared for the tier by
    prepare_routed_body (model swap, max_tokens clamp, thinking strip, effort
    normalization) but never compressed or shrunk, so consecutive tool-result
    turns keep byte-identical prefixes and the provider's prefix cache does the
    work. DeepSeek's input is never reduced, preserving cache stability; any
    token reduction is delegated to the upstream Headroom layer.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key.
        client: Optional shared HTTP client for direct provider calls.

    Returns:
        A pair of the routed response in Anthropic format (or None when the
        direct call fails and the turn must pass through) and a read-only
        ShrinkStats. No shrink pass runs, so the stats report a no-op (path
        "none", zero chars saved).
    """
    stats = shrink_stats.compute_shrink(path="none", before=body, after=body)
    routed_body = prepare_routed_body(body, tier_cfg)
    response = await anthropic_direct.direct_call(
        body=routed_body, tier_cfg=tier_cfg, api_key=api_key, client=client
    )
    return response, stats


def _usage_from_anthropic(body: JsonObject) -> RouteUsage:
    """Extract normalized token counts from an Anthropic usage block."""
    usage_value = body.get("usage")
    usage: JsonObject = usage_value if isinstance(usage_value, dict) else {}
    return RouteUsage(
        input_tokens=jsonnum.int_value(usage.get("input_tokens", 0), minimum=0),
        output_tokens=jsonnum.int_value(usage.get("output_tokens", 0), minimum=0),
        cache_read_tokens=jsonnum.int_value(usage.get("cache_read_input_tokens", 0), minimum=0),
        cache_creation_tokens=jsonnum.int_value(
            usage.get("cache_creation_input_tokens", 0), minimum=0
        ),
    )


async def record_route_outcome(outcome: RouteOutcome) -> None:
    """Record routed savings and live metrics from normalized usage."""
    shrink = outcome.shrink or shrink_stats.compute_shrink(path="none", before={}, after={})
    usage = outcome.usage
    await asyncio.to_thread(
        savings.record_routing_savings,
        premium_model=outcome.premium_model,
        routed_model=outcome.routed_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        routed_prices=savings.routed_prices_from_tier(outcome.tier_cfg),
        tier_name=outcome.tier_name,
        tool_names=",".join(outcome.tool_names),
        user_agent=outcome.user_agent,
        agent=outcome.agent,
        session_id=outcome.session_id,
        shrink_path=shrink.path,
        shrink_chars_before=shrink.chars_before,
        shrink_chars_after=shrink.chars_after,
    )
    mtr = metrics_mod.get_metrics()
    mtr.add_routed_tokens(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read=usage.cache_read_tokens,
        cache_creation=usage.cache_creation_tokens,
    )
    mtr.add_shrink(chars_before=shrink.chars_before, chars_after=shrink.chars_after)
    mtr.incr_routed_model(model=outcome.routed_model)
    mtr.record_request(
        method=outcome.method,
        path=outcome.request_path,
        status=outcome.status,
        tier=outcome.tier_name,
        route=outcome.route,
        model=outcome.routed_model,
        user_agent=outcome.user_agent,
        client_ip="",
        tool_names=outcome.tool_names,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read=usage.cache_read_tokens,
        cache_creation=usage.cache_creation_tokens,
        duration=outcome.elapsed,
        premium_model=outcome.premium_model or "",
        agent=outcome.agent,
        session_id=outcome.session_id,
        shrink_chars_before=shrink.chars_before,
        shrink_chars_after=shrink.chars_after,
        request_id=outcome.request_id,
    )


async def routed_call(
    *,
    body: JsonObject,
    tier_name: str,
    tier_cfg: JsonObject,
    api_key: str,
    settings: JsonObject,
    tool_names: list[str] | None = None,
    premium_tools: list[str] | None = None,
    request_path: str = "/v1/messages",
    user_agent: str = "",
    agent: str = "",
    session_id: str = "",
    on_premium_guard: Callable[[list[str]], None] | None = None,
    client: httpx.AsyncClient | None = None,
) -> BackendResponse | None:
    """Serve a request on the tier model, falling back to None on any failure.

    Dispatches on the tier model: providers with a native Anthropic endpoint
    (DeepSeek) are served directly without LiteLLM conversion; every other
    provider goes through LiteLLM. The direct path never compresses (it must
    keep byte-identical prefixes); the LiteLLM path compresses the converted
    messages with headroom when settings["compress_routed"] is true (the
    default). Escalating responses (premium
    tool calls) are discarded. On success, savings are recorded under the true
    routed model and the served body is then masked to claim the
    client-requested model.

    Args:
        body: Anthropic-format request body from the client.
        tier_name: Tier name, used only for logging.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key; the client's credentials never reach here.
        settings: The config "settings" section.
        tool_names: Tool names extracted from the request body.
        premium_tools: Agent-specific premium tool names for response-side
            escalation. When omitted, legacy settings.premium_tools is used.
        request_path: Client-facing request path for metrics.
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label.
        session_id: Session fingerprint hex string.
        on_premium_guard: Optional callback invoked with response tool names
            when the escalation guard rejects a routed response.
        client: Optional shared HTTP client for direct provider calls.

    Returns:
        A BackendResponse with the masked Anthropic body, or None when the
        caller must replay the turn on premium passthrough.
    """
    premium_model_value = body.get("model")
    premium_model = premium_model_value if isinstance(premium_model_value, str) else None
    model = tier_cfg.get("model")
    direct = anthropic_direct.direct_endpoint(model) is not None
    _tool_names = tool_names or []
    try:
        started = time.monotonic()
        if direct:
            anthropic_body, shrink = await _serve_via_direct(
                body=body, tier_cfg=tier_cfg, api_key=api_key, client=client
            )
        else:
            anthropic_body, shrink = await _serve_via_litellm(
                body=body,
                tier_cfg=tier_cfg,
                api_key=api_key,
                compress=bool(settings.get("compress_routed", True)),
            )
        elapsed = time.monotonic() - started
        if anthropic_body is None:
            logger.warning(
                "acr: tier=%s model=%s direct call returned no body (%.2fs); passing through",
                tier_name,
                model,
                elapsed,
            )
            return None
        premium_guard_tools = premium_tool_names_from_anthropic(
            anthropic_body, settings, premium_tools=premium_tools
        )
        if premium_guard_tools:
            if on_premium_guard is not None:
                on_premium_guard(premium_guard_tools)
            logger.info(
                "acr: tier=%s model=%s escalated to premium tools=%s (%.2fs)",
                tier_name,
                model,
                premium_guard_tools,
                elapsed,
            )
            return None
    except Exception as exc:
        logger.warning(
            "acr: tier=%s model=%s routed call failed (%s); passing through",
            tier_name,
            model,
            exc,
            exc_info=True,
        )
        return None

    logger.info(
        "acr: routed tier=%s model=%s route=%s %s %s duration=%.2fs",
        tier_name,
        model,
        "direct" if direct else "litellm",
        _usage_summary(anthropic_body),
        _shrink_summary(shrink),
        elapsed,
    )
    await record_route_outcome(
        RouteOutcome(
            premium_model=premium_model,
            routed_model=cast("str", model),
            tier_name=tier_name,
            tier_cfg=tier_cfg,
            tool_names=_tool_names,
            usage=_usage_from_anthropic(anthropic_body),
            request_path=request_path,
            route="direct" if direct else "litellm",
            user_agent=user_agent,
            agent=agent,
            session_id=session_id,
            elapsed=elapsed,
            shrink=shrink,
        )
    )
    if isinstance(premium_model, str) and premium_model:
        anthropic_body = {**anthropic_body, "model": premium_model}
    return BackendResponse(body=anthropic_body)
