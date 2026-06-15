"""Routed call engine: serve a tool-result turn on a cheap tier model.

routed_call rewrites the client's Anthropic request for the tier model
(_prepare_routed_body) and serves it on one of two paths: providers with a
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

import logging
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from ai_calls_router._lib.conversion import (
    BackendResponse,
    completion_kwargs,
    to_anthropic_response,
)
from ai_calls_router._lib.litellm_guard import load_litellm
from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.accounting import savings, shrink_stats
from ai_calls_router.routing import direct as anthropic_direct
from ai_calls_router.routing.compression import compress_litellm_messages

logger = logging.getLogger("acr.routed_call")


def _strip_thinking_from_messages(body: dict[str, Any], routed: dict[str, Any]) -> None:
    """Strip thinking blocks from messages, dropping emptied assistant messages."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    cleaned: list[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        blocks = [
            b
            for b in content
            if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
        ]
        if not blocks and msg.get("role") == "assistant":
            continue
        cleaned.append({**msg, "content": blocks})
    routed["messages"] = cleaned


def _clamp_max_tokens(
    body: dict[str, Any], tier_cfg: dict[str, Any], routed: dict[str, Any]
) -> None:
    """Clamp max_tokens to the tier limit when needed."""
    tier_max = tier_cfg.get("max_tokens")
    if isinstance(tier_max, int) and not isinstance(tier_max, bool) and tier_max > 0:
        requested = body.get("max_tokens")
        if not isinstance(requested, int) or requested > tier_max:
            routed["max_tokens"] = tier_max


def _prepare_routed_body(body: dict[str, Any], tier_cfg: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a client request body for the tier model.

    Swaps in the tier model, removes the stream flag (routed calls are
    buffered), clamps max_tokens to the tier limit when the requested value
    is missing, non-int, or larger, and strips thinking/redacted_thinking
    blocks the routed provider cannot interpret. Assistant messages emptied
    by the stripping are dropped. The input body is never mutated.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".

    Returns:
        A new request body ready for conversion to the routed provider.
    """
    routed = dict(body)
    routed["model"] = tier_cfg["model"]
    routed.pop("stream", None)
    _clamp_max_tokens(body, tier_cfg, routed)
    _strip_thinking_from_messages(body, routed)
    return routed


def _usage_int(usage: dict[str, Any], key: str) -> int:
    """Return a non-negative integer usage counter from a response body."""
    value = usage.get(key, 0)
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def _usage_summary(response_body: dict[str, Any]) -> str:
    """Return a compact token/cache summary for routing logs."""
    raw_usage = response_body.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    return (
        f"in={_usage_int(usage, 'input_tokens')} "
        f"out={_usage_int(usage, 'output_tokens')} "
        f"cache_hit={_usage_int(usage, 'cache_read_input_tokens')} "
        f"cache_miss={_usage_int(usage, 'cache_creation_input_tokens')}"
    )


def _shrink_summary(stats: shrink_stats.ShrinkStats) -> str:
    """Return a compact tool_result compression summary for routing logs."""
    return (
        f"shrink={stats.path} "
        f"chars={stats.chars_before}->{stats.chars_after} "
        f"saved={stats.chars_saved}({stats.ratio:.0%}) "
        f"est_tok=-{stats.est_tokens_saved()}"
    )


def escalates(response_body: dict[str, Any], settings: dict[str, Any]) -> bool:
    """Check whether a routed response invokes a premium tool.

    Args:
        response_body: Anthropic-format response body from the routed call.
        settings: The config "settings" section; reads
            escalate_on_premium_tools (default True) and premium_tools.

    Returns:
        True when the guard is enabled and any tool_use block names a
        premium tool, otherwise False.
    """
    return bool(_premium_tool_names(response_body, settings))


def _premium_tool_names(response_body: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    """Return premium tool names invoked by a routed response."""
    if not settings.get("escalate_on_premium_tools", True):
        return []
    premium_tools = set(settings.get("premium_tools") or [])
    if not premium_tools:
        return []
    content = response_body.get("content")
    if not isinstance(content, list):
        return []
    return [
        block["name"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name") in premium_tools
        and isinstance(block.get("name"), str)
    ]


async def _serve_via_litellm(
    *,
    body: dict[str, Any],
    tier_cfg: dict[str, Any],
    api_key: str,
    compress: bool,
) -> tuple[dict[str, Any], shrink_stats.ShrinkStats]:
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
    routed_body = _prepare_routed_body(body, tier_cfg)
    kwargs = completion_kwargs(routed_body, api_key)
    if compress:
        kwargs["messages"], stats = compress_litellm_messages(
            kwargs["messages"], model=tier_cfg["model"]
        )
    else:
        stats = shrink_stats.compute_shrink(path="none", before=body, after=body)
    litellm = load_litellm()
    raw = await litellm.acompletion(**kwargs)
    return to_anthropic_response(raw, tier_cfg["model"]), stats


async def _serve_via_direct(
    *,
    body: dict[str, Any],
    tier_cfg: dict[str, Any],
    api_key: str,
) -> tuple[dict[str, Any] | None, shrink_stats.ShrinkStats]:
    """Serve a turn directly on the provider's native Anthropic endpoint.

    Skips LiteLLM conversion and sends the body unmodified so consecutive
    tool-result turns keep byte-identical prefixes, letting the provider's
    prefix cache do the work. The router applies no compression or reduction on
    this path: DeepSeek's input is never shrunk, preserving cache stability, and
    any token reduction is delegated to the upstream Headroom layer.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key.

    Returns:
        A pair of the routed response in Anthropic format (or None when the
        direct call fails and the turn must pass through) and a read-only
        ShrinkStats. No shrink pass runs, so the stats report a no-op (path
        "none", zero chars saved).
    """
    stats = shrink_stats.compute_shrink(path="none", before=body, after=body)
    routed_body = _prepare_routed_body(body, tier_cfg)
    response = await anthropic_direct.direct_call(
        body=routed_body, tier_cfg=tier_cfg, api_key=api_key
    )
    return response, stats


def _token_fields(body: dict[str, Any]) -> dict[str, int]:
    """Extract token counts from an Anthropic usage block.

    Returns a dict with keys: input, output, cache_read, cache_creation.
    """
    usage = body.get("usage") or {}
    return {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cache_read": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation": int(usage.get("cache_creation_input_tokens", 0) or 0),
    }


def _record_metrics(
    *,
    tier_name: str,
    model: str,
    premium_model: str,
    direct: bool,
    tool_names: list[str],
    user_agent: str,
    anthropic_body: dict[str, Any],
    elapsed: float,
    shrink: shrink_stats.ShrinkStats,
    agent: str = "",
    session_id: str = "",
) -> None:
    """Record per-request metrics after a successful routed call."""
    mtr = metrics_mod.get_metrics()
    tok = _token_fields(anthropic_body)
    mtr.add_routed_tokens(
        input_tokens=tok["input"],
        output_tokens=tok["output"],
        cache_read=tok["cache_read"],
        cache_creation=tok["cache_creation"],
    )
    mtr.add_shrink(chars_before=shrink.chars_before, chars_after=shrink.chars_after)
    mtr.record_request(
        method="POST",
        path="/v1/messages",
        status=200,
        tier=tier_name,
        route="direct" if direct else "litellm",
        model=model,
        user_agent=user_agent,
        client_ip="",
        tool_names=tool_names,
        input_tokens=tok["input"],
        output_tokens=tok["output"],
        cache_read=tok["cache_read"],
        cache_creation=tok["cache_creation"],
        duration=elapsed,
        premium_model=premium_model,
        agent=agent,
        session_id=session_id,
        shrink_chars_before=shrink.chars_before,
        shrink_chars_after=shrink.chars_after,
    )


async def routed_call(
    *,
    body: dict[str, Any],
    tier_name: str,
    tier_cfg: dict[str, Any],
    api_key: str,
    settings: dict[str, Any],
    tool_names: list[str] | None = None,
    user_agent: str = "",
    agent: str = "",
    session_id: str = "",
    on_premium_guard: Callable[[list[str]], None] | None = None,
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
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label.
        session_id: Session fingerprint hex string.
        on_premium_guard: Optional callback invoked with response tool names
            when the escalation guard rejects a routed response.

    Returns:
        A BackendResponse with the masked Anthropic body, or None when the
        caller must replay the turn on premium passthrough.
    """
    premium_model = body.get("model")
    model = tier_cfg.get("model")
    direct = anthropic_direct.direct_endpoint(model) is not None
    _tool_names = tool_names or []
    try:
        started = time.monotonic()
        if direct:
            anthropic_body, shrink = await _serve_via_direct(
                body=body, tier_cfg=tier_cfg, api_key=api_key
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
        premium_guard_tools = _premium_tool_names(anthropic_body, settings)
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
    savings.record_savings_from_response(
        premium_model=premium_model,
        routed_model=cast("str", model),
        response_body=anthropic_body,
        tier_cfg=tier_cfg,
        tier_name=tier_name,
        tool_names=tool_names,
        user_agent=user_agent,
        agent=agent,
        session_id=session_id,
        shrink=shrink,
    )
    if isinstance(premium_model, str) and premium_model:
        anthropic_body = {**anthropic_body, "model": premium_model}
    _record_metrics(
        tier_name=tier_name,
        model=cast("str", model),
        premium_model=str(premium_model) if premium_model else "",
        direct=direct,
        tool_names=_tool_names,
        user_agent=user_agent,
        anthropic_body=anthropic_body,
        elapsed=elapsed,
        shrink=shrink,
        agent=agent,
        session_id=session_id,
    )
    return BackendResponse(body=anthropic_body)
