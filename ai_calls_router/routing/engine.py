"""Routed call engine: serve a tool-result turn on a cheap tier model.

routed_call rewrites the client's Anthropic request for the tier model
(_prepare_routed_body) and serves it on one of two paths: providers with a
native Anthropic endpoint (DeepSeek) receive the body directly -- no LiteLLM
conversion, no compression -- so consecutive tool-result turns keep
byte-identical prefixes for the provider's prefix cache; every other provider
goes through LiteLLM with built-in compression. Responses that invoke a premium
tool are discarded (escalates) so the caller replays the turn on premium
passthrough, and savings are recorded under the true routed model before the
served body is masked to the client-requested model. Every failure path returns
None so routing never breaks a turn. SSE rendering of the buffered response
lives in the sibling synthesis module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from ai_calls_router._lib.conversion import (
    BackendResponse,
    completion_kwargs,
    to_anthropic_response,
)
from ai_calls_router._lib.litellm_guard import load_litellm
from ai_calls_router.accounting import savings
from ai_calls_router.routing import compression, reduce
from ai_calls_router.routing import direct as anthropic_direct

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
    if not settings.get("escalate_on_premium_tools", True):
        return False
    premium_tools = set(settings.get("premium_tools") or [])
    if not premium_tools:
        return False
    content = response_body.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name") in premium_tools
        for block in content
    )


async def _serve_via_litellm(
    *,
    body: dict[str, Any],
    tier_cfg: dict[str, Any],
    api_key: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Serve a turn through LiteLLM with built-in compression.

    Compresses old tool results, rewrites the body for the tier model,
    converts it to OpenAI format, calls the provider through litellm with only
    the tier key, and converts the result back to Anthropic format. This is the
    provider-agnostic path for every provider without a native Anthropic
    endpoint.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key.
        settings: The config "settings" section (drives compression).

    Returns:
        The routed response in Anthropic format, tagged with the tier model.
    """
    compressed = compression.compress_body(body, settings)
    routed_body = _prepare_routed_body(compressed, tier_cfg)
    kwargs = completion_kwargs(routed_body, api_key)
    litellm = load_litellm()
    raw = await litellm.acompletion(**kwargs)
    return to_anthropic_response(raw, tier_cfg["model"])


async def _serve_via_direct(
    *,
    body: dict[str, Any],
    tier_cfg: dict[str, Any],
    api_key: str,
) -> dict[str, Any] | None:
    """Serve a turn directly on the provider's native Anthropic endpoint.

    Skips LiteLLM conversion and the position-dependent compression pass so
    consecutive tool-result turns keep byte-identical prefixes, letting the
    provider's prefix cache do the work its own caching outperforms our
    compression. A deterministic, position-independent reduction of tool_result
    content runs first: because it is a pure function of the text, the same
    tool_result reduces to identical bytes on every turn, so the prefix stays
    cache-stable while shedding non-informative bytes.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key.

    Returns:
        The routed response in Anthropic format, or None when the direct call
        fails (non-200, transport error) and the turn must pass through.
    """
    routed_body = _prepare_routed_body(reduce.reduce_tool_results(body), tier_cfg)
    return await anthropic_direct.direct_call(body=routed_body, tier_cfg=tier_cfg, api_key=api_key)


async def routed_call(
    *,
    body: dict[str, Any],
    tier_name: str,
    tier_cfg: dict[str, Any],
    api_key: str,
    settings: dict[str, Any],
) -> BackendResponse | None:
    """Serve a request on the tier model, falling back to None on any failure.

    Dispatches on the tier model: providers with a native Anthropic endpoint
    (DeepSeek) are served directly with no compression or LiteLLM conversion;
    every other provider goes through LiteLLM. Escalating responses (premium
    tool calls) are discarded. On success, savings are recorded under the true
    routed model and the served body is then masked to claim the
    client-requested model.

    Args:
        body: Anthropic-format request body from the client.
        tier_name: Tier name, used only for logging.
        tier_cfg: Tier config with "model" and optional "max_tokens".
        api_key: The tier API key; the client's credentials never reach here.
        settings: The config "settings" section.

    Returns:
        A BackendResponse with the masked Anthropic body, or None when the
        caller must replay the turn on premium passthrough.
    """
    premium_model = body.get("model")
    model = tier_cfg.get("model")
    direct = anthropic_direct.direct_endpoint(model) is not None
    try:
        started = time.monotonic()
        if direct:
            anthropic_body = await _serve_via_direct(body=body, tier_cfg=tier_cfg, api_key=api_key)
        else:
            anthropic_body = await _serve_via_litellm(
                body=body, tier_cfg=tier_cfg, api_key=api_key, settings=settings
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
        if escalates(anthropic_body, settings):
            logger.info(
                "acr: tier=%s model=%s escalated to premium (%.2fs)",
                tier_name,
                model,
                elapsed,
            )
            return None
    except Exception as exc:
        logger.warning(
            "acr: tier=%s model=%s routed call failed (%s); passing through",
            tier_name,
            model,
            exc,
        )
        return None

    logger.info(
        "acr: /v1/messages -> tier=%s model=%s%s (%.2fs)",
        tier_name,
        model,
        " [direct]" if direct else "",
        elapsed,
    )
    savings.record_savings_from_response(
        premium_model=premium_model,
        routed_model=cast("str", model),
        response_body=anthropic_body,
        tier_cfg=tier_cfg,
    )
    if isinstance(premium_model, str) and premium_model:
        anthropic_body = {**anthropic_body, "model": premium_model}
    return BackendResponse(body=anthropic_body)
