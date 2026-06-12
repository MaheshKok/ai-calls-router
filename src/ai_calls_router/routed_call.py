"""Routed call engine: serve a tool-result turn on a cheap tier model.

routed_call rewrites the client's Anthropic request for the tier model
(_prepare_routed_body), sends it through litellm with only the tier API key,
and converts the result back to Anthropic format. Responses that invoke a
premium tool are discarded (escalates) so the caller replays the turn on
premium passthrough, savings are recorded under the true routed model before
the served body is masked to the client-requested model, and synthesize_sse
renders the finished response as the Messages SSE stream streaming clients
expect. Every failure path returns None so routing never breaks a turn.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ai_calls_router import compression, savings
from ai_calls_router.conversion import (
    BackendResponse,
    completion_kwargs,
    to_anthropic_response,
)
from ai_calls_router.litellm_guard import load_litellm

logger = logging.getLogger("acr.routed_call")


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

    tier_max = tier_cfg.get("max_tokens")
    if isinstance(tier_max, int) and not isinstance(tier_max, bool) and tier_max > 0:
        requested = body.get("max_tokens")
        if not isinstance(requested, int) or requested > tier_max:
            routed["max_tokens"] = tier_max

    messages = body.get("messages")
    if isinstance(messages, list):
        cleaned: list[Any] = []
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                cleaned.append(msg)
                continue
            blocks = [
                b
                for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") in ("thinking", "redacted_thinking")
                )
            ]
            if not blocks and msg.get("role") == "assistant":
                continue
            cleaned.append({**msg, "content": blocks})
        routed["messages"] = cleaned

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


async def routed_call(
    body: dict[str, Any],
    tier_name: str,
    tier_cfg: dict[str, Any],
    api_key: str,
    settings: dict[str, Any],
) -> BackendResponse | None:
    """Serve a request on the tier model, falling back to None on any failure.

    Compresses old tool results, rewrites the body for the tier model, calls
    the provider through litellm with only the tier API key, and converts the
    result to Anthropic format. Escalating responses (premium tool calls) are
    discarded. On success, savings are recorded under the true routed model
    and the served body is then masked to claim the client-requested model.

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
    try:
        compressed = compression.compress_body(body, settings)
        routed_body = _prepare_routed_body(compressed, tier_cfg)
        kwargs = completion_kwargs(routed_body, api_key)
        litellm = load_litellm()
        started = time.monotonic()
        raw = await litellm.acompletion(**kwargs)
        elapsed = time.monotonic() - started
        anthropic_body = to_anthropic_response(raw, tier_cfg["model"])
        if escalates(anthropic_body, settings):
            logger.info(
                "acr: tier=%s model=%s escalated to premium (%.2fs)",
                tier_name,
                tier_cfg["model"],
                elapsed,
            )
            return None
    except Exception as exc:
        logger.warning(
            "acr: tier=%s model=%s routed call failed (%s); passing through",
            tier_name,
            tier_cfg.get("model"),
            exc,
        )
        return None

    logger.info(
        "acr: /v1/messages -> tier=%s model=%s (%.2fs)",
        tier_name,
        tier_cfg["model"],
        elapsed,
    )
    savings.record_savings_from_response(
        premium_model, tier_cfg["model"], anthropic_body
    )
    if isinstance(premium_model, str) and premium_model:
        anthropic_body = {**anthropic_body, "model": premium_model}
    return BackendResponse(body=anthropic_body)


def _sse_event(name: str, data: dict[str, Any]) -> str:
    """Render one server-sent event.

    Args:
        name: SSE event name.
        data: JSON-serializable event payload.

    Returns:
        The formatted "event:/data:" block including the trailing blank line.
    """
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def synthesize_sse(response_body: dict[str, Any]) -> bytes:
    """Render a finished Anthropic response as a Messages SSE stream.

    Routed calls are buffered (the escalation check needs the complete
    response), so streaming clients receive the result as a synthesized
    stream: message_start, then start/delta/stop per content block, then
    message_delta with the stop reason and output tokens, then message_stop.

    Args:
        response_body: Complete Anthropic-format response body.

    Returns:
        UTF-8 encoded SSE payload.
    """
    usage = response_body.get("usage") or {}
    parts: list[str] = [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": response_body.get("id", "msg_routed"),
                    "type": "message",
                    "role": "assistant",
                    "model": response_body.get("model"),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        )
    ]

    for index, block in enumerate(response_body.get("content") or []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            parts.append(
                _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id", f"toolu_routed_{index}"),
                            "name": block.get("name"),
                            "input": {},
                        },
                    },
                )
            )
            parts.append(
                _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    },
                )
            )
        else:
            parts.append(
                _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            parts.append(
                _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": block.get("text", ""),
                        },
                    },
                )
            )
        parts.append(
            _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        )

    parts.append(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response_body.get("stop_reason", "end_turn"),
                    "stop_sequence": response_body.get("stop_sequence"),
                },
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    parts.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(parts).encode("utf-8")
