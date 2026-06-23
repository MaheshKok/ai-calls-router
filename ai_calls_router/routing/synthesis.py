"""SSE synthesis for buffered routed responses.

Routed calls are served buffered because the escalation check needs the
complete response before it can be returned to the client. Streaming clients,
however, expect the Anthropic Messages SSE event sequence, so synthesize_sse
renders a finished response body as that stream: message_start, then
start/delta/stop per content block, then message_delta with the stop reason and
output tokens, then message_stop. This module is pure data transformation with
no network or routing dependencies.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject


def _sse_event(name: str, data: JsonObject) -> str:
    """Render one server-sent event.

    Args:
        name: SSE event name.
        data: JSON-serializable event payload.

    Returns:
        The formatted "event:/data:" block including the trailing blank line.
    """
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def synthesize_sse(response_body: JsonObject) -> bytes:
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
    usage_raw = response_body.get("usage")
    usage: JsonObject = usage_raw if isinstance(usage_raw, dict) else {}
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

    content = response_body.get("content")
    blocks: JsonArray = cast("JsonArray", content) if isinstance(content, list) else []
    for index, block in enumerate(blocks):
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
