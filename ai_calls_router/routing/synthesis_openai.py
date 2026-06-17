"""SSE synthesis for OpenAI Chat Completions responses.

Routed calls are buffered so the premium-tool escalation guard can inspect the
whole response before serving it. This module renders a completed Anthropic
response as deterministic Chat Completions SSE chunks for streaming clients.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.openai_inbound import anthropic_to_chat_response

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue


def _sse_data(data: JsonObject) -> bytes:
    """Render one OpenAI Chat SSE data block."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _chunk(
    *,
    response_id: JsonValue,
    model: JsonValue,
    delta: JsonObject,
    finish_reason: str | None = None,
) -> JsonObject:
    """Build one chat.completion.chunk object."""
    return cast(
        "JsonObject",
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        },
    )


def _content_delta(text: str) -> JsonObject:
    """Return a Chat content delta."""
    return {"content": text}


def _tool_call_delta(tool_call: JsonObject, index: int) -> JsonObject:
    """Return a Chat tool-call delta."""
    return cast(
        "JsonObject",
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": tool_call.get("id", ""),
                    "type": "function",
                    "function": tool_call.get("function", {}),
                }
            ]
        },
    )


def synthesize_chat_sse(anthropic_body: JsonObject) -> Iterator[bytes]:
    """Yield Chat Completions SSE bytes for an Anthropic response body.

    Args:
        anthropic_body: Complete Anthropic-format routed response.

    Yields:
        UTF-8 encoded SSE ``data:`` blocks, ending with ``data: [DONE]``.
    """
    chat = anthropic_to_chat_response(anthropic_body)
    response_id = chat["id"]
    model = chat["model"]
    choices = chat.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    choice = cast("JsonObject", choices[0])
    message_value = choice.get("message")
    if not isinstance(message_value, dict):
        return
    message = message_value

    yield _sse_data(_chunk(response_id=response_id, model=model, delta={"role": "assistant"}))

    content = message.get("content")
    if isinstance(content, str) and content:
        yield _sse_data(_chunk(response_id=response_id, model=model, delta=_content_delta(content)))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, tool_call in enumerate(cast("JsonArray", tool_calls)):
            if isinstance(tool_call, dict):
                yield _sse_data(
                    _chunk(
                        response_id=response_id,
                        model=model,
                        delta=_tool_call_delta(tool_call, index),
                    )
                )

    yield _sse_data(
        _chunk(
            response_id=response_id,
            model=model,
            delta={},
            finish_reason=(
                str(choice.get("finish_reason"))
                if choice.get("finish_reason") is not None
                else None
            ),
        )
    )
    yield b"data: [DONE]\n\n"
