"""SSE synthesis for OpenAI Responses clients.

Responses streaming uses named semantic events, not Chat Completions chunks, so
this module stays separate from the Phase 3 Chat synthesizer. Event fields
follow OpenAI's streaming guide and the Codex parser's accepted event set.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.responses_inbound import anthropic_to_responses
from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

if TYPE_CHECKING:
    from collections.abc import Iterator

RawEvent = tuple[str, JsonObject]


def _sse_event(name: str, data: JsonObject) -> bytes:
    """Render one Responses SSE event."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _event(event_type: str, sequence_number: int, **fields: JsonValue) -> JsonObject:
    """Build a Responses stream event payload."""
    return {"type": event_type, "sequence_number": sequence_number, **fields}


def _raw_event(event_type: str, **fields: JsonValue) -> RawEvent:
    """Build an event before sequence numbering."""
    return event_type, fields


def _base_response(response: JsonObject, status: str) -> JsonObject:
    """Return the response envelope used by lifecycle events."""
    return {
        "id": response["id"],
        "object": "response",
        "created_at": response["created_at"],
        "status": status,
        "model": response["model"],
        "output": [] if status != response["status"] else response["output"],
        "usage": None if status != response["status"] else response["usage"],
    }


def _item_id(item: JsonObject, index: int) -> str:
    """Return a stable item identifier."""
    existing = item.get("id")
    if isinstance(existing, str) and existing:
        return existing
    return f"msg_routed_{index}" if item.get("type") == "message" else f"fc_routed_{index}"


def _message_events(
    *,
    item: JsonObject,
    output_index: int,
) -> Iterator[RawEvent]:
    """Yield SSE payloads for one assistant message output item."""
    item_id = _item_id(item, output_index)
    text = ""
    content = item.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = str(content[0].get("text", ""))
    added_item = cast(
        "JsonObject", {"id": item_id, "type": "message", "role": "assistant", "content": []}
    )
    part = cast("JsonObject", {"type": "output_text", "text": ""})
    final_part = cast("JsonObject", {"type": "output_text", "text": text})
    final_item = cast("JsonObject", {**added_item, "content": [final_part]})

    yield _raw_event(
        "response.output_item.added",
        output_index=output_index,
        item=added_item,
    )
    yield _raw_event(
        "response.content_part.added",
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        part=part,
    )
    if text:
        yield _raw_event(
            "response.output_text.delta",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            delta=text,
        )
    yield _raw_event(
        "response.output_text.done",
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        text=text,
    )
    yield _raw_event(
        "response.content_part.done",
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        part=final_part,
    )
    yield _raw_event(
        "response.output_item.done",
        output_index=output_index,
        item=final_item,
    )


def _function_call_events(
    *,
    item: JsonObject,
    output_index: int,
) -> Iterator[RawEvent]:
    """Yield SSE payloads for one function_call output item."""
    item_id = _item_id(item, output_index)
    call_id = str(item.get("call_id", item_id))
    name = str(item.get("name", ""))
    arguments = str(item.get("arguments", ""))
    added_item = cast(
        "JsonObject",
        {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": "",
        },
    )
    final_item = {**added_item, "arguments": arguments}
    yield _raw_event(
        "response.output_item.added",
        output_index=output_index,
        item=added_item,
    )
    if arguments:
        yield _raw_event(
            "response.function_call_arguments.delta",
            item_id=item_id,
            output_index=output_index,
            call_id=call_id,
            name=name,
            delta=arguments,
        )
    yield _raw_event(
        "response.function_call_arguments.done",
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
    yield _raw_event(
        "response.output_item.done",
        output_index=output_index,
        item=final_item,
    )


def _output_events(response: JsonObject) -> Iterator[RawEvent]:
    """Yield stream events for every final output item."""
    output = response.get("output")
    if not isinstance(output, list):
        return
    for output_index, item in enumerate(cast("JsonArray", output)):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            yield from _message_events(
                item=item,
                output_index=output_index,
            )
        elif item.get("type") == "function_call":
            yield from _function_call_events(
                item=item,
                output_index=output_index,
            )


def _numbered_events(response: JsonObject) -> Iterator[RawEvent]:
    """Yield all stream events with monotonically increasing sequence numbers."""
    events = [
        _raw_event("response.created", response=_base_response(response, "in_progress")),
        _raw_event("response.in_progress", response=_base_response(response, "in_progress")),
        *_output_events(response),
        _raw_event("response.completed", response=response),
    ]
    for sequence_number, (name, fields) in enumerate(events):
        yield name, _event(name, sequence_number, **fields)


def synthesize_responses_sse(anthropic_body: JsonObject, model: str) -> Iterator[bytes]:
    """Yield Responses API SSE bytes for an Anthropic response body."""
    response = anthropic_to_responses(anthropic_body, model)
    for name, payload in _numbered_events(response):
        yield _sse_event(name, payload)


def synthesize_response_object_sse(response: JsonObject) -> Iterator[bytes]:
    """Yield Responses API SSE bytes for an already-assembled response object."""
    for name, payload in _numbered_events(response):
        yield _sse_event(name, payload)
