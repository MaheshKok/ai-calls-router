"""Client adapter for OpenAI Responses requests.

Codex and compatible clients speak the Responses wire format while the router
core serves Anthropic Messages. This adapter holds no state and delegates all
format work to pure conversion and synthesis helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.openai_schemas import validate_responses_request
from ai_calls_router._lib.responses_inbound import (
    anthropic_to_responses,
    responses_request_to_anthropic,
)
from ai_calls_router.routing.synthesis_responses import synthesize_responses_sse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

_OUTPUT_ITEM_TYPES: frozenset[str] = frozenset({"function_call_output", "custom_tool_call_output"})
_CALL_ITEM_TYPES: frozenset[str] = frozenset({"function_call", "custom_tool_call"})


def _responses_input_items(body: JsonObject) -> JsonArray:
    """Return Responses input as a list for pending-tool extraction."""
    input_items = body.get("input")
    if isinstance(input_items, list):
        return cast("JsonArray", input_items)
    return []


def _last_output_run_start(items: JsonArray) -> int | None:
    """Return the start index of the final tool-output run."""
    index = len(items) - 1
    if index < 0 or not _is_tool_output_item(items[index]):
        return None
    while index > 0 and _is_tool_output_item(items[index - 1]):
        index -= 1
    return index


def _is_tool_output_item(item: JsonValue) -> bool:
    """Return whether an item is a routable tool output."""
    return isinstance(item, dict) and item.get("type") in _OUTPUT_ITEM_TYPES


def _call_id_to_name(items: JsonArray, stop: int) -> dict[str, str]:
    """Build call_id to tool-name map before ``stop``."""
    mapping: dict[str, str] = {}
    for item in items[:stop]:
        if not isinstance(item, dict) or item.get("type") not in _CALL_ITEM_TYPES:
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        if call_id and name:
            mapping[str(call_id)] = str(name)
    return mapping


class OpenAIResponsesAdapter:
    """Bridge OpenAI Responses clients to the routing engine."""

    default_agent_group = "hermes"

    def extract_pending_tools(self, body: JsonObject) -> list[str]:
        """Return pending tool names from the final Responses output run."""
        items = _responses_input_items(body)
        start = _last_output_run_start(items)
        if start is None:
            return []
        id_to_name = _call_id_to_name(items, start)
        names: list[str] = []
        for item in items[start:]:
            if not isinstance(item, dict):
                continue
            name = id_to_name.get(str(item.get("call_id")))
            if name is None:
                return ["<unknown>"]
            if name not in names:
                names.append(name)
        return names

    def to_anthropic_request(self, body: JsonObject) -> JsonObject:
        """Convert a Responses request to the Anthropic canonical format."""
        validate_responses_request(body)
        return responses_request_to_anthropic(body)

    def to_client_response(self, anthropic_response: JsonObject) -> JsonObject:
        """Convert an Anthropic response to a Responses response."""
        return anthropic_to_responses(anthropic_response, str(anthropic_response.get("model", "")))

    def to_client_sse(self, anthropic_response: JsonObject) -> Iterator[bytes]:
        """Yield OpenAI Responses SSE chunks."""
        yield from synthesize_responses_sse(
            anthropic_response,
            str(anthropic_response.get("model", "")),
        )
