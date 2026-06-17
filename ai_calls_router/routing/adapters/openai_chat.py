"""Client adapter for OpenAI Chat Completions requests.

Hermes can speak OpenAI Chat Completions to the proxy while the routing core
uses Anthropic Messages internally. This adapter holds no state; it delegates
edge conversion to pure helpers and exposes the shared ClientAdapter methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.openai_inbound import (
    anthropic_to_chat_response,
    chat_request_to_anthropic,
)
from ai_calls_router.routing.synthesis_openai import synthesize_chat_sse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue


def _has_role(value: JsonValue, role: str) -> bool:
    """Return whether a JSON message has the requested role."""
    return isinstance(value, dict) and value.get("role") == role


def _last_tool_run_start(messages: JsonArray) -> int | None:
    """Return the start index of the final role=tool run, if present."""
    index = len(messages) - 1
    if index < 0 or not _has_role(messages[index], "tool"):
        return None
    while index > 0 and _has_role(messages[index - 1], "tool"):
        index -= 1
    return index


def _assistant_tool_id_map(messages: JsonArray, stop: int) -> dict[str, str]:
    """Build a tool_call id to function-name map before ``stop``."""
    id_to_name: dict[str, str] = {}
    for message in messages[:stop]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            resolved = _tool_call_id_and_name(tool_call)
            if resolved is not None:
                tool_id, name = resolved
                id_to_name[tool_id] = name
    return id_to_name


def _tool_call_id_and_name(tool_call: JsonValue) -> tuple[str, str] | None:
    """Return a tool call id/name pair when both are present."""
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    tool_id = tool_call.get("id")
    name = function.get("name")
    if not tool_id or not name:
        return None
    return str(tool_id), str(name)


class OpenAIChatAdapter:
    """Bridge OpenAI Chat Completions clients to the routing engine."""

    default_agent_group = "hermes"

    def extract_pending_tools(self, body: JsonObject) -> list[str]:
        """Return pending tool names from the final Chat tool-result run.

        Args:
            body: OpenAI Chat Completions request body.

        Returns:
            Ordered, deduplicated tool names; ``["<unknown>"]`` when any
            final tool result cannot be resolved to a prior assistant call.
        """
        messages = body.get("messages")
        if not isinstance(messages, list):
            return []
        message_items = cast("JsonArray", messages)
        start = _last_tool_run_start(message_items)
        if start is None:
            return []

        id_to_name = _assistant_tool_id_map(message_items, start)
        names: list[str] = []
        for message in message_items[start:]:
            if not isinstance(message, dict):
                continue
            tool_call_id = message.get("tool_call_id")
            name = id_to_name.get(str(tool_call_id))
            if name is None:
                return ["<unknown>"]
            if name not in names:
                names.append(name)
        return names

    def to_anthropic_request(self, body: JsonObject) -> JsonObject:
        """Convert a Chat request to the Anthropic canonical format.

        Args:
            body: OpenAI Chat Completions request body.

        Returns:
            New Anthropic-format request body.
        """
        return chat_request_to_anthropic(body)

    def to_client_response(self, anthropic_response: JsonObject) -> JsonObject:
        """Convert an Anthropic response to a Chat response.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Returns:
            OpenAI Chat Completions response body.
        """
        return anthropic_to_chat_response(anthropic_response)

    def to_client_sse(self, anthropic_response: JsonObject) -> Iterator[bytes]:
        """Yield OpenAI Chat Completions SSE chunks.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Yields:
            UTF-8 encoded Chat Completions SSE chunks.
        """
        yield from synthesize_chat_sse(anthropic_response)
