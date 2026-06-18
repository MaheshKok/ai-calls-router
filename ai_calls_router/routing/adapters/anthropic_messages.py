"""Implement the Anthropic Messages adapter for Claude Code requests.

Claude Code mostly speaks the internal canonical format, so valid request and response
bodies still pass through by identity. Some Claude clients send system entries inside
messages; the routed path normalizes those into top-level system text without mutating the
original body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_calls_router._lib.anthropic_schemas import validate_anthropic_messages_request
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import synthesis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonObject, JsonValue


def _text_block_value(value: JsonValue) -> str:
    """Return text from a string or Anthropic text block."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or value.get("type") != "text":
        return ""
    text = value.get("text")
    return text if isinstance(text, str) else ""


def _system_text(value: JsonValue) -> str:
    """Extract deterministic text from a Claude system value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)
    parts = [_text_block_value(item) for item in value]
    text_parts = [part for part in parts if part]
    return "\n\n".join(text_parts) if text_parts else str(value)


def _system_message_content(message: JsonValue) -> tuple[bool, JsonValue]:
    """Return whether a message is system-role and its content value."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return False, None
    return True, message.get("content")


def _append_system_text(parts: list[str], value: JsonValue) -> None:
    """Append a non-empty system text value to an accumulator."""
    text = _system_text(value)
    if text:
        parts.append(text)


def _normalize_system_role_messages(body: JsonObject) -> JsonObject:
    """Move Claude `role: system` messages into the top-level system field."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    changed = False
    routed_messages: list[JsonValue] = []
    system_parts: list[str] = []
    existing_system = body.get("system")
    if existing_system is not None:
        _append_system_text(system_parts, existing_system)

    for message in messages:
        is_system, content = _system_message_content(message)
        if is_system:
            changed = True
            if content is not None:
                _append_system_text(system_parts, content)
            continue
        routed_messages.append(message)

    if not changed:
        return body

    routed_body: JsonObject = dict(body)
    routed_body["messages"] = routed_messages
    if system_parts:
        routed_body["system"] = "\n\n".join(system_parts)
    return routed_body


class AnthropicMessagesAdapter:
    """Bridge Anthropic Messages clients to the existing routing engine unchanged."""

    default_agent_group = "claude_code"

    @staticmethod
    def extract_pending_tools(body: JsonObject) -> list[str]:
        """Return pending tool names using the existing Anthropic request parser.

        Args:
            body: Anthropic Messages request body.

        Returns:
            Tool names extracted by the existing routing decision module.
        """
        return routing.pending_tool_names(_normalize_system_role_messages(body))

    @staticmethod
    def to_anthropic_request(body: JsonObject) -> JsonObject:
        """Return the Anthropic request body, normalizing Claude system messages if needed.

        Args:
            body: Anthropic Messages request body.

        Returns:
            The same object for already-valid Messages bodies, or a routed copy
            with `role: system` messages moved to top-level `system`.
        """
        try:
            validate_anthropic_messages_request(body)
        except ValueError:
            routed_body = _normalize_system_role_messages(body)
            if routed_body is body:
                raise
            validate_anthropic_messages_request(routed_body)
            return routed_body
        else:
            return body

    @staticmethod
    def to_client_response(anthropic_response: JsonObject) -> JsonObject:
        """Return the Anthropic response body by identity.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Returns:
            The same object because the client already expects Anthropic Messages.
        """
        return anthropic_response

    @staticmethod
    def to_client_sse(anthropic_response: JsonObject) -> Iterator[bytes]:
        """Yield the existing Anthropic SSE synthesis as a single byte chunk.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Yields:
            The exact bytes produced by the current SSE synthesizer.
        """
        yield synthesis.synthesize_sse(anthropic_response)
