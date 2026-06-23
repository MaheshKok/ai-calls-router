"""Strip blank text blocks from Anthropic message content.

The Anthropic Messages API rejects any text content block whose text is empty or
whitespace-only ("messages: text content blocks must be non-empty"). A routed
tier model can emit such a block ahead of a tool_use; left untouched it poisons
the client's persisted history and 400s every later turn that replays it. These
helpers remove blank text blocks from routed responses (so the proxy never emits
one) and from outbound request bodies (so an already-poisoned history still
serves). Every function returns new objects and never mutates its input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue


def _is_blank_text_block(block: JsonValue) -> bool:
    """Return whether a content block is a text block with no visible text."""
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and not str(block.get("text", "")).strip()
    )


def _strip_blank_text_blocks(content: JsonArray) -> JsonArray:
    """Return the content list without its blank text blocks."""
    return [block for block in content if not _is_blank_text_block(block)]


def _clean_message(message: JsonValue) -> JsonValue:
    """Return a message copy without blank text blocks, or the original.

    The original is returned unchanged when the message has no list content, no
    blank text block, or when stripping would empty its content (an empty content
    array is itself invalid, so the blank block is left for upstream to reject).
    """
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if not isinstance(content, list):
        return message
    stripped = _strip_blank_text_blocks(content)
    if len(stripped) == len(content) or not stripped:
        return message
    return {**message, "content": stripped}


def clean_response_content(response: JsonObject) -> JsonObject:
    """Return a response copy whose content has no blank text blocks.

    Leaves the response object unchanged when its content is not a list or
    carries no blank text block, so already-valid responses pass through without
    a copy.

    Args:
        response: Anthropic-format response body.

    Returns:
        The original response, or a copy with blank text blocks removed.
    """
    content = response.get("content")
    if not isinstance(content, list):
        return response
    stripped = _strip_blank_text_blocks(content)
    if len(stripped) == len(content):
        return response
    return {**response, "content": stripped}


def clean_request_messages(body: JsonObject) -> JsonObject | None:
    """Return a body copy whose message content has no blank text blocks.

    Returns None when nothing changed so the caller can forward the original
    bytes untouched, preserving the upstream prompt cache for valid bodies. A
    message that stripping would empty is left intact (see :func:`_clean_message`).

    Args:
        body: Anthropic-format request body.

    Returns:
        A copy with blank text blocks removed, or None when no message changed.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    new_messages: JsonArray = []
    changed = False
    for message in messages:
        cleaned = _clean_message(message)
        if cleaned is not message:
            changed = True
        new_messages.append(cleaned)
    if not changed:
        return None
    return {**body, "messages": new_messages}
