"""Deterministic, position-independent reduction of tool_result content.

This module strips non-informative bytes from tool_result payloads in a routed
request body so the cheap tier processes fewer tokens. Unlike
``compression.compress_body`` the reduction depends only on the text itself --
never on a message's position in the conversation -- so the same tool_result
reduces to byte-identical output on every turn and the provider's prefix cache
keeps hitting. Every transform is a pure function and the input body is never
mutated; an unchanged body is returned unchanged so callers can rely on identity
for the no-op case.
"""

from __future__ import annotations

import re
from typing import Any

# ANSI CSI escape sequence: ESC '[', parameter bytes (0x30-0x3F), intermediate
# bytes (0x20-0x2F), then a final byte (0x40-0x7E). Covers SGR colour codes plus
# cursor/clear sequences emitted by terminal-oriented tools. Removing them is
# lossless: they carry no textual content.
_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def reduce_text(text: str, *, drop_duplicate_lines: bool = False) -> str:
    """Strip non-informative bytes from a tool-output string, deterministically.

    Always removes ANSI CSI escape sequences and collapses runs of blank lines
    to a single blank line -- both lossless. When ``drop_duplicate_lines`` is
    enabled it additionally drops consecutive duplicate lines; that transform is
    opt-in because adjacent identical lines occur in real source (e.g. nested
    closing braces), so it is only safe for log/progress output, not code reads.
    The result is a pure function of the arguments, so identical input always
    yields byte-identical output.

    Args:
        text: Raw tool-output text.
        drop_duplicate_lines: When True, also drop consecutive duplicate lines.

    Returns:
        The reduced text; identical to the input when there is nothing to strip.
    """
    stripped = _ANSI_CSI.sub("", text)
    out: list[str] = []
    blank_run = 0
    previous: str | None = None
    for line in stripped.split("\n"):
        if line.strip():
            blank_run = 0
        else:
            blank_run += 1
            if blank_run > 1:
                continue
        if drop_duplicate_lines and line == previous:
            continue
        out.append(line)
        previous = line
    return "\n".join(out)


def _reduce_content(content: Any) -> tuple[Any, bool]:
    """Reduce one tool_result content value.

    Args:
        content: The tool_result block's content (string, text-block list, or
            any other shape).

    Returns:
        A (value, changed) pair; value is the reduced content and changed is
        True only when the reduction altered it.
    """
    if isinstance(content, str):
        reduced = reduce_text(content)
        return (reduced, reduced != content)
    if isinstance(content, list):
        return _reduce_text_blocks(content)
    return (content, False)


def _reduce_text_blocks(blocks: list[Any]) -> tuple[list[Any], bool]:
    """Reduce the text blocks inside a tool_result list content.

    Non-text and malformed blocks pass through untouched; only ``type == text``
    blocks with a string ``text`` are reduced.

    Args:
        blocks: The list content of a tool_result block.

    Returns:
        A (blocks, changed) pair sharing original block objects where unchanged.
    """
    new_blocks: list[Any] = []
    changed = False
    for block in blocks:
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            reduced = reduce_text(block["text"])
            if reduced != block["text"]:
                new_blocks.append({**block, "text": reduced})
                changed = True
                continue
        new_blocks.append(block)
    return (new_blocks, changed)


def _reduce_message(message: Any) -> tuple[Any, bool]:
    """Reduce every tool_result block in one message.

    Args:
        message: A message from the request body (any shape; non-dicts and
            non-list content pass through untouched).

    Returns:
        A (message, changed) pair; the original message object is returned when
        nothing changed, otherwise a new message with reduced content.
    """
    if not isinstance(message, dict):
        return (message, False)
    content = message.get("content")
    if not isinstance(content, list):
        return (message, False)
    new_content: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            reduced, block_changed = _reduce_content(block.get("content"))
            if block_changed:
                new_content.append({**block, "content": reduced})
                changed = True
                continue
        new_content.append(block)
    if not changed:
        return (message, False)
    return ({**message, "content": new_content}, True)


def reduce_tool_results(body: dict[str, Any]) -> dict[str, Any]:
    """Reduce every tool_result block in a request body, position-independently.

    Walks all messages with no recency window and replaces each tool_result's
    content with its reduced form. The input body is never mutated (new objects
    are built only along the changed path, unchanged objects are shared); when
    no tool_result changes, the original body object is returned so the caller
    can rely on identity for the no-op case.

    Args:
        body: Anthropic-format request body.

    Returns:
        A reduced copy of the body, or the original body when nothing changed.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    new_messages: list[Any] = []
    changed = False
    for message in messages:
        new_message, message_changed = _reduce_message(message)
        new_messages.append(new_message)
        changed = changed or message_changed
    return {**body, "messages": new_messages} if changed else body
