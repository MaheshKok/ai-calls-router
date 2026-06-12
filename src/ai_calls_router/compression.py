"""Built-in lightweight compression for routed request bodies.

compress_body shrinks the token footprint of a routed call by truncating
tool_result text in messages older than the recent-message window; the recent
window and all non-tool_result content stay byte-identical, and the input body
is never mutated because the passthrough fallback still needs it. An optional
rtk (Rust Token Killer) pass is applied first to old tool results whose
producing tool has a known-safe rtk filter; rtk is detected on PATH, bounded
by a timeout, and fail-open -- plain truncation always runs regardless.
"""

from __future__ import annotations

import copy
import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger("acr.compression")

DEFAULT_KEEP_RECENT_MESSAGES = 6
DEFAULT_MAX_TOOL_RESULT_CHARS = 4000
RTK_TIMEOUT_SECONDS = 5

# Tool name -> rtk pipe filter, only where the tool's output format matches
# the filter's expected input by construction. rtk filters are format-specific
# (pytest, grep, git-log, ...) and destroy arbitrary text, so unmapped tools
# never reach rtk and fall back to plain truncation.
RTK_FILTERS: dict[str, str] = {"Grep": "grep"}


def run_rtk(text: str, filter_name: str) -> str | None:
    """Filter text through an rtk pipe subprocess, fail-open.

    Args:
        text: Raw tool output text to compress.
        filter_name: rtk pipe filter name (e.g. "grep").

    Returns:
        rtk stdout when the subprocess succeeds with non-blank output,
        otherwise None (non-zero exit, timeout, missing binary, any error).
    """
    try:
        proc = subprocess.run(
            ["rtk", "pipe", "--filter", filter_name],
            input=text,
            capture_output=True,
            text=True,
            timeout=RTK_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return proc.stdout
    except Exception as exc:
        logger.warning("rtk pipe failed (filter=%s): %s", filter_name, exc)
        return None


def _positive_int(value: Any, default: int) -> int:
    """Return value when it is a positive int, otherwise the default.

    Args:
        value: Candidate config value (bool excluded).
        default: Fallback for missing or malformed values.

    Returns:
        A usable positive integer setting.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _tool_names_by_id(messages: list[Any]) -> dict[str, str]:
    """Map tool_use ids to tool names across all assistant messages.

    Args:
        messages: Anthropic-format message list (malformed entries skipped).

    Returns:
        Mapping of tool_use_id to the tool name that produced it.
    """
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            block_id = block.get("id")
            name = block.get("name")
            if isinstance(block_id, str) and isinstance(name, str):
                names[block_id] = name
    return names


def _truncate(text: str, budget: int) -> str:
    """Cut text to a character budget with an explicit truncation marker.

    Args:
        text: Text longer than the budget.
        budget: Number of leading characters to keep.

    Returns:
        The kept prefix plus a marker reporting how many chars were removed.
    """
    removed = len(text) - budget
    return text[:budget] + f"\n...[truncated {removed} chars]"


def _compress_text(text: str, budget: int, rtk_filter: str | None) -> str:
    """Compress one over-budget text payload, optionally via rtk first.

    Args:
        text: Tool result text exceeding the budget.
        budget: Character budget for the final text.
        rtk_filter: rtk pipe filter to try first, or None to skip rtk.

    Returns:
        rtk-filtered and/or truncated text, always within budget plus marker.
    """
    if rtk_filter is not None:
        filtered = run_rtk(text, rtk_filter)
        if filtered is not None:
            text = filtered
    if len(text) <= budget:
        return text
    return _truncate(text, budget)


def _compress_tool_result_content(
    content: Any, budget: int, rtk_filter: str | None
) -> Any:
    """Compress a tool_result content payload to the character budget.

    String content is compressed directly; list content shares one budget
    across its text blocks while non-text and malformed blocks pass through
    untouched. Any other content shape is returned as-is.

    Args:
        content: The tool_result block's content value.
        budget: Character budget for the whole tool result.
        rtk_filter: rtk pipe filter to try on over-budget text, or None.

    Returns:
        The compressed content, same shape as the input.
    """
    if isinstance(content, str):
        if len(content) <= budget:
            return content
        return _compress_text(content, budget, rtk_filter)
    if isinstance(content, list):
        remaining = budget
        new_blocks: list[Any] = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else None
            if not isinstance(text, str) or block.get("type") != "text":
                new_blocks.append(block)
                continue
            if len(text) <= remaining:
                new_blocks.append(block)
                remaining -= len(text)
                continue
            new_blocks.append({**block, "text": _compress_text(text, remaining, rtk_filter)})
            remaining = 0
        return new_blocks
    return content


def _compress_message(
    message: Any, budget: int, tool_names: dict[str, str], rtk_enabled: bool
) -> None:
    """Compress every tool_result block in one (already copied) message.

    Args:
        message: A message from the deep-copied body; mutated in place.
        budget: Character budget per tool_result block.
        tool_names: tool_use_id to tool name mapping for rtk filter lookup.
        rtk_enabled: Whether rtk is available and allowed by config.
    """
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        rtk_filter = None
        if rtk_enabled:
            tool_use_id = block.get("tool_use_id")
            tool_name = tool_names.get(tool_use_id) if isinstance(tool_use_id, str) else None
            rtk_filter = RTK_FILTERS.get(tool_name) if tool_name is not None else None
        block["content"] = _compress_tool_result_content(
            block.get("content"), budget, rtk_filter
        )


def compress_body(body: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Compress old tool_result content in a request body for routing.

    The last keep_recent_messages messages are preserved byte-identical;
    tool_result text in older messages is truncated to max_tool_result_chars
    (after an optional rtk pass for tools with a safe filter mapping). The
    input body is never mutated. Compression is an optimization, never a
    gate: any error returns the original body unchanged.

    Args:
        body: Anthropic-format request body.
        settings: The config "settings" section; reads compress_routed and
            the compression sub-mapping (keep_recent_messages,
            max_tool_result_chars, use_rtk: auto|never).

    Returns:
        A compressed copy of the body, or the original body object when
        compression is disabled, inapplicable, or fails.
    """
    try:
        if not settings.get("compress_routed", True):
            return body
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body
        cfg = settings.get("compression")
        cfg = cfg if isinstance(cfg, dict) else {}
        keep_recent = _positive_int(
            cfg.get("keep_recent_messages"), DEFAULT_KEEP_RECENT_MESSAGES
        )
        budget = _positive_int(
            cfg.get("max_tool_result_chars"), DEFAULT_MAX_TOOL_RESULT_CHARS
        )
        cutoff = len(messages) - keep_recent
        if cutoff <= 0:
            return body
        rtk_enabled = cfg.get("use_rtk", "auto") == "auto" and shutil.which("rtk") is not None
        tool_names = _tool_names_by_id(messages)
        out = copy.deepcopy(body)
        for message in out["messages"][:cutoff]:
            _compress_message(message, budget, tool_names, rtk_enabled)
        return out
    except Exception as exc:
        logger.warning("compression failed, using original body: %s", exc)
        return body
