"""Read-only measurement of how much a shrink pass removed from a request body.

The two routing shrink passes -- ``reduce.reduce_tool_results`` (DeepSeek direct
path) and ``compression.compress_body`` (LiteLLM path) -- are pure functions that
map a request body to a smaller body. This module counts the tool_result
characters in a body and packages a before/after measurement (``ShrinkStats``)
so the engine can report compression savings to logs, the savings ledger, and
the metrics snapshot without changing what is sent. Counting walks the Anthropic
body shape, never mutates it, and treats any non-tool_result content as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Average characters per token. A coarse, model-agnostic divisor used only to
# turn a measured character delta into an order-of-magnitude token estimate; the
# authoritative savings figure is always the character count, not this estimate.
DEFAULT_CHARS_PER_TOKEN = 3.5


def _content_chars(content: Any) -> int:
    """Count characters in one tool_result content value.

    Args:
        content: A tool_result block's content (string, text-block list, or any
            other shape).

    Returns:
        The character length of string content, the summed length of every
        ``type == text`` block in list content, or 0 for any other shape.
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(block["text"])
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return 0


def _message_tool_result_chars(message: Any) -> int:
    """Count tool_result characters in one message.

    Args:
        message: A message from a request body; non-dict messages and non-list
            content contribute 0.

    Returns:
        The summed character length of every tool_result block in the message.
    """
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        _content_chars(block.get("content"))
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )


def tool_result_chars(body: dict[str, Any]) -> int:
    """Sum the character length of every tool_result text in a request body.

    Walks all messages and counts only tool_result content -- the bytes the
    shrink passes target. Non-tool_result content, malformed messages, and a
    missing or non-list ``messages`` field all contribute 0. The body is never
    mutated.

    Args:
        body: Anthropic-format request body.

    Returns:
        Total tool_result characters in the body (0 when there are none).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(_message_tool_result_chars(message) for message in messages)


@dataclass(frozen=True)
class ShrinkStats:
    """Before/after tool_result measurement for one shrink pass.

    Attributes:
        path: Which shrink pass produced the measurement -- ``"reduce"``,
            ``"compress"``, or ``"none"`` when no pass ran.
        chars_before: tool_result characters before the pass.
        chars_after: tool_result characters after the pass.
    """

    path: str
    chars_before: int
    chars_after: int

    @property
    def chars_saved(self) -> int:
        """Characters removed by the pass, floored at 0.

        Returns:
            ``chars_before - chars_after`` when positive, else 0 (a pass never
            legitimately grows tool_result content, so a negative delta is
            reported as no savings rather than negative savings).
        """
        return max(0, self.chars_before - self.chars_after)

    @property
    def ratio(self) -> float:
        """Fraction of tool_result characters removed.

        Returns:
            ``chars_saved / chars_before`` in [0.0, 1.0], or 0.0 when there were
            no tool_result characters to shrink.
        """
        if self.chars_before <= 0:
            return 0.0
        return self.chars_saved / self.chars_before

    def est_tokens_saved(self, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
        """Estimate tokens saved from the character delta.

        This is a coarse estimate, not a tokenizer count; the character figures
        are the authoritative savings record.

        Args:
            chars_per_token: Average characters per token divisor.

        Returns:
            ``chars_saved / chars_per_token`` truncated to an int, or 0 when the
            divisor is not positive.
        """
        if chars_per_token <= 0:
            return 0
        return int(self.chars_saved / chars_per_token)


def compute_shrink(*, path: str, before: dict[str, Any], after: dict[str, Any]) -> ShrinkStats:
    """Measure the tool_result shrink between a body and its shrunk form.

    Args:
        path: Label for the shrink pass that produced ``after``.
        before: Request body before the shrink pass.
        after: Request body returned by the shrink pass (may be ``before`` itself
            when the pass was a no-op).

    Returns:
        A ShrinkStats with the tool_result character counts of both bodies.
    """
    return ShrinkStats(
        path=path,
        chars_before=tool_result_chars(before),
        chars_after=tool_result_chars(after),
    )
