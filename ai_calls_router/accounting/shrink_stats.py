"""Read-only measurement of how much a shrink pass removed from a request body.

This module counts tool-output characters in a request body and packages a
before/after measurement (``ShrinkStats``) so the engine can report compression
savings to logs, the savings ledger, and the metrics snapshot without changing
what is sent. The router no longer runs a shrink pass of its own on native
direct paths -- token reduction is delegated to the upstream Headroom layer --
so those paths usually feed an unchanged body as both before and after and the
stats report a no-op (path "none", zero chars saved). Counting walks Anthropic
tool_result and OpenAI Responses function_call_output shapes, never mutates the
body, and treats any non-tool-output content as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue

# Average characters per token. A coarse, model-agnostic divisor used only to
# turn a measured character delta into an order-of-magnitude token estimate; the
# authoritative savings figure is always the character count, not this estimate.
DEFAULT_CHARS_PER_TOKEN = 3.5


def _content_chars(content: JsonValue) -> int:
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
            len(cast("str", block["text"]))
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return 0


def _message_tool_result_chars(message: JsonValue) -> int:
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


def _responses_tool_output_chars(item: JsonValue) -> int:
    """Count OpenAI Responses tool-output characters in one input item."""
    if not isinstance(item, dict):
        return 0
    if item.get("type") not in {"function_call_output", "custom_tool_call_output"}:
        return 0
    return _content_chars(item.get("output"))


def _responses_input_tool_output_chars(body: JsonObject) -> int:
    """Count OpenAI Responses tool-output characters in a request body."""
    input_value = body.get("input")
    if not isinstance(input_value, list):
        return 0
    return sum(_responses_tool_output_chars(item) for item in input_value)


def tool_result_chars(body: JsonObject) -> int:
    """Sum the character length of every tool-output text in a request body.

    Walks Anthropic ``messages`` and OpenAI Responses ``input`` and counts only
    tool-output content -- the bytes the shrink passes target. Non-tool-output
    content, malformed messages, and missing containers all contribute 0. The
    body is never mutated.

    Args:
        body: Anthropic-format request body.

    Returns:
        Total tool_result characters in the body (0 when there are none).
    """
    messages = body.get("messages")
    message_chars = (
        sum(_message_tool_result_chars(message) for message in messages)
        if isinstance(messages, list)
        else 0
    )
    return message_chars + _responses_input_tool_output_chars(body)


@dataclass(frozen=True)
class ShrinkStats:
    """Before/after tool_result measurement for one shrink pass.

    Attributes:
        path: Which shrink pass produced the measurement -- ``"reduce"``,
            ``"compress"``, or ``"none"`` when no pass ran.
        chars_before: tool_result characters before the pass.
        chars_after: tool_result characters after the pass.
        content_types: Distinct content-type labels reduced from headroom's own
            ``router:*`` routing markers for this request (e.g. ``("excluded",)``
            or ``("smart_crusher", "kompress")``). Empty when no classification
            was captured (native no-op paths, headroom absent). This is captured
            metadata only -- it never influences what is sent upstream.
    """

    path: str
    chars_before: int
    chars_after: int
    content_types: tuple[str, ...] = ()

    @property
    def content_type_label(self) -> str:
        """Comma-joined content-type label for the metrics DB column.

        Returns:
            ``", ".join(self.content_types)`` -- the empty string when no
            content types were captured.
        """
        return ", ".join(self.content_types)

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

    def est_tokens_before(self, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
        """Estimate tool_result tokens before the pass.

        Args:
            chars_per_token: Positive characters per token divisor.

        Returns:
            ``chars_before / chars_per_token`` truncated to an int, or 0 when the
            divisor is not positive.
        """
        if chars_per_token <= 0:
            return 0
        return int(self.chars_before / chars_per_token)

    def est_tokens_after(self, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
        """Estimate tool_result tokens after the pass.

        Args:
            chars_per_token: Positive characters per token divisor.

        Returns:
            ``chars_after / chars_per_token`` truncated to an int, or 0 when the
            divisor is not positive.
        """
        if chars_per_token <= 0:
            return 0
        return int(self.chars_after / chars_per_token)


def compute_shrink(
    *,
    path: str,
    before: JsonObject,
    after: JsonObject,
    content_types: tuple[str, ...] = (),
) -> ShrinkStats:
    """Measure the tool_result shrink between a body and its shrunk form.

    Args:
        path: Label for the shrink pass that produced ``after``.
        before: Request body before the shrink pass.
        after: Request body returned by the shrink pass (may be ``before`` itself
            when the pass was a no-op).
        content_types: Content-type labels captured from headroom's routing
            markers to attach to the measurement (empty by default).

    Returns:
        A ShrinkStats with the tool_result character counts of both bodies.
    """
    return ShrinkStats(
        path=path,
        chars_before=tool_result_chars(before),
        chars_after=tool_result_chars(after),
        content_types=content_types,
    )
