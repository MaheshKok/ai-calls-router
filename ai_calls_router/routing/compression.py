"""Optional headroom compression for the LiteLLM serving path.

Compresses the OpenAI-format messages that litellm sends upstream, after the
Anthropic-to-OpenAI conversion in completion_kwargs. That is the wire shape
headroom's content router is effective on: tool output lives in role="tool"
messages with string content, so the list-form and user-message protection gates
that zero out compression on native Anthropic traffic do not fire. Headroom's
default tool exclusions still leave coding-agent output (Read/Edit/Bash/...)
verbatim. Compression is best-effort: when headroom-ai is not installed, or a
compression call raises, the messages pass through unchanged so a proxied request
is never broken by the optimizer.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Protocol, cast

from ai_calls_router.accounting.shrink_stats import ShrinkStats

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray

logger = logging.getLogger("acr.compression")

DEFAULT_MODEL_LIMIT = 200_000


class _CompressResult(Protocol):
    """Headroom result shape used by this module."""

    messages: JsonArray


class _Compressor(Protocol):
    """Headroom compression callable shape used by this module."""

    def __call__(
        self,
        messages: JsonArray,
        *,
        model: str,
        model_limit: int,
        optimize: bool,
        **kwargs: object,
    ) -> _CompressResult: ...


# Sentinel passed to headroom as ``kompress_model`` to skip its ML (Kompress)
# plain-text compressor. Lossless compressors (SmartCrusher, code, log, diff,
# search, html) still run; only the lossy ModernBERT token-dropping stage is off.
_DISABLE_TEXT_ML = "disabled"

# Marker segments whose strategy/label is the second segment, not the first
# (``router:tool_result:smart_crusher`` classifies as ``smart_crusher``).
_NESTED_MARKER_HEADS = frozenset({"tool_result", "text_block"})


def _strategy_token(marker: str) -> str:
    """Reduce one headroom ``router:*`` marker to its strategy/label token.

    Args:
        marker: A routing marker such as ``"router:smart_crusher:0.34"``,
            ``"router:excluded:tool"``, or ``"router:tool_result:kompress"``.

    Returns:
        The strategy or protection label (``"smart_crusher"``, ``"excluded"``,
        ``"kompress"``, ...), or ``""`` when the marker has no label segment.
    """
    parts = marker.split(":")
    rest = parts[1:] if parts and parts[0] == "router" else parts
    if not rest:
        return ""
    if rest[0] in _NESTED_MARKER_HEADS and len(rest) >= 2:
        return rest[1]
    return rest[0]


def summarize_content_types(markers: list[str]) -> tuple[str, ...]:
    """Reduce headroom's ``router:*`` markers to distinct content-type labels.

    This surfaces headroom's own per-message classification for the dashboard; it
    never re-derives the type from tool names or shrink path. Labels are deduped
    while preserving first-seen order so the column reads left-to-right in the
    order headroom applied them.

    Args:
        markers: Routing markers filtered to the ``router:`` namespace.

    Returns:
        The distinct strategy/label tokens (e.g. ``("smart_crusher",)`` or
        ``("protected", "smart_crusher")``), empty when no markers were given.
    """
    seen: list[str] = []
    for marker in markers:
        label = _strategy_token(marker)
        if label and label not in seen:
            seen.append(label)
    return tuple(seen)


def _routing_markers(result: object) -> list[str]:
    """Extract ``router:*`` markers from a headroom ``CompressResult``.

    Mirrors ``headroom.pipeline.summarize_routing_markers`` but reads defensively:
    a headroom build without ``transforms_applied`` yields no markers rather than
    breaking the serve path.

    Args:
        result: The object returned by ``headroom.compress``.

    Returns:
        The ``router:``-prefixed marker strings, or ``[]`` when absent.
    """
    transforms = getattr(result, "transforms_applied", None)
    if not isinstance(transforms, list):
        return []
    return [m for m in transforms if isinstance(m, str) and m.startswith("router:")]


@functools.lru_cache(maxsize=1)
def _load_compressor() -> _Compressor | None:
    """Return ``headroom.compress`` when importable, else ``None``.

    Cached so the optional import is attempted once per process and the
    missing-dependency warning is emitted at most once. This is also the seam
    tests monkeypatch to inject a fake compressor or simulate headroom's absence.

    Returns:
        The ``headroom.compress`` callable, or ``None`` when headroom-ai is not
        installed.
    """
    try:
        import headroom  # pyrefly: ignore[missing-import]
    except ImportError:
        logger.warning(
            "headroom-ai not installed; LiteLLM-path compression is a no-op "
            "(install the 'compression' extra to enable it)"
        )
        return None
    else:
        return cast("_Compressor", headroom.compress)


def _messages_chars(messages: JsonArray) -> int:
    """Sum the character length of string message content.

    Only string ``content`` values are counted (the OpenAI ``role="tool"`` and
    assistant text strings); list or ``None`` content contributes nothing, so the
    figure tracks exactly the text headroom can shrink.

    Args:
        messages: OpenAI-format chat messages.

    Returns:
        Total characters across string ``content`` fields.
    """
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def compress_litellm_messages(
    messages: JsonArray,
    *,
    model: str,
    model_limit: int = DEFAULT_MODEL_LIMIT,
    enable_text_ml: bool = False,
) -> tuple[JsonArray, ShrinkStats]:
    """Compress OpenAI-format messages with headroom's default policy.

    Runs on the messages after the Anthropic-to-OpenAI conversion, where
    headroom's content router is effective. Headroom's other defaults apply:
    coding-agent tool output stays verbatim and user/system messages stay
    protected. Savings are measured as the string-content character delta and
    returned as a ShrinkStats so they flow through the existing accounting
    plumbing. Headroom also exposes exact token counts on its ``CompressResult``;
    the character delta is kept here to match the dashboard's existing units.

    Best-effort: when headroom-ai is not installed, or compression raises, the
    input messages are returned unchanged with a no-op (``"none"``) stat.

    Args:
        messages: OpenAI-format chat messages to compress (never mutated).
        model: Provider model id, used by headroom for tokenizer/limit selection.
        model_limit: Token budget passed to ``headroom.compress``.
        enable_text_ml: When False (default), headroom's lossy ML plain-text
            compressor (Kompress) is disabled so only the lossless content
            compressors run; installing the ML extra never changes behaviour on
            its own. Set True per tier to opt that tier's prose tool output into
            ML compression.

    Returns:
        A pair of the (possibly new) message list and a ShrinkStats. The list is
        headroom's compressed output on success, or the input list unchanged on
        no-op.
    """
    compress = _load_compressor()
    before = _messages_chars(messages)
    if compress is None:
        return messages, ShrinkStats(path="none", chars_before=before, chars_after=before)
    extra: dict[str, object] = {} if enable_text_ml else {"kompress_model": _DISABLE_TEXT_ML}
    try:
        result = compress(messages, model=model, model_limit=model_limit, optimize=True, **extra)
    except Exception:
        logger.exception("headroom compression failed; sending messages uncompressed")
        return messages, ShrinkStats(path="none", chars_before=before, chars_after=before)
    compressed = result.messages
    after = _messages_chars(compressed)
    content_types = summarize_content_types(_routing_markers(result))
    return compressed, ShrinkStats(
        path="compress",
        chars_before=before,
        chars_after=after,
        content_types=content_types,
    )
