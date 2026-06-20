"""Wire-aware headroom compression for forwarded request bodies.

Runs headroom over the tool-output text of a request body before it is relayed
to a non-DeepSeek upstream (premium passthrough) or sent on the routed Codex
Responses path. Headroom only shrinks OpenAI ``role="tool"`` string content, so
each wire's tool outputs are converted to that shape, compressed, then the
shrunk text is mapped back into the original body by tool-call id. DeepSeek calls
are never compressed: their byte-identical prefixes feed the provider cache.
Compression is best-effort and fail-open -- any parse, conversion, or headroom
error returns the original body unchanged so a proxied request is never broken by
the optimizer.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.conversion import convert_messages_for_litellm
from ai_calls_router.accounting import shrink_stats
from ai_calls_router.accounting.shrink_stats import ShrinkStats
from ai_calls_router.routing.compression import compress_litellm_messages

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

logger = logging.getLogger("acr.compression")

# Substring marking a DeepSeek upstream. DeepSeek input is never compressed so its
# repetitive tool-result turns keep byte-identical prefixes for the provider cache.
_DEEPSEEK_MARKER = "deepseek"

_NOOP = ShrinkStats(path="none", chars_before=0, chars_after=0)


def _is_deepseek_upstream(upstream: str) -> bool:
    """Return whether an upstream targets DeepSeek (never compressed).

    Args:
        upstream: Upstream base URL the body would be relayed to.

    Returns:
        True when the URL names DeepSeek, so compression must be skipped.
    """
    return _DEEPSEEK_MARKER in upstream.lower()


def _model_of(body: JsonObject) -> str:
    """Return the request body's model id, or an empty string when absent."""
    model = body.get("model")
    return model if isinstance(model, str) else ""


def _stats(before: int, after: int) -> ShrinkStats:
    """Build a ShrinkStats, labelling a real reduction as a compress pass."""
    if after < before:
        return ShrinkStats(path="compress", chars_before=before, chars_after=after)
    return ShrinkStats(path="none", chars_before=before, chars_after=after)


def _flatten_text(value: JsonValue) -> str:
    """Flatten Responses/OpenAI content (string or text-part list) to a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def _compressed_tool_content_by_id(messages: JsonArray) -> dict[str, str]:
    """Map ``tool_call_id`` to compressed string content from headroom output."""
    out: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        content = msg.get("content")
        if isinstance(call_id, str) and isinstance(content, str):
            out[call_id] = content
    return out


def _openai_tool_chars(messages: JsonArray) -> int:
    """Sum the character length of every OpenAI ``role="tool"`` string content."""
    return sum(
        len(cast("str", msg["content"]))
        for msg in messages
        if isinstance(msg, dict)
        and msg.get("role") == "tool"
        and isinstance(msg.get("content"), str)
    )


def compress_openai_chat(body: JsonObject) -> tuple[JsonObject, ShrinkStats]:
    """Compress an OpenAI Chat Completions body in place.

    The messages are already the shape headroom compresses, so the whole list is
    replaced with headroom's output (the same wholesale swap the LiteLLM routed
    path uses) rather than mapping ids back.

    Args:
        body: OpenAI Chat Completions request body (never mutated).

    Returns:
        A new body with compressed messages and a ShrinkStats measuring the
        tool-output character delta, or the input body with a no-op stat when
        there is nothing to compress.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, _NOOP
    compressed, _ = compress_litellm_messages(cast("JsonArray", messages), model=_model_of(body))
    before = _openai_tool_chars(cast("JsonArray", messages))
    after = _openai_tool_chars(compressed)
    if after >= before:
        return body, _stats(before, after)
    return {**body, "messages": compressed}, _stats(before, after)


def _apply_anthropic_tool_results(messages: list[JsonObject], by_id: dict[str, str]) -> JsonArray:
    """Return new Anthropic messages with tool_result content swapped by id."""
    new_messages: JsonArray = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content, changed = _swap_anthropic_blocks(cast("list[JsonValue]", content), by_id)
        new_messages.append({**msg, "content": new_content} if changed else msg)
    return new_messages


def _swap_anthropic_blocks(
    content: list[JsonValue], by_id: dict[str, str]
) -> tuple[list[JsonValue], bool]:
    """Replace tool_result block content with its compressed form when shorter."""
    new_content: list[JsonValue] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            call_id = block.get("tool_use_id")
            if isinstance(call_id, str) and call_id in by_id:
                new_content.append({**block, "content": by_id[call_id]})
                changed = True
                continue
        new_content.append(block)
    return new_content, changed


def compress_anthropic(body: JsonObject) -> tuple[JsonObject, ShrinkStats]:
    """Compress the tool_result text of an Anthropic Messages body.

    Anthropic ``tool_result`` blocks are protected by headroom's gates, so the
    body is converted to OpenAI messages (which preserve ``tool_use_id`` as
    ``tool_call_id``), compressed, and the shrunk tool text is written back into
    the original blocks by id. Only tool_result content changes; assistant and
    user text are left verbatim.

    Args:
        body: Anthropic Messages request body (never mutated).

    Returns:
        A new body with compressed tool_result content and a ShrinkStats, or the
        input body with a no-op stat when nothing was shrunk.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, _NOOP
    converted = convert_messages_for_litellm(cast("list[JsonObject]", messages))
    compressed, _ = compress_litellm_messages(cast("JsonArray", converted), model=_model_of(body))
    by_id = _compressed_tool_content_by_id(compressed)
    if not by_id:
        return body, _NOOP
    new_messages = _apply_anthropic_tool_results(cast("list[JsonObject]", messages), by_id)
    new_body = {**body, "messages": new_messages}
    stats = shrink_stats.compute_shrink(path="compress", before=body, after=new_body)
    if stats.chars_saved <= 0:
        return body, ShrinkStats("none", stats.chars_before, stats.chars_after)
    return new_body, stats


def _responses_input_to_openai(items: list[JsonValue]) -> list[JsonObject]:
    """Build OpenAI messages from Responses input items, preserving call ids."""
    messages: list[JsonObject] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in {"function_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id if isinstance(call_id, str) else "",
                    "content": _flatten_text(item.get("output")),
                }
            )
        elif kind == "function_call":
            messages.append(_responses_function_call_message(item))
        elif kind == "message":
            role = item.get("role")
            messages.append(
                {
                    "role": role if isinstance(role, str) else "user",
                    "content": _flatten_text(item.get("content")),
                }
            )
    return messages


def _responses_function_call_message(item: JsonObject) -> JsonObject:
    """Build an assistant tool_call message from a Responses function_call item."""
    arguments = item.get("arguments")
    name = item.get("name")
    call_id = item.get("call_id")
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id if isinstance(call_id, str) else "",
                "type": "function",
                "function": {
                    "name": name if isinstance(name, str) else "",
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments or {}),
                },
            }
        ],
    }


def _apply_responses_outputs(items: list[JsonValue], by_id: dict[str, str]) -> list[JsonValue]:
    """Return new Responses input with function_call_output text swapped by id."""
    new_items: list[JsonValue] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id in by_id:
                new_items.append({**item, "output": by_id[call_id]})
                continue
        new_items.append(item)
    return new_items


def compress_responses(body: JsonObject) -> tuple[JsonObject, ShrinkStats]:
    """Compress the tool-output text of an OpenAI Responses body.

    Responses ``input`` items are converted to OpenAI messages (preserving
    ``call_id`` as ``tool_call_id``), compressed, and the shrunk text is written
    back into the matching ``function_call_output`` items by id.

    Args:
        body: OpenAI Responses request body (never mutated).

    Returns:
        A new body with compressed function_call_output text and a ShrinkStats,
        or the input body with a no-op stat when nothing was shrunk.
    """
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return body, _NOOP
    converted = _responses_input_to_openai(cast("list[JsonValue]", input_items))
    compressed, _ = compress_litellm_messages(cast("JsonArray", converted), model=_model_of(body))
    by_id = _compressed_tool_content_by_id(compressed)
    if not by_id:
        return body, _NOOP
    new_input = _apply_responses_outputs(cast("list[JsonValue]", input_items), by_id)
    new_body = {**body, "input": new_input}
    stats = shrink_stats.compute_shrink(path="compress", before=body, after=new_body)
    if stats.chars_saved <= 0:
        return body, ShrinkStats("none", stats.chars_before, stats.chars_after)
    return new_body, stats


def _dispatch(body: JsonObject, request_path: str) -> tuple[JsonObject, ShrinkStats]:
    """Route a parsed body to the per-wire compressor for its request path."""
    if request_path == "/v1/chat/completions":
        return compress_openai_chat(body)
    if request_path == "/v1/responses":
        return compress_responses(body)
    if request_path == "/v1/messages":
        return compress_anthropic(body)
    return body, _NOOP


def compress_forward_body(
    body_bytes: bytes, *, request_path: str, upstream: str
) -> tuple[bytes, ShrinkStats]:
    """Compress a forwardable request body, skipping DeepSeek upstreams.

    Best-effort: a DeepSeek upstream, an unparseable or non-object body, an
    unknown wire, or any compression error returns the original bytes unchanged
    with a no-op stat. The body is only re-serialized when compression actually
    removed characters, so non-compressing turns relay byte-identical and keep
    the upstream prompt cache intact.

    Args:
        body_bytes: Raw request body to relay upstream.
        request_path: Client-facing request path, selecting the wire format.
        upstream: Upstream base URL; DeepSeek targets are never compressed.

    Returns:
        A pair of the (possibly re-serialized) body bytes and a ShrinkStats.
    """
    if _is_deepseek_upstream(upstream):
        return body_bytes, _NOOP
    try:
        parsed = cast("JsonValue", json.loads(body_bytes))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return body_bytes, _NOOP
    if not isinstance(parsed, dict):
        return body_bytes, _NOOP
    try:
        new_body, stats = _dispatch(parsed, request_path)
    except Exception:
        logger.exception("forward compression failed; sending body uncompressed")
        return body_bytes, _NOOP
    if stats.chars_saved <= 0:
        return body_bytes, stats
    return json.dumps(new_body).encode("utf-8"), stats
