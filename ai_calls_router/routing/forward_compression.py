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

import hashlib
import json
import logging
from typing import TYPE_CHECKING, cast

from ai_calls_router.accounting import shrink_stats
from ai_calls_router.accounting.shrink_stats import ShrinkStats
from ai_calls_router.routing.compression import compress_litellm_messages

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

logger = logging.getLogger("acr.compression")

# Substring marking a DeepSeek upstream. DeepSeek input is never compressed so its
# repetitive tool-result turns keep byte-identical prefixes for the provider cache.
_DEEPSEEK_MARKER = "deepseek"

_NOOP = ShrinkStats(path="none", chars_before=0, chars_after=0)
_ANTHROPIC_BLOCK_COMPRESSOR_VERSION = "anthropic-block-v1"
_ANTHROPIC_BLOCK_CACHE_MAX = 4096
_ANTHROPIC_BLOCK_CACHE: dict[str, str] = {}
_ANTHROPIC_AUTO_CACHE_CONTROL: JsonObject = {"type": "ephemeral"}


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


def _blocks_have_cache_control(blocks: JsonValue) -> bool:
    """Return whether any dict in a block list declares cache_control."""
    return isinstance(blocks, list) and any(
        isinstance(block, dict) and "cache_control" in block for block in blocks
    )


def _messages_have_cache_control(messages: JsonValue) -> bool:
    """Return whether any message content block declares cache_control."""
    if not isinstance(messages, list):
        return False
    return any(
        _blocks_have_cache_control(msg.get("content")) for msg in messages if isinstance(msg, dict)
    )


def _has_anthropic_cache_control(body: JsonObject) -> bool:
    """Return whether an Anthropic request already declares any cache policy."""
    return (
        "cache_control" in body
        or _blocks_have_cache_control(body.get("tools"))
        or _blocks_have_cache_control(body.get("system"))
        or _messages_have_cache_control(body.get("messages"))
    )


def _anthropic_has_cacheable_content(body: JsonObject) -> bool:
    """Return whether a Messages body has any cacheable prompt content."""
    return any(body.get(key) for key in ("tools", "system", "messages"))


def apply_anthropic_prompt_cache(body: JsonObject) -> tuple[JsonObject, bool]:
    """Add Anthropic automatic prompt caching when safe and absent."""
    if _has_anthropic_cache_control(body) or not _anthropic_has_cacheable_content(body):
        return body, False
    return {**body, "cache_control": dict(_ANTHROPIC_AUTO_CACHE_CONTROL)}, True


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


def _anthropic_tool_names_by_id(messages: list[JsonObject]) -> dict[str, str]:
    """Map Anthropic tool_use ids to tool names for cache-key stability."""
    names: dict[str, str] = {}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id = block.get("id")
            name = block.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def _flatten_anthropic_tool_result_content(value: JsonValue) -> str:
    """Flatten Anthropic tool_result content the same way conversion does."""
    if isinstance(value, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in cast("JsonArray", value)
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(value)


def _compressed_block_cache_key(
    *,
    tool_name: str,
    tool_use_id: str,
    original_block_content: JsonValue,
    compressor_version: str,
) -> str:
    """Return the stable cache key for one Anthropic tool_result block."""
    raw = json.dumps(
        {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "original_block_content": original_block_content,
            "compressor_version": compressor_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_anthropic_block(key: str, value: str) -> None:
    # ponytail: process-local clear-all eviction; add LRU only if this grows in prod.
    if len(_ANTHROPIC_BLOCK_CACHE) >= _ANTHROPIC_BLOCK_CACHE_MAX:
        _ANTHROPIC_BLOCK_CACHE.clear()
    _ANTHROPIC_BLOCK_CACHE[key] = value


def _compress_anthropic_tool_result_block(
    block: JsonObject,
    *,
    model: str,
    tool_name: str,
    compressor_version: str,
    enable_text_ml: bool,
) -> str | None:
    """Compress one tool_result block as a pure function of its content key."""
    call_id = block.get("tool_use_id")
    if not isinstance(call_id, str):
        return None
    original_content = block.get("content", "")
    key = _compressed_block_cache_key(
        tool_name=tool_name,
        tool_use_id=call_id,
        original_block_content=original_content,
        compressor_version=compressor_version,
    )
    if key in _ANTHROPIC_BLOCK_CACHE:
        return _ANTHROPIC_BLOCK_CACHE[key]

    original_text = _flatten_anthropic_tool_result_content(original_content)
    compressed, _ = compress_litellm_messages(
        cast(
            "JsonArray",
            [{"role": "tool", "tool_call_id": call_id, "content": original_text}],
        ),
        model=model,
        enable_text_ml=enable_text_ml,
    )
    shrunk = _compressed_tool_content_by_id(compressed).get(call_id)
    if not isinstance(shrunk, str) or len(shrunk) >= len(original_text):
        return None
    _cache_anthropic_block(key, shrunk)
    return shrunk


def _openai_tool_chars(messages: JsonArray) -> int:
    """Sum the character length of every OpenAI ``role="tool"`` string content."""
    return sum(
        len(cast("str", msg["content"]))
        for msg in messages
        if isinstance(msg, dict)
        and msg.get("role") == "tool"
        and isinstance(msg.get("content"), str)
    )


def compress_openai_chat(
    body: JsonObject, *, enable_text_ml: bool = False
) -> tuple[JsonObject, ShrinkStats]:
    """Compress an OpenAI Chat Completions body in place.

    The messages are already the shape headroom compresses, so the whole list is
    replaced with headroom's output (the same wholesale swap the LiteLLM routed
    path uses) rather than mapping ids back.

    Args:
        body: OpenAI Chat Completions request body (never mutated).
        enable_text_ml: Opt into headroom's lossy ML plain-text compressor; off
            by default so only lossless compressors run.

    Returns:
        A new body with compressed messages and a ShrinkStats measuring the
        tool-output character delta, or the input body with a no-op stat when
        there is nothing to compress.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, _NOOP
    compressed, _ = compress_litellm_messages(
        cast("JsonArray", messages), model=_model_of(body), enable_text_ml=enable_text_ml
    )
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


def _iter_anthropic_blocks(messages: list[JsonObject]) -> Iterator[JsonObject]:
    """Yield every dict content block across all message contents."""
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                yield cast("JsonObject", block)


def _collect_anthropic_compressed_blocks(
    messages: list[JsonObject],
    *,
    model: str,
    tool_names: dict[str, str],
    compressor_version: str,
    enable_text_ml: bool,
) -> tuple[dict[str, str], bool]:
    """Compress every tool_result block, returning shrunk-by-id and whether any was seen."""
    by_id: dict[str, str] = {}
    saw_tool_result = False
    for block in _iter_anthropic_blocks(messages):
        if block.get("type") != "tool_result":
            continue
        call_id = block.get("tool_use_id")
        if not isinstance(call_id, str):
            continue
        saw_tool_result = True
        shrunk = _compress_anthropic_tool_result_block(
            block,
            model=model,
            tool_name=tool_names.get(call_id, ""),
            compressor_version=compressor_version,
            enable_text_ml=enable_text_ml,
        )
        if shrunk is not None:
            by_id[call_id] = shrunk
    return by_id, saw_tool_result


def compress_anthropic(
    body: JsonObject, *, enable_text_ml: bool = False
) -> tuple[JsonObject, ShrinkStats]:
    """Compress the tool_result text of an Anthropic Messages body.

    Anthropic ``tool_result`` blocks are protected by headroom's gates, so the
    body is converted to OpenAI messages (which preserve ``tool_use_id`` as
    ``tool_call_id``), compressed, and the shrunk tool text is written back into
    the original blocks by id. Only tool_result content changes; assistant and
    user text are left verbatim.

    Args:
        body: Anthropic Messages request body (never mutated).
        enable_text_ml: Opt into headroom's lossy ML plain-text compressor; off
            by default so only lossless compressors run.

    Returns:
        A new body with compressed tool_result content and a ShrinkStats, or the
        input body with a no-op stat when nothing was shrunk.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, _NOOP
    typed_messages = cast("list[JsonObject]", messages)
    model = _model_of(body)
    compressor_version = (
        f"{_ANTHROPIC_BLOCK_COMPRESSOR_VERSION}:model={model}:text_ml={int(enable_text_ml)}"
    )
    by_id, saw_tool_result = _collect_anthropic_compressed_blocks(
        typed_messages,
        model=model,
        tool_names=_anthropic_tool_names_by_id(typed_messages),
        compressor_version=compressor_version,
        enable_text_ml=enable_text_ml,
    )
    if not by_id:
        if saw_tool_result:
            return body, shrink_stats.compute_shrink(path="none", before=body, after=body)
        return body, _NOOP
    new_messages = _apply_anthropic_tool_results(typed_messages, by_id)
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


def compress_responses(
    body: JsonObject, *, enable_text_ml: bool = False
) -> tuple[JsonObject, ShrinkStats]:
    """Compress the tool-output text of an OpenAI Responses body.

    Responses ``input`` items are converted to OpenAI messages (preserving
    ``call_id`` as ``tool_call_id``), compressed, and the shrunk text is written
    back into the matching ``function_call_output`` items by id.

    Args:
        body: OpenAI Responses request body (never mutated).
        enable_text_ml: Opt into headroom's lossy ML plain-text compressor; off
            by default so only lossless compressors run.

    Returns:
        A new body with compressed function_call_output text and a ShrinkStats,
        or the input body with a no-op stat when nothing was shrunk.
    """
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return body, _NOOP
    converted = _responses_input_to_openai(cast("list[JsonValue]", input_items))
    compressed, _ = compress_litellm_messages(
        cast("JsonArray", converted), model=_model_of(body), enable_text_ml=enable_text_ml
    )
    by_id = _compressed_tool_content_by_id(compressed)
    if not by_id:
        return body, _NOOP
    new_input = _apply_responses_outputs(cast("list[JsonValue]", input_items), by_id)
    new_body = {**body, "input": new_input}
    stats = shrink_stats.compute_shrink(path="compress", before=body, after=new_body)
    if stats.chars_saved <= 0:
        return body, ShrinkStats("none", stats.chars_before, stats.chars_after)
    return new_body, stats


def _dispatch(
    body: JsonObject, request_path: str, *, enable_text_ml: bool = False
) -> tuple[JsonObject, ShrinkStats]:
    """Route a parsed body to the per-wire compressor for its request path."""
    if request_path == "/v1/chat/completions":
        return compress_openai_chat(body, enable_text_ml=enable_text_ml)
    if request_path == "/v1/responses":
        return compress_responses(body, enable_text_ml=enable_text_ml)
    if request_path == "/v1/messages":
        return compress_anthropic(body, enable_text_ml=enable_text_ml)
    return body, _NOOP


def compress_forward_body(
    body_bytes: bytes,
    *,
    request_path: str,
    upstream: str,
    enable_text_ml: bool = False,
    prompt_cache: bool = False,
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
        enable_text_ml: Opt into headroom's lossy ML plain-text compressor; off
            by default so premium passthrough stays lossless.
        prompt_cache: Add Anthropic automatic prompt caching to Messages bodies.

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
        cache_applied = False
        cache_body = parsed
        if prompt_cache and request_path == "/v1/messages":
            cache_body, cache_applied = apply_anthropic_prompt_cache(parsed)
        new_body, stats = _dispatch(cache_body, request_path, enable_text_ml=enable_text_ml)
    except Exception:
        logger.exception("forward compression failed; sending body uncompressed")
        return body_bytes, _NOOP
    if stats.chars_saved <= 0 and not cache_applied:
        return body_bytes, stats
    return json.dumps(new_body).encode("utf-8"), stats
