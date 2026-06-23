"""Anthropic Messages API to LiteLLM/OpenAI format conversion.

Extracted from Headroom's proven LiteLLMBackend (headroom/backends/litellm.py)
and the tool-router's RoutedLiteLLMBackend reasoning shim; fidelity against
the originals is enforced by golden-pair tests (tests/fixtures/). Pure dict
transforms only -- this module never imports litellm and never performs IO.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue


class _LiteLLMFunction(Protocol):
    """Function-call object shape returned by LiteLLM."""

    name: str
    arguments: str


class _LiteLLMToolCall(Protocol):
    """Tool-call object shape returned by LiteLLM."""

    id: str
    function: _LiteLLMFunction


class _LiteLLMMessage(Protocol):
    """Assistant message object shape returned by LiteLLM."""

    content: str | None
    tool_calls: Sequence[_LiteLLMToolCall] | None


class _LiteLLMChoice(Protocol):
    """Choice object shape returned by LiteLLM."""

    message: _LiteLLMMessage
    finish_reason: str | None


class _LiteLLMUsage(Protocol):
    """Usage object shape returned by LiteLLM."""

    prompt_tokens: int
    completion_tokens: int


class LiteLLMResponse(Protocol):
    """LiteLLM response shape consumed by this converter."""

    choices: Sequence[_LiteLLMChoice]
    usage: _LiteLLMUsage


@dataclass
class BackendResponse:
    """Standardized routed-call response in Anthropic Messages format.

    Attributes:
        body: Response body (Anthropic Messages API format).
        status_code: HTTP status code.
        headers: Response headers to forward.
        error: Error message, if any.
    """

    body: JsonObject
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=lambda: {})
    error: str | None = None


def convert_anthropic_tool(tool: JsonObject) -> JsonObject:
    """Convert an Anthropic tool definition to OpenAI function format.

    Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}

    Args:
        tool: Anthropic tool definition.

    Returns:
        OpenAI-format function tool definition.
    """
    func: JsonObject = {"name": tool.get("name", "")}
    if "description" in tool:
        func["description"] = tool["description"]
    if "input_schema" in tool:
        func["parameters"] = tool["input_schema"]
    return {"type": "function", "function": func}


def convert_tool_choice(choice: JsonValue) -> str | JsonObject:
    """Convert an Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": ...}
    OpenAI: "auto", "required", {"type": "function", "function": {"name": ...}}

    Args:
        choice: Anthropic tool_choice value.

    Returns:
        OpenAI-format tool_choice ("auto" for anything unrecognized).
    """
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        choice_type = choice.get("type", "auto")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {"type": "function", "function": {"name": choice.get("name", "")}}
    return "auto"


def parse_tool_arguments(arguments: JsonValue) -> JsonValue:
    """Parse tool call arguments from a JSON string to a dict.

    LiteLLM/OpenAI returns arguments as a JSON string, but Anthropic expects
    input as a parsed dict. Unparseable strings pass through unchanged.

    Args:
        arguments: Raw arguments value from an OpenAI tool call.

    Returns:
        Parsed value, or the input unchanged when not a parseable string.
    """
    if isinstance(arguments, str):
        try:
            return cast("JsonValue", json.loads(arguments))
        except (json.JSONDecodeError, TypeError):
            return arguments
    return arguments


def _partition_content_blocks(
    content: Sequence[JsonValue],
) -> tuple[list[str], list[JsonObject], list[JsonObject]]:
    """Partition Anthropic content blocks into text / tool_use / tool_result."""
    text_parts: list[str] = []
    tool_use_blocks: list[JsonObject] = []
    tool_result_blocks: list[JsonObject] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_use_blocks.append(block)
        elif block_type == "tool_result":
            tool_result_blocks.append(block)
    return text_parts, tool_use_blocks, tool_result_blocks


def _flatten_tool_result_content(tr_content: JsonValue) -> str:
    """Flatten a tool_result content (list of text blocks) to a single string."""
    if isinstance(tr_content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in cast("JsonArray", tr_content)
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(tr_content)


def _emit_tool_result_messages(
    tool_result_blocks: list[JsonObject],
) -> list[JsonObject]:
    """Emit OpenAI role=tool messages for Anthropic tool_result blocks."""
    messages: list[JsonObject] = []
    for tr in tool_result_blocks:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": _flatten_tool_result_content(tr.get("content", "")),
            }
        )
    return messages


def _emit_tool_use_message(tool_use_blocks: list[JsonObject], text_parts: list[str]) -> JsonObject:
    """Emit an OpenAI assistant message with tool_calls."""
    msg: JsonObject = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    msg["tool_calls"] = [
        {
            "id": tu["id"],
            "type": "function",
            "function": {
                "name": tu["name"],
                "arguments": json.dumps(tu.get("input", {}), sort_keys=True),
            },
        }
        for tu in tool_use_blocks
    ]
    return msg


def _backfill_reasoning(converted: list[JsonObject]) -> None:
    """Ensure every assistant message has reasoning_content (DeepSeek requirement)."""
    for msg in converted:
        if msg.get("role") == "assistant":
            msg.setdefault("reasoning_content", "")


def convert_messages_for_litellm(messages: list[JsonObject]) -> list[JsonObject]:
    """Convert Anthropic messages to LiteLLM/OpenAI format.

    Anthropic represents tool traffic as content blocks (type=tool_use on
    assistant messages, type=tool_result on user messages); OpenAI uses an
    assistant tool_calls field and separate role=tool messages. Tool role
    messages are emitted with no intervening user text (a Bedrock ordering
    requirement); text alongside tool_result blocks is discarded. Converted
    assistant messages get reasoning_content backfilled to "" because
    DeepSeek V4 thinking mode rejects assistant turns without it.

    Args:
        messages: Anthropic-format message list (never mutated).

    Returns:
        New OpenAI-format message list.
    """
    converted: list[JsonObject] = []
    for msg in messages:
        converted.extend(_convert_single_message(msg))
    _backfill_reasoning(converted)
    return converted


def _convert_single_message(msg: JsonObject) -> list[JsonObject]:
    """Convert one Anthropic message to zero or more LiteLLM/OpenAI messages."""
    role = msg.get("role", "user")
    content = msg.get("content", "")

    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if not isinstance(content, list):
        return []
    content = cast("list[JsonObject]", content)

    text_parts, tool_use_blocks, tool_result_blocks = _partition_content_blocks(content)

    if tool_result_blocks:
        return _emit_tool_result_messages(tool_result_blocks)

    if tool_use_blocks:
        return [_emit_tool_use_message(tool_use_blocks, text_parts)]

    if text_parts:
        return [{"role": role, "content": "\n".join(text_parts)}]

    return [{"role": role, "content": ""}]


def to_anthropic_response(litellm_response: LiteLLMResponse, original_model: str) -> JsonObject:
    """Convert a LiteLLM/OpenAI completion response to Anthropic format.

    Args:
        litellm_response: litellm ModelResponse (attribute access only).
        original_model: Model id to echo in the response body.

    Returns:
        Anthropic Messages API response body with a fresh message id.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    choice = litellm_response.choices[0]
    message = choice.message

    content: list[JsonObject] = []
    if message.content:
        content.append({"type": "text", "text": message.content})

    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": parse_tool_arguments(tc.function.arguments),
                }
            )

    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(choice.finish_reason or "", "end_turn")

    usage = {
        "input_tokens": getattr(litellm_response.usage, "prompt_tokens", 0),
        "output_tokens": getattr(litellm_response.usage, "completion_tokens", 0),
    }

    return cast(
        "JsonObject",
        {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        },
    )


def _insert_system_message(messages: list[JsonObject], system: JsonValue) -> None:
    """Insert a system message at the front of the message list."""
    if isinstance(system, str):
        messages.insert(0, {"role": "system", "content": system})
    elif isinstance(system, list):
        parts = (
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in cast("JsonArray", system)
        )
        messages.insert(0, {"role": "system", "content": " ".join(parts)})


def completion_kwargs(body: JsonObject, api_key: str | None = None) -> JsonObject:
    """Assemble litellm.acompletion kwargs from an Anthropic request body.

    Only explicitly recognized fields are translated; everything else in the
    client body (stream, metadata, betas, auth material) never reaches the
    provider. The api_key is the resolved tier key -- client credentials are
    not accepted here by design.

    Args:
        body: Anthropic Messages API request body (never mutated).
        api_key: Tier API key from key_env/env_file, if resolved.

    Returns:
        Keyword arguments for litellm.acompletion.
    """
    messages = body.get("messages")
    converted_messages = convert_messages_for_litellm(
        cast("list[JsonObject]", messages) if isinstance(messages, list) else []
    )
    kwargs: JsonObject = {
        "model": body.get("model", ""),
        "messages": cast("JsonValue", converted_messages),
    }

    field_pairs: tuple[tuple[str, str], ...] = (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    )
    for src, dst in field_pairs:
        if src in body:
            kwargs[dst] = body[src]
    if "stop_sequences" in body:
        kwargs["stop"] = body["stop_sequences"]

    if "tools" in body:
        tools = body["tools"]
        kwargs["tools"] = cast(
            "JsonValue",
            [
                convert_anthropic_tool(tool)
                for tool in cast("JsonArray", tools)
                if isinstance(tool, dict)
            ]
            if isinstance(tools, list)
            else [],
        )
    if "tool_choice" in body:
        kwargs["tool_choice"] = convert_tool_choice(body["tool_choice"])

    if "system" in body:
        _insert_system_message(converted_messages, body["system"])

    if api_key:
        kwargs["api_key"] = api_key

    return kwargs
