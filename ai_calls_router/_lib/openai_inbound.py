"""OpenAI Chat Completions conversion for routed requests.

This module converts OpenAI Chat bodies at the proxy edge into the Anthropic
Messages shape used by the routing core, then converts routed Anthropic
responses back to Chat Completions. The transforms are pure and deterministic
so native-Anthropic providers receive stable cache prefixes.
"""

from __future__ import annotations

import json
from typing import Any

from ai_calls_router._lib.conversion import parse_tool_arguments

_STOP_REASON_TO_FINISH_REASON: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _content_text(content: Any) -> str:
    """Return text from a Chat message content value."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _tool_call_to_content_block(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI assistant tool call to an Anthropic tool_use block."""
    function = tool_call.get("function")
    function_obj = function if isinstance(function, dict) else {}
    return {
        "type": "tool_use",
        "id": str(tool_call.get("id", "")),
        "name": str(function_obj.get("name", "")),
        "input": parse_tool_arguments(function_obj.get("arguments", "{}")),
    }


def _message_to_anthropic(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one non-system Chat message to an Anthropic message."""
    role = message.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id", "")),
                    "content": _content_text(message.get("content", "")),
                }
            ],
        }
    if role not in {"user", "assistant"}:
        return None

    content: list[dict[str, Any]] = []
    text = _content_text(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        content.extend(
            _tool_call_to_content_block(tool_call)
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        )
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": role, "content": content}


def _system_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Extract system text from Chat messages."""
    parts = [
        _content_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    ]
    kept = [part for part in parts if part]
    if not kept:
        return None
    return "\n".join(kept)


def openai_chat_to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat messages to Anthropic Messages content blocks.

    Args:
        messages: OpenAI Chat message objects.

    Returns:
        New Anthropic-format messages. System messages are excluded because
        Chat request conversion places them in top-level ``system``.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        anthropic = _message_to_anthropic(message)
        if anthropic is not None:
            converted.append(anthropic)
    return converted


def openai_tool_to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI function tool definition to Anthropic format.

    Args:
        tool: OpenAI Chat tool definition.

    Returns:
        Anthropic tool definition using ``input_schema``.
    """
    function = tool.get("function")
    function_obj = function if isinstance(function, dict) else {}
    converted: dict[str, Any] = {"name": function_obj.get("name", "")}
    if "description" in function_obj:
        converted["description"] = function_obj["description"]
    if "parameters" in function_obj:
        converted["input_schema"] = function_obj["parameters"]
    return converted


def _tool_choice_to_anthropic(choice: Any) -> Any:
    """Convert OpenAI tool_choice to Anthropic format."""
    if isinstance(choice, str):
        if choice == "required":
            return {"type": "any"}
        return {"type": "auto"}
    if isinstance(choice, dict):
        function = choice.get("function")
        function_obj = function if isinstance(function, dict) else {}
        return {"type": "tool", "name": function_obj.get("name", "")}
    return {"type": "auto"}


def chat_request_to_anthropic(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat request body to Anthropic Messages format.

    Args:
        body: OpenAI Chat Completions request body.

    Returns:
        A new Anthropic request body with deterministic key order.

    Raises:
        ValueError: When ``messages`` is not a list of mappings.
    """
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict) for message in raw_messages
    ):
        raise ValueError("chat messages must be a list of objects")
    messages = list(raw_messages)

    converted: dict[str, Any] = {"model": body.get("model", ""), "messages": []}
    system = _system_from_messages(messages)
    if system is not None:
        converted = {"model": converted["model"], "system": system, "messages": []}
    converted["messages"] = openai_chat_to_anthropic_messages(messages)

    tools = body.get("tools")
    if isinstance(tools, list):
        converted["tools"] = [
            openai_tool_to_anthropic(tool) for tool in tools if isinstance(tool, dict)
        ]
    if "tool_choice" in body:
        converted["tool_choice"] = _tool_choice_to_anthropic(body["tool_choice"])

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if max_tokens is not None:
        converted["max_tokens"] = max_tokens
    for key in ("temperature", "top_p"):
        if key in body:
            converted[key] = body[key]
    if "stop" in body:
        converted["stop_sequences"] = body["stop"]
    return converted


def _chat_tool_call(block: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one Anthropic tool_use block to an OpenAI tool call."""
    return {
        "id": block.get("id", f"toolu_routed_{index}"),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
        },
    }


def _usage_to_chat(usage: dict[str, Any]) -> dict[str, int]:
    """Convert Anthropic usage counters to Chat usage counters."""
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def anthropic_to_chat_response(anthropic_body: dict[str, Any]) -> dict[str, Any]:
    """Convert a routed Anthropic response to a Chat Completions response.

    Args:
        anthropic_body: Anthropic Messages response body.

    Returns:
        OpenAI Chat Completions response body.
    """
    content = anthropic_body.get("content")
    blocks = content if isinstance(content, list) else []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            tool_calls.append(_chat_tool_call(block, len(tool_calls)))

    message: dict[str, Any] = {"role": "assistant", "content": None}
    if text_parts:
        message["content"] = "\n".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = anthropic_body.get("usage")
    usage_obj = usage if isinstance(usage, dict) else {}
    finish_reason = _STOP_REASON_TO_FINISH_REASON.get(
        str(anthropic_body.get("stop_reason", "end_turn")), "stop"
    )
    return {
        "id": anthropic_body.get("id", "chatcmpl_routed"),
        "object": "chat.completion",
        "created": 0,
        "model": anthropic_body.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage_to_chat(usage_obj),
    }
