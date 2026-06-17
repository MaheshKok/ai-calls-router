"""OpenAI Responses conversion for routed requests.

Responses is a separate wire format from Chat Completions, so this module keeps
its pure edge conversion out of the Phase 3 Chat modules. Reasoning items are
stripped during input conversion because encrypted reasoning content is not a
semantic prompt prefix and must not reach cache-keyed routed providers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib.conversion import parse_tool_arguments

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

_HOSTED_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "local_shell_call",
        "web_search_call",
        "image_generation_call",
        "file_search_call",
        "computer_call",
        "code_interpreter_call",
        "mcp_call",
    }
)


def _text_from_content(content: JsonValue) -> str:
    """Return text from Responses message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in cast("JsonArray", content):
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in {
                "input_text",
                "output_text",
                "text",
            }:
                text = part.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _image_block(part: JsonObject) -> JsonObject:
    """Convert a Responses image part to Anthropic image format."""
    image_url = part.get("image_url")
    if isinstance(image_url, str) and image_url:
        return {"type": "image", "source": {"type": "url", "url": image_url}}
    raise ValueError("unsupported Responses input_image part")


def _message_content_blocks(item: JsonObject) -> list[JsonObject]:
    """Convert a Responses message content field to Anthropic blocks."""
    content = item.get("content")
    if isinstance(content, list):
        blocks: list[JsonObject] = []
        for part in cast("JsonArray", content):
            if isinstance(part, dict) and part.get("type") == "input_image":
                blocks.append(_image_block(cast("JsonObject", part)))
                continue
            text = _text_from_content([part])
            if text:
                blocks.append({"type": "text", "text": text})
        return blocks or [{"type": "text", "text": ""}]
    return [{"type": "text", "text": _text_from_content(content)}]


def _message_item_to_anthropic(item: JsonObject) -> JsonObject | None:
    """Convert a Responses message item to an Anthropic message."""
    role = item.get("role")
    if role == "system":
        return None
    if role not in {"user", "assistant"}:
        raise ValueError("Responses message role must be user, assistant, or system")
    return {"role": role, "content": cast("JsonArray", _message_content_blocks(item))}


def _function_call_to_message(item: JsonObject) -> JsonObject:
    """Convert a Responses function_call item to an Anthropic assistant message."""
    call_id = item.get("call_id")
    name = item.get("name")
    if not call_id or not name or "arguments" not in item:
        raise ValueError("function_call requires call_id, name, and arguments")
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": str(call_id),
                "name": str(name),
                "input": parse_tool_arguments(item.get("arguments", "{}")),
            }
        ],
    }


def _custom_tool_call_to_message(item: JsonObject) -> JsonObject:
    """Convert a Responses custom_tool_call item to an Anthropic assistant message."""
    call_id = item.get("call_id")
    name = item.get("name")
    if not call_id or not name or "input" not in item:
        raise ValueError("custom_tool_call requires call_id, name, and input")
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": str(call_id),
                "name": str(name),
                "input": {"input": item.get("input", "")},
            }
        ],
    }


def _tool_output_to_message(item: JsonObject) -> JsonObject:
    """Convert a Responses tool output item to an Anthropic user message."""
    call_id = item.get("call_id")
    if not call_id or "output" not in item:
        raise ValueError("tool output requires call_id and output")
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": str(call_id),
                "content": str(item.get("output", "")),
            }
        ],
    }


def _input_item_to_message(item: JsonObject) -> JsonObject | None:
    """Convert one Responses input item to Anthropic, or skip non-routed items."""
    item_type = item.get("type", "message" if "role" in item else "")
    if item_type == "message":
        return _message_item_to_anthropic(item)
    if item_type == "function_call":
        return _function_call_to_message(item)
    if item_type == "function_call_output":
        return _tool_output_to_message(item)
    if item_type == "custom_tool_call":
        return _custom_tool_call_to_message(item)
    if item_type == "custom_tool_call_output":
        return _tool_output_to_message(item)
    if item_type == "reasoning" or item_type in _HOSTED_ITEM_TYPES:
        return None
    raise ValueError(f"unsupported Responses input item type: {item_type}")


def responses_input_to_anthropic_messages(input_items: JsonValue) -> list[JsonObject]:
    """Convert Responses input into Anthropic Messages.

    Args:
        input_items: Full Responses conversation input, or a string shorthand.

    Returns:
        New Anthropic-format messages with reasoning items stripped.

    Raises:
        ValueError: If a routed item is malformed or unsupported.
    """
    if isinstance(input_items, str):
        return [{"role": "user", "content": [{"type": "text", "text": input_items}]}]
    if not isinstance(input_items, list):
        raise ValueError("Responses input must be a string or list")  # noqa: TRY004
    messages: list[JsonObject] = []
    for item in input_items:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects")  # noqa: TRY004
        message = _input_item_to_message(item)
        if message is not None:
            messages.append(message)
    return messages


def responses_tool_to_anthropic(tool: JsonObject) -> JsonObject:
    """Convert a Responses tool definition to Anthropic format."""
    tool_type = tool.get("type")
    name = tool.get("name")
    if not name:
        raise ValueError("Responses tool requires name")
    converted: JsonObject = {"name": name}
    if "description" in tool:
        converted["description"] = tool["description"]
    if tool_type == "function":
        converted["input_schema"] = tool.get("parameters", {"type": "object"})
        return converted
    if tool_type == "custom":
        converted["input_schema"] = {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
        return converted
    raise ValueError("unsupported Responses tool type")


def _tool_choice_to_anthropic(choice: JsonValue) -> JsonObject:
    """Convert Responses tool_choice to Anthropic format."""
    if isinstance(choice, str):
        if choice == "required":
            return {"type": "any"}
        return {"type": "auto"}
    if isinstance(choice, dict):
        name = choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    return {"type": "auto"}


def _system_from_input(input_items: JsonValue) -> str | None:
    """Extract system message text from Responses input."""
    if not isinstance(input_items, list):
        return None
    parts = [
        _text_from_content(item.get("content"))
        for item in cast("JsonArray", input_items)
        if isinstance(item, dict)
        and item.get("type", "message") == "message"
        and item.get("role") == "system"
    ]
    kept = [part for part in parts if part]
    return "\n".join(kept) if kept else None


def responses_request_to_anthropic(body: JsonObject) -> JsonObject:
    """Convert a Responses request body to Anthropic Messages format.

    The router is stateless, so ``store`` and ``previous_response_id`` are not
    implemented here; clients must send self-contained ``input``. Unsupported
    or malformed routed items raise ``ValueError`` so the server fail-opens to
    verbatim passthrough.
    """
    if "input" not in body:
        raise ValueError("Responses request requires input")
    converted: JsonObject = {"model": body.get("model", ""), "messages": []}
    system_parts = [
        value
        for value in (body.get("instructions"), _system_from_input(body.get("input")))
        if isinstance(value, str) and value
    ]
    if system_parts:
        converted = {"model": converted["model"], "system": "\n".join(system_parts), "messages": []}
    raw_input = body["input"]
    if isinstance(raw_input, str | list):
        converted["messages"] = cast("JsonArray", responses_input_to_anthropic_messages(raw_input))
    else:
        raise ValueError("Responses input must be a string or list")  # noqa: TRY004

    tools = body.get("tools")
    if isinstance(tools, list):
        converted["tools"] = [
            responses_tool_to_anthropic(cast("JsonObject", tool))
            for tool in cast("JsonArray", tools)
            if isinstance(tool, dict)
        ]
    if "tool_choice" in body:
        converted["tool_choice"] = _tool_choice_to_anthropic(body["tool_choice"])
    if "max_output_tokens" in body:
        converted["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if key in body:
            converted[key] = body[key]
    if "stop" in body:
        converted["stop_sequences"] = body["stop"]
    return converted


def _json_int(value: JsonValue) -> int:
    """Coerce simple JSON values to int, defaulting malformed values to zero."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _usage_to_responses(usage: JsonObject) -> dict[str, int]:
    """Convert Anthropic usage counters to Responses usage."""
    input_tokens = _json_int(usage.get("input_tokens", 0))
    output_tokens = _json_int(usage.get("output_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _response_status(stop_reason: JsonValue) -> tuple[str, JsonObject | None]:
    """Return Responses status and incomplete details."""
    if stop_reason == "max_tokens":
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _function_output_item(block: JsonObject, index: int) -> JsonObject:
    """Convert Anthropic tool_use block to a Responses function_call item."""
    return {
        "id": block.get("id", f"fc_routed_{index}"),
        "type": "function_call",
        "call_id": block.get("id", f"fc_routed_{index}"),
        "name": block.get("name", ""),
        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
    }


def _text_output_item(text_parts: list[str]) -> JsonObject:
    """Build one Responses assistant message output item."""
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
    }


def _output_items(anthropic_response: JsonObject) -> list[JsonObject]:
    """Convert Anthropic response content blocks to Responses output items."""
    content = anthropic_response.get("content")
    blocks = cast("JsonArray", content) if isinstance(content, list) else []
    output: list[JsonObject] = []
    text_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if text_parts:
            output.append(_text_output_item(text_parts))
            text_parts = []
        if block.get("type") == "tool_use":
            output.append(_function_output_item(block, len(output)))
    if text_parts:
        output.append(_text_output_item(text_parts))
    return output


def anthropic_to_responses(anthropic_response: JsonObject, model: str) -> JsonObject:
    """Convert an Anthropic response to non-streaming Responses JSON."""
    usage = anthropic_response.get("usage")
    usage_obj = usage if isinstance(usage, dict) else {}
    status, incomplete_details = _response_status(anthropic_response.get("stop_reason"))
    response = cast(
        "JsonObject",
        {
            "id": anthropic_response.get("id", "resp_routed"),
            "object": "response",
            "created_at": 0,
            "status": status,
            "model": model,
            "output": _output_items(anthropic_response),
            "usage": _usage_to_responses(usage_obj),
        },
    )
    if incomplete_details is not None:
        response["incomplete_details"] = incomplete_details
    return response
