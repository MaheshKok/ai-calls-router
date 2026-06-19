"""OpenAI Responses conversion for routed requests.

Responses is a separate wire format from Chat Completions, so this module keeps
its pure edge conversion out of the Phase 3 Chat modules. Reasoning items are
stripped during input conversion because encrypted reasoning content is not a
semantic prompt prefix and must not reach cache-keyed routed providers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib import jsonnum
from ai_calls_router._lib.conversion import parse_tool_arguments
from ai_calls_router._lib.openai_schemas import validate_responses_request

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
        "tool_search_call",
        "tool_search_output",
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
    if role in {"system", "developer"}:
        return None
    if role not in {"user", "assistant"}:
        raise ValueError("Responses message role must be user, assistant, system, or developer")
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


def _sanitize_tool_adjacency(messages: list[JsonObject]) -> list[JsonObject]:
    """Drop Anthropic tool blocks DeepSeek rejects as orphaned history."""
    coalesced = _coalesce_tool_use_turns(messages)
    sanitized: list[JsonObject] = []
    index = 0
    while index < len(coalesced):
        message = coalesced[index]
        content = message.get("content")
        blocks = cast("JsonArray", content) if isinstance(content, list) else cast("JsonArray", [])
        if message.get("role") == "assistant":
            next_index = _append_safe_assistant_turn(sanitized, coalesced, index, blocks)
            index = next_index
            continue
        if message.get("role") == "user" and blocks:
            safe_blocks = _without_tool_results(blocks)
            if safe_blocks:
                sanitized.append({**message, "content": safe_blocks})
            index += 1
            continue
        sanitized.append(dict(message))
        index += 1
    return sanitized


def _contains_tool_output(input_items: JsonValue) -> bool:
    """Return whether Responses input contains tool output items."""
    if not isinstance(input_items, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") in {"function_call_output", "custom_tool_call_output"}
        for item in input_items
    )


def _coalesce_tool_use_turns(messages: list[JsonObject]) -> list[JsonObject]:
    """Merge adjacent tool-use-only assistant messages into one Anthropic turn."""
    coalesced: list[JsonObject] = []
    pending: JsonArray = []
    for message in messages:
        content = message.get("content")
        blocks = cast("JsonArray", content) if isinstance(content, list) else cast("JsonArray", [])
        if message.get("role") == "assistant" and _only_tool_use_blocks(blocks):
            pending.extend(blocks)
            continue
        if pending:
            coalesced.append({"role": "assistant", "content": pending})
            pending = []
        coalesced.append(dict(message))
    if pending:
        coalesced.append({"role": "assistant", "content": pending})
    return coalesced


def _only_tool_use_blocks(blocks: JsonArray) -> bool:
    """Return whether content is a non-empty tool_use-only block list."""
    return bool(blocks) and all(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in blocks
    )


def _append_safe_assistant_turn(
    sanitized: list[JsonObject],
    messages: list[JsonObject],
    index: int,
    blocks: JsonArray,
) -> int:
    """Append one assistant turn with only immediately matched tool pairs."""
    tool_ids = _tool_use_ids(blocks)
    if not tool_ids:
        sanitized.append(dict(messages[index]))
        return index + 1

    next_message = messages[index + 1] if index + 1 < len(messages) else None
    next_blocks = _user_blocks(next_message)
    matched_ids = tool_ids & _tool_result_ids(next_blocks)
    safe_assistant_blocks = _without_unmatched_tool_uses(blocks, matched_ids)
    if safe_assistant_blocks:
        sanitized.append({**messages[index], "content": safe_assistant_blocks})
    if matched_ids and isinstance(next_message, dict):
        sanitized.append(
            {**next_message, "content": _matched_user_blocks(next_blocks, matched_ids)}
        )
        return index + 2
    return index + 1


def _user_blocks(message: JsonObject | None) -> JsonArray:
    """Return user content blocks when the next turn can answer tool uses."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return []
    content = message.get("content")
    return cast("JsonArray", content) if isinstance(content, list) else []


def _tool_use_ids(blocks: JsonArray) -> set[str]:
    """Collect tool_use ids from Anthropic content blocks."""
    return {
        str(block["id"])
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
    }


def _tool_result_ids(blocks: JsonArray) -> set[str]:
    """Collect tool_result ids from Anthropic content blocks."""
    return {
        str(block["tool_use_id"])
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    }


def _without_unmatched_tool_uses(blocks: JsonArray, matched_ids: set[str]) -> JsonArray:
    """Return assistant blocks with orphaned tool_use entries removed."""
    return [
        block
        for block in blocks
        if not (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and str(block.get("id", "")) not in matched_ids
        )
    ]


def _without_tool_results(blocks: JsonArray) -> JsonArray:
    """Return user blocks with orphaned tool_result entries removed."""
    return [
        block
        for block in blocks
        if not (isinstance(block, dict) and block.get("type") == "tool_result")
    ]


def _matched_user_blocks(blocks: JsonArray, matched_ids: set[str]) -> JsonArray:
    """Return matched tool results plus ordinary user blocks."""
    matched_results = [
        block
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and str(block.get("tool_use_id", "")) in matched_ids
    ]
    return [*matched_results, *_without_tool_results(blocks)]


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


def _routable_tool_to_anthropic(tool: JsonObject) -> JsonObject | None:
    """Convert function/custom tools and ignore hosted Responses tools."""
    if tool.get("type") not in {"function", "custom"}:
        return None
    return responses_tool_to_anthropic(tool)


def _routable_tools_to_anthropic(tools: JsonValue) -> JsonArray:
    """Convert routable Responses tools while skipping hosted entries."""
    converted_tools: JsonArray = []
    if not isinstance(tools, list):
        return converted_tools
    for tool in cast("JsonArray", tools):
        if not isinstance(tool, dict):
            continue
        converted_tool = _routable_tool_to_anthropic(cast("JsonObject", tool))
        if converted_tool is not None:
            converted_tools.append(converted_tool)
    return converted_tools


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


def _anthropic_tool_choice_to_responses(choice: JsonValue) -> JsonValue:
    """Convert Anthropic tool_choice to Responses shape."""
    if not isinstance(choice, dict):
        return None
    choice_type = choice.get("type")
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and isinstance(choice.get("name"), str):
        return {"type": "function", "name": choice["name"]}
    if choice_type in {"auto", "none"}:
        return choice_type
    return None


def _system_from_input(input_items: JsonValue) -> str | None:
    """Extract system message text from Responses input."""
    if not isinstance(input_items, list):
        return None
    parts = [
        _text_from_content(item.get("content"))
        for item in cast("JsonArray", input_items)
        if isinstance(item, dict)
        and item.get("type", "message") == "message"
        and item.get("role") in {"system", "developer"}
    ]
    kept = [part for part in parts if part]
    return "\n".join(kept) if kept else None


def _messages_from_responses_body(body: JsonObject) -> list[JsonObject]:
    """Return Anthropic messages from a validated Responses request."""
    raw_input = body["input"]
    if not isinstance(raw_input, str | list):  # pyrefly: ignore[implicit-any-type-argument]
        raise ValueError("Responses input must be a string or list")  # noqa: TRY004
    messages = responses_input_to_anthropic_messages(raw_input)
    if _contains_tool_output(raw_input):
        return _sanitize_tool_adjacency(messages)
    return messages


def _copy_optional_request_fields(source: JsonObject, target: JsonObject) -> None:
    """Copy optional Responses request fields to Anthropic shape."""
    if "tools" in source:
        converted_tools = _routable_tools_to_anthropic(source["tools"])
        if converted_tools:
            target["tools"] = converted_tools
    if "tool_choice" in source:
        target["tool_choice"] = _tool_choice_to_anthropic(source["tool_choice"])
    if "max_output_tokens" in source:
        target["max_tokens"] = source["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if key in source:
            target[key] = source[key]
    if "stop" in source:
        target["stop_sequences"] = source["stop"]


def responses_request_to_anthropic(body: JsonObject) -> JsonObject:
    """Convert a Responses request body to Anthropic Messages format.

    The router is stateless, so ``store`` and ``previous_response_id`` are not
    implemented here; clients must send self-contained ``input``. Unsupported
    or malformed routed items raise ``ValueError`` so the server fail-opens to
    verbatim passthrough.
    """
    if "input" not in body:
        raise ValueError("Responses request requires input")
    validate_responses_request(body)
    converted: JsonObject = {"model": body.get("model", ""), "messages": []}
    system_parts = [
        value
        for value in (body.get("instructions"), _system_from_input(body.get("input")))
        if isinstance(value, str) and value
    ]
    if system_parts:
        converted = {"model": converted["model"], "system": "\n".join(system_parts), "messages": []}
    converted["messages"] = cast("JsonArray", _messages_from_responses_body(body))
    _copy_optional_request_fields(body, converted)
    return converted


def _responses_text_part(text: str, *, role: str) -> JsonObject:
    """Build one Responses text content part."""
    part_type = "output_text" if role == "assistant" else "input_text"
    return {"type": part_type, "text": text}


def _anthropic_content_to_responses_input(message: JsonObject) -> list[JsonObject]:
    """Convert one Anthropic message into Responses input items."""
    role = message.get("role")
    content = message.get("content")
    blocks = cast("JsonArray", content) if isinstance(content, list) else cast("JsonArray", [])
    if not isinstance(role, str):
        return []
    items: list[JsonObject] = []
    text_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if text_parts:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [_responses_text_part("\n".join(text_parts), role=role)],
                }
            )
            text_parts = []
        if block_type == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                }
            )
        elif block_type == "tool_result":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": str(block.get("content", "")),
                }
            )
    if text_parts:
        items.append(
            {
                "type": "message",
                "role": role,
                "content": [_responses_text_part("\n".join(text_parts), role=role)],
            }
        )
    if not blocks and isinstance(content, str):
        items.append(
            {
                "type": "message",
                "role": role,
                "content": [_responses_text_part(content, role=role)],
            }
        )
    return items


def _anthropic_tool_to_responses(tool: JsonObject) -> JsonObject:
    """Convert Anthropic tool definition to Responses function tool."""
    converted: JsonObject = {
        "type": "function",
        "name": str(tool.get("name", "")),
        "parameters": tool.get("input_schema", {"type": "object"}),
    }
    description = tool.get("description")
    if isinstance(description, str) and description:
        converted["description"] = description
    return converted


def anthropic_request_to_responses(body: JsonObject) -> JsonObject:
    """Convert an Anthropic canonical routed request into Responses JSON."""
    converted: JsonObject = {"model": str(body.get("model", "")), "input": []}
    system = body.get("system")
    if isinstance(system, str) and system:
        converted["instructions"] = system
    input_items: list[JsonObject] = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in cast("JsonArray", messages):
            if isinstance(message, dict):
                input_items.extend(_anthropic_content_to_responses_input(message))
    converted["input"] = cast("JsonArray", input_items)
    tools = body.get("tools")
    if isinstance(tools, list):
        converted["tools"] = cast(
            "JsonArray",
            [
                _anthropic_tool_to_responses(tool)
                for tool in cast("JsonArray", tools)
                if isinstance(tool, dict)
            ],
        )
    choice = _anthropic_tool_choice_to_responses(body.get("tool_choice"))
    if choice is not None:
        converted["tool_choice"] = choice
    if "max_tokens" in body:
        converted["max_output_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in body:
            converted[key] = body[key]
    return converted


def _usage_to_responses(usage: JsonObject) -> dict[str, int]:
    """Convert Anthropic usage counters to Responses usage."""
    input_tokens = jsonnum.int_value(usage.get("input_tokens", 0))
    output_tokens = jsonnum.int_value(usage.get("output_tokens", 0))
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
    blocks: JsonArray = cast("JsonArray", content) if isinstance(content, list) else []
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


def _anthropic_content_from_responses(output_items: JsonArray) -> list[JsonObject]:
    """Convert Responses output items to Anthropic content blocks."""
    content: list[JsonObject] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content.extend(_text_blocks_from_response_message(item))
        elif item.get("type") == "function_call":
            content.append(_tool_use_block_from_response_call(item))
    return content


def _text_blocks_from_response_message(item: JsonObject) -> list[JsonObject]:
    """Return Anthropic text blocks from one Responses message item."""
    blocks: list[JsonObject] = []
    for part in cast("JsonArray", item.get("content", [])):
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def _tool_use_block_from_response_call(item: JsonObject) -> JsonObject:
    """Return an Anthropic tool_use block from one Responses function call."""
    return {
        "type": "tool_use",
        "id": str(item.get("call_id") or item.get("id", "")),
        "name": str(item.get("name", "")),
        "input": parse_tool_arguments(item.get("arguments", "{}")),
    }


def responses_to_anthropic_response(response: JsonObject, model: str) -> JsonObject:
    """Convert a Responses object into Anthropic Messages response JSON."""
    output = response.get("output")
    output_items = cast("JsonArray", output) if isinstance(output, list) else cast("JsonArray", [])
    content = _anthropic_content_from_responses(output_items)
    usage = response.get("usage")
    usage_obj = usage if isinstance(usage, dict) else cast("JsonObject", {})
    has_tool = any(block.get("type") == "tool_use" for block in content)
    return cast(
        "JsonObject",
        {
            "id": str(response.get("id", "msg_routed")),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if has_tool else "end_turn",
            "usage": {
                "input_tokens": jsonnum.int_value(usage_obj.get("input_tokens", 0)),
                "output_tokens": jsonnum.int_value(usage_obj.get("output_tokens", 0)),
            },
        },
    )


def anthropic_to_responses(anthropic_response: JsonObject, model: str) -> JsonObject:
    """Convert an Anthropic response to non-streaming Responses JSON."""
    usage = anthropic_response.get("usage")
    usage_obj: JsonObject = usage if isinstance(usage, dict) else {}
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
