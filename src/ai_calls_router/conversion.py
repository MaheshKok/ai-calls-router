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
from typing import Any


@dataclass
class BackendResponse:
    """Standardized routed-call response in Anthropic Messages format.

    Attributes:
        body: Response body (Anthropic Messages API format).
        status_code: HTTP status code.
        headers: Response headers to forward.
        error: Error message, if any.
    """

    body: dict[str, Any]
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def convert_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool definition to OpenAI function format.

    Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}

    Args:
        tool: Anthropic tool definition.

    Returns:
        OpenAI-format function tool definition.
    """
    func: dict[str, Any] = {"name": tool.get("name", "")}
    if "description" in tool:
        func["description"] = tool["description"]
    if "input_schema" in tool:
        func["parameters"] = tool["input_schema"]
    return {"type": "function", "function": func}


def convert_tool_choice(choice: Any) -> Any:
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


def parse_tool_arguments(arguments: Any) -> Any:
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
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return arguments
    return arguments


def convert_messages_for_litellm(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            tool_use_blocks: list[dict[str, Any]] = []
            tool_result_blocks: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_use_blocks.append(block)
                elif block_type == "tool_result":
                    tool_result_blocks.append(block)

            if tool_result_blocks:
                for tr in tool_result_blocks:
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = "\n".join(
                            b.get("text", "") for b in tr_content if b.get("type") == "text"
                        )
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": str(tr_content),
                        }
                    )
                continue

            if tool_use_blocks:
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                assistant_msg["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_use_blocks
                ]
                converted.append(assistant_msg)
                continue

            if text_parts:
                converted.append({"role": role, "content": "\n".join(text_parts)})
            else:
                converted.append({"role": role, "content": ""})

    for msg in converted:
        if msg.get("role") == "assistant":
            msg.setdefault("reasoning_content", "")
    return converted


def to_anthropic_response(litellm_response: Any, original_model: str) -> dict[str, Any]:
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

    content: list[dict[str, Any]] = []
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
    stop_reason = stop_reason_map.get(choice.finish_reason, "end_turn")

    usage = {
        "input_tokens": getattr(litellm_response.usage, "prompt_tokens", 0),
        "output_tokens": getattr(litellm_response.usage, "completion_tokens", 0),
    }

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def completion_kwargs(body: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
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
    kwargs: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": convert_messages_for_litellm(body.get("messages", [])),
    }

    if "max_tokens" in body:
        kwargs["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        kwargs["stop"] = body["stop_sequences"]

    if "tools" in body:
        kwargs["tools"] = [convert_anthropic_tool(t) for t in body["tools"]]
    if "tool_choice" in body:
        kwargs["tool_choice"] = convert_tool_choice(body["tool_choice"])

    if "system" in body:
        system = body["system"]
        if isinstance(system, str):
            kwargs["messages"].insert(0, {"role": "system", "content": system})
        elif isinstance(system, list):
            system_text = " ".join(
                s.get("text", "") if isinstance(s, dict) else str(s) for s in system
            )
            kwargs["messages"].insert(0, {"role": "system", "content": system_text})

    if api_key:
        kwargs["api_key"] = api_key

    return kwargs
