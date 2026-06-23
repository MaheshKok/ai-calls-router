"""Spec-derived tests for OpenAI Chat inbound conversion.

These tests pin the edge conversion contract for Hermes Chat traffic: request
conversion must be deterministic and immutable, response conversion must
preserve tool-call identity, and pending-tool extraction must fail open on
unresolvable ids.
"""

from __future__ import annotations

import copy
import json

import pytest

from ai_calls_router._lib.openai_inbound import (
    anthropic_to_chat_response,
    chat_request_to_anthropic,
    openai_chat_to_anthropic_messages,
    openai_tool_to_anthropic,
)
from ai_calls_router.routing.adapters.openai_chat import OpenAIChatAdapter


def _tool_call() -> dict[str, object]:
    """Return an OpenAI function tool call with ordered arguments."""
    return {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "terminal", "arguments": '{"b": 2, "a": 1}'},
    }


def test_assistant_tool_calls_convert_to_tool_use_blocks() -> None:
    messages = [{"role": "assistant", "content": None, "tool_calls": [_tool_call()]}]
    converted = openai_chat_to_anthropic_messages(messages)
    assert converted == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "terminal",
                    "input": {"b": 2, "a": 1},
                }
            ],
        }
    ]


def test_tool_message_converts_to_tool_result_with_verbatim_id() -> None:
    converted = openai_chat_to_anthropic_messages(
        [{"role": "tool", "tool_call_id": "call_abc", "content": "stdout"}]
    )
    assert converted == [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_abc", "content": "stdout"}],
        }
    ]


def test_tool_definition_conversion() -> None:
    converted = openai_tool_to_anthropic(
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command",
                "parameters": {"type": "object"},
            },
        }
    )
    assert converted == {
        "name": "terminal",
        "description": "Run a command",
        "input_schema": {"type": "object"},
    }


def test_chat_request_handles_system_and_text_turns() -> None:
    converted = chat_request_to_anthropic(
        {
            "model": "openai/gpt-test",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "max_completion_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
        }
    )
    assert list(converted) == [
        "model",
        "system",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
    ]
    assert converted["system"] == "policy"
    assert converted["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    assert converted["stop_sequences"] == ["END"]


def test_chat_request_handles_developer_as_system() -> None:
    converted = chat_request_to_anthropic(
        {
            "model": "openai/gpt-test",
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "developer", "content": "style"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert converted["system"] == "policy\nstyle"
    assert converted["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_list_content_and_unknown_roles_are_handled() -> None:
    converted = openai_chat_to_anthropic_messages(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}, "world", {"type": "image"}],
            },
            {"role": "developer", "content": "ignored"},
            {"role": "assistant", "content": 42},
        ]
    )
    assert converted == [
        {"role": "user", "content": [{"type": "text", "text": "hello\nworld"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "42"}]},
    ]


def test_tool_choice_conversion_variants() -> None:
    required = chat_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": "required"}
    )
    named = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "function", "function": {"name": "terminal"}},
        }
    )
    unknown = chat_request_to_anthropic(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "tool_choice": 123}
    )
    assert required["tool_choice"] == {"type": "any"}
    assert named["tool_choice"] == {"type": "tool", "name": "terminal"}
    assert unknown["tool_choice"] == {"type": "auto"}


def test_malformed_messages_raise_for_fail_open_server_path() -> None:
    with pytest.raises(ValueError, match="chat messages"):
        chat_request_to_anthropic({"model": "m", "messages": "bad"})


def test_chat_schema_rejects_bad_envelope_fields() -> None:
    with pytest.raises(ValueError, match="String should have at least 1 character"):
        chat_request_to_anthropic({"model": "", "messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(ValueError, match="greater than 0"):
        chat_request_to_anthropic(
            {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 0}
        )


def test_request_conversion_is_immutable() -> None:
    body = {
        "model": "openai/gpt-test",
        "messages": [{"role": "assistant", "content": None, "tool_calls": [_tool_call()]}],
        "tools": [{"type": "function", "function": {"name": "terminal"}}],
    }
    before = copy.deepcopy(body)
    chat_request_to_anthropic(body)
    assert body == before


def test_argument_key_order_survives_parsing_and_response_round_trip() -> None:
    converted = chat_request_to_anthropic(
        {"model": "m", "messages": [{"role": "assistant", "tool_calls": [_tool_call()]}]}
    )
    tool_input = converted["messages"][0]["content"][0]["input"]
    assert list(tool_input) == ["b", "a"]

    chat = anthropic_to_chat_response(
        {
            "id": "msg_1",
            "model": "m",
            "content": [
                {"type": "tool_use", "id": "call_abc", "name": "terminal", "input": tool_input}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )
    arguments = chat["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert list(json.loads(arguments)) == ["b", "a"]


def test_unparseable_arguments_string_passes_through() -> None:
    converted = openai_chat_to_anthropic_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "function": {"name": "terminal", "arguments": "{not-json"},
                    }
                ],
            }
        ]
    )
    assert converted[0]["content"][0]["input"] == "{not-json"


def test_byte_stability_for_shared_prefix() -> None:
    prefix = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "line1\nline2"},
    ]
    turn_n = chat_request_to_anthropic({"model": "m", "messages": prefix})
    turn_n_plus_1 = chat_request_to_anthropic(
        {"model": "m", "messages": [*prefix, {"role": "user", "content": "continue"}]}
    )

    prefix_json = json.dumps(turn_n["messages"], ensure_ascii=False)
    longer_prefix_json = json.dumps(
        turn_n_plus_1["messages"][: len(turn_n["messages"])], ensure_ascii=False
    )
    assert longer_prefix_json == prefix_json


def test_anthropic_to_chat_response_text_only() -> None:
    chat = anthropic_to_chat_response(
        {
            "id": "msg_1",
            "model": "deepseek-v4",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    )
    assert chat["id"] == "msg_1"
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "hello"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_anthropic_to_chat_response_tool_call_and_mixed_content() -> None:
    chat = anthropic_to_chat_response(
        {
            "id": "msg_2",
            "model": "deepseek-v4",
            "content": [
                {"type": "text", "text": "using tool"},
                {"type": "tool_use", "id": "call_1", "name": "terminal", "input": {"cmd": "ls"}},
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
    )
    message = chat["choices"][0]["message"]
    assert message["content"] == "using tool"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_to_chat_response_empty_content() -> None:
    chat = anthropic_to_chat_response({"id": "msg_3", "model": "m", "content": []})
    assert chat["choices"][0]["message"] == {"role": "assistant", "content": None}
    assert chat["choices"][0]["finish_reason"] == "stop"


def test_anthropic_to_chat_response_skips_malformed_blocks_and_content() -> None:
    non_list = anthropic_to_chat_response({"id": "msg_4", "model": "m", "content": "plain"})
    mixed = anthropic_to_chat_response(
        {
            "id": "msg_5",
            "model": "m",
            "content": [None, {"type": "unknown"}, {"type": "text", "text": "kept"}],
        }
    )
    assert non_list["choices"][0]["message"]["content"] is None
    assert mixed["choices"][0]["message"]["content"] == "kept"


def test_extract_pending_tools_resolves_ordered_deduped_names() -> None:
    adapter = OpenAIChatAdapter()
    body = {
        "messages": [
            {"role": "assistant", "tool_calls": [_tool_call(), {**_tool_call(), "id": "call_def"}]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "one"},
            {"role": "tool", "tool_call_id": "call_def", "content": "two"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "three"},
        ]
    }
    assert adapter.extract_pending_tools(body) == ["terminal"]


def test_extract_pending_tools_unresolvable_id_returns_unknown() -> None:
    body = {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "x"}]}
    assert OpenAIChatAdapter().extract_pending_tools(body) == ["<unknown>"]


def test_extract_pending_tools_turn_opener_returns_empty() -> None:
    adapter = OpenAIChatAdapter()
    assert adapter.extract_pending_tools({"messages": [{"role": "user", "content": "hello"}]}) == []
    assert adapter.extract_pending_tools({"messages": []}) == []
    assert adapter.extract_pending_tools({"messages": "bad"}) == []


def test_extract_pending_tools_skips_malformed_prior_tool_calls() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": ["bad", {"id": "x"}, {"id": "y", "function": "bad"}],
            },
            {"role": "tool", "tool_call_id": "y", "content": "out"},
        ]
    }
    assert OpenAIChatAdapter().extract_pending_tools(body) == ["<unknown>"]


def test_chat_adapter_validates_request_body_before_conversion() -> None:
    adapter = OpenAIChatAdapter()

    with pytest.raises(ValueError, match="Field required"):
        adapter.to_anthropic_request({"messages": [{"role": "user", "content": "hello"}]})
