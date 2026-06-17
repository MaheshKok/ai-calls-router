"""Tests for OpenAI Responses edge conversion.

The router is stateless and converts only self-contained Responses input into
Anthropic Messages. These tests pin deterministic request conversion,
reasoning stripping, tool mapping, and non-streaming response synthesis.
"""

from __future__ import annotations

import copy
import json

import pytest

from ai_calls_router._lib.responses_inbound import (
    anthropic_to_responses,
    responses_input_to_anthropic_messages,
    responses_request_to_anthropic,
    responses_tool_to_anthropic,
)


def test_input_string_becomes_single_user_message() -> None:
    assert responses_input_to_anthropic_messages("hello") == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_function_call_converts_to_tool_use_with_parsed_arguments() -> None:
    messages = responses_input_to_anthropic_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
            }
        ]
    )
    assert messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "exec_command", "input": {"cmd": "ls"}}
            ],
        }
    ]


def test_invalid_function_arguments_pass_through_per_parser_contract() -> None:
    messages = responses_input_to_anthropic_messages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{bad",
            }
        ]
    )
    assert messages[0]["content"][0]["input"] == "{bad"


def test_function_call_output_links_to_tool_result_id() -> None:
    messages = responses_input_to_anthropic_messages(
        [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}]
    )
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}],
        }
    ]


def test_message_content_variants_and_image_url_convert() -> None:
    messages = responses_input_to_anthropic_messages(
        [
            {
                "role": "user",
                "content": [
                    "raw",
                    {"type": "input_text", "text": "typed"},
                    {"type": "input_image", "image_url": "https://example.test/image.png"},
                ],
            },
            {"type": "message", "role": "assistant", "content": None},
        ]
    )
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "raw"},
                {"type": "text", "text": "typed"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/image.png"},
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def test_input_image_without_url_raises_for_fail_open() -> None:
    with pytest.raises(ValueError, match="unsupported Responses input_image part"):
        responses_input_to_anthropic_messages(
            [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}]
        )


def test_custom_tool_call_preserves_freeform_input() -> None:
    messages = responses_input_to_anthropic_messages(
        [
            {
                "type": "custom_tool_call",
                "call_id": "patch_1",
                "name": "apply_patch",
                "input": "*** Begin",
            }
        ]
    )
    assert messages[0]["content"][0] == {
        "type": "tool_use",
        "id": "patch_1",
        "name": "apply_patch",
        "input": {"input": "*** Begin"},
    }


def test_custom_tool_call_output_links_to_tool_result_id() -> None:
    messages = responses_input_to_anthropic_messages(
        [{"type": "custom_tool_call_output", "call_id": "patch_1", "output": "done"}]
    )
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "patch_1", "content": "done"}],
        }
    ]


def test_reasoning_encrypted_content_is_stripped_from_routed_body() -> None:
    body = {
        "model": "gpt-5-codex",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "encrypted_content": "secret-bytes"},
        ],
    }
    converted = responses_request_to_anthropic(body)
    assert "secret-bytes" not in json.dumps(converted)
    assert converted["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_instructions_and_system_messages_become_top_level_system() -> None:
    body = {
        "model": "gpt-5-codex",
        "instructions": "be terse",
        "input": [{"type": "message", "role": "system", "content": "system note"}],
    }
    assert responses_request_to_anthropic(body) == {
        "model": "gpt-5-codex",
        "system": "be terse\nsystem note",
        "messages": [],
    }


def test_response_tools_convert_to_anthropic_schemas() -> None:
    assert responses_tool_to_anthropic(
        {
            "type": "function",
            "name": "exec_command",
            "description": "run",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ) == {
        "name": "exec_command",
        "description": "run",
        "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
    }
    assert responses_tool_to_anthropic({"type": "custom", "name": "apply_patch"}) == {
        "name": "apply_patch",
        "input_schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
    }


def test_tool_conversion_rejects_missing_name_and_unknown_type() -> None:
    with pytest.raises(ValueError, match="requires name"):
        responses_tool_to_anthropic({"type": "function"})
    with pytest.raises(ValueError, match="unsupported Responses tool type"):
        responses_tool_to_anthropic({"type": "hosted", "name": "web_search"})


def test_malformed_item_raises_for_fail_open_path() -> None:
    with pytest.raises(ValueError, match="function_call requires"):
        responses_input_to_anthropic_messages([{"type": "function_call", "call_id": "missing"}])


@pytest.mark.parametrize(
    ("input_items", "match"),
    [
        ({"bad": "shape"}, "string or list"),
        ([1], "items must be objects"),
        ([{"type": "message", "role": "bad", "content": "x"}], "role must be"),
        ([{"type": "unknown"}], "unsupported Responses input item type"),
        ([{"type": "input_image", "image_url": ""}], "unsupported Responses input item type"),
        ([{"type": "custom_tool_call", "call_id": "c"}], "custom_tool_call requires"),
        ([{"type": "function_call_output", "call_id": "c"}], "tool output requires"),
    ],
)
def test_input_malformed_branches_raise(*, input_items: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        responses_input_to_anthropic_messages(input_items)


def test_request_conversion_is_deterministic_and_immutable() -> None:
    body: dict[str, object] = {
        "model": "gpt-5-codex",
        "instructions": "rules",
        "input": [
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": '{"first":1,"second":2}',
            },
        ],
        "max_output_tokens": 100,
        "temperature": 0,
    }
    original = copy.deepcopy(body)
    one = responses_request_to_anthropic(body)
    two = responses_request_to_anthropic(body)
    assert body == original
    assert json.dumps(one, ensure_ascii=False, sort_keys=False) == json.dumps(
        two,
        ensure_ascii=False,
        sort_keys=False,
    )
    assert "opaque" not in json.dumps(one)
    assert list(one["messages"][0]["content"][0]["input"]) == ["first", "second"]


def test_request_conversion_maps_controls_in_stable_order() -> None:
    body = {
        "model": "gpt-5-codex",
        "input": "hi",
        "tools": [{"type": "function", "name": "exec_command"}],
        "tool_choice": {"type": "function", "name": "exec_command"},
        "max_output_tokens": 10,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["END"],
        "store": True,
    }
    converted = responses_request_to_anthropic(body)
    assert list(converted) == [
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
    ]
    assert "store" not in converted


def test_missing_input_and_tool_choice_variants() -> None:
    with pytest.raises(ValueError, match="requires input"):
        responses_request_to_anthropic({"model": "gpt-5-codex"})
    assert responses_request_to_anthropic({"model": "m", "input": "hi", "tool_choice": "required"})[
        "tool_choice"
    ] == {"type": "any"}
    assert responses_request_to_anthropic({"model": "m", "input": "hi", "tool_choice": "auto"})[
        "tool_choice"
    ] == {"type": "auto"}
    assert responses_request_to_anthropic(
        {"model": "m", "input": "hi", "tool_choice": {"type": "function"}}
    )["tool_choice"] == {"type": "auto"}


def test_anthropic_to_responses_text_tool_mixed_and_usage() -> None:
    response = anthropic_to_responses(
        {
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "exec_command",
                    "input": {"cmd": "ls"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
        "gpt-5-codex",
    )
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["model"] == "gpt-5-codex"
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    assert response["output"][0]["content"][0] == {"type": "output_text", "text": "hello"}
    assert response["output"][1] == {
        "id": "call_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": '{"cmd": "ls"}',
    }


def test_anthropic_to_responses_skips_malformed_blocks_and_empty_usage() -> None:
    response = anthropic_to_responses(
        {
            "id": "msg_2",
            "content": [None, {"type": "ignored"}, {"type": "text", "text": "ok"}],
        },
        "m",
    )
    assert response["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert response["output"] == [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
    ]


@pytest.mark.parametrize(
    ("stop_reason", "status", "details"),
    [
        ("end_turn", "completed", None),
        ("tool_use", "completed", None),
        ("max_tokens", "incomplete", {"reason": "max_output_tokens"}),
    ],
)
def test_anthropic_to_responses_status_mapping(
    *, stop_reason: str, status: str, details: dict[str, str] | None
) -> None:
    response = anthropic_to_responses({"content": [], "stop_reason": stop_reason}, "m")
    assert response["status"] == status
    assert response.get("incomplete_details") == details
