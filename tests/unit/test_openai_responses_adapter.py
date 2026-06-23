"""Tests for the OpenAI Responses client adapter."""

from __future__ import annotations

import pytest

from ai_calls_router.routing.adapters.openai_responses import OpenAIResponsesAdapter


def test_default_agent_group_is_hermes() -> None:
    assert OpenAIResponsesAdapter().default_agent_group == "hermes"


def test_extract_pending_tools_opener_returns_empty() -> None:
    body = {"input": [{"type": "message", "role": "user", "content": "hi"}]}
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == []


def test_extract_pending_tools_resolves_single_function_call_output() -> None:
    body = {
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
    }
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == ["exec_command"]


def test_extract_pending_tools_resolves_parallel_last_run_ordered_deduped() -> None:
    body = {
        "input": [
            {"type": "function_call", "call_id": "a", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "write_stdin", "arguments": "{}"},
            {"type": "function_call", "call_id": "c", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "ok"},
            {"type": "function_call_output", "call_id": "b", "output": "ok"},
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
        ]
    }
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == ["exec_command", "write_stdin"]


def test_extract_pending_tools_unknown_call_id_returns_unknown() -> None:
    body = {"input": [{"type": "function_call_output", "call_id": "missing", "output": "ok"}]}
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == ["<unknown>"]


def test_extract_pending_tools_skips_hosted_tool_items() -> None:
    body = {
        "input": [
            {"type": "web_search_call", "id": "ws_1"},
            {"type": "message", "role": "user", "content": "done"},
        ]
    }
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == []


def test_extract_pending_tools_resolves_custom_tool_output() -> None:
    body = {
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "patch_1",
                "name": "apply_patch",
                "input": "***",
            },
            {"type": "custom_tool_call_output", "call_id": "patch_1", "output": "ok"},
        ]
    }
    assert OpenAIResponsesAdapter().extract_pending_tools(body) == ["apply_patch"]


def test_to_anthropic_request_validates_request_body_before_conversion() -> None:
    adapter = OpenAIResponsesAdapter()

    with pytest.raises(ValueError, match="Field required"):
        adapter.to_anthropic_request({"input": "hello"})
