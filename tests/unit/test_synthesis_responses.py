"""Tests for OpenAI Responses SSE synthesis.

Goldens use the event names and required fields documented by OpenAI's
streaming guide and accepted by Codex's Responses parser:
https://developers.openai.com/api/docs/guides/streaming-responses and
https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/sse/responses.rs.
"""

from __future__ import annotations

import json

from ai_calls_router._lib.responses_inbound import anthropic_to_responses
from ai_calls_router.routing.synthesis_responses import synthesize_responses_sse


def _events(body: bytes) -> list[dict[str, object]]:
    """Parse generated SSE data payloads."""
    events: list[dict[str, object]] = []
    for raw_event in body.decode("utf-8").strip().split("\n\n"):
        lines = raw_event.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        data = json.loads(lines[1][6:])
        assert data["type"] == lines[0][7:]
        events.append(data)
    return events


def test_text_response_sse_event_sequence_and_completed_output() -> None:
    body = {
        "id": "resp_1",
        "model": "gpt-5-codex",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    payload = b"".join(synthesize_responses_sse(body, "gpt-5-codex"))
    events = _events(payload)
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[4]["delta"] == "hello"
    assert events[-1]["response"] == anthropic_to_responses(body, "gpt-5-codex")
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert payload.endswith(b"\n\n")


def test_tool_call_response_sse_event_sequence_and_arguments() -> None:
    body = {
        "id": "resp_2",
        "model": "gpt-5-codex",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "exec_command", "input": {"cmd": "ls"}}
        ],
        "stop_reason": "tool_use",
    }
    events = _events(b"".join(synthesize_responses_sse(body, "gpt-5-codex")))
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[2]["item"] == {
        "id": "call_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "exec_command",
        "arguments": "",
    }
    assert events[3]["call_id"] == "call_1"
    assert events[3]["name"] == "exec_command"
    assert events[3]["delta"] == '{"cmd": "ls"}'
    assert events[-1]["response"]["output"][0]["arguments"] == '{"cmd": "ls"}'
