"""Tests for OpenAI Chat Completions SSE synthesis.

The routed engine returns buffered Anthropic responses, so this module verifies
that streaming Chat clients receive deterministic chunk sequences. Goldens are
written by hand to pin the exact wire bytes.
"""

from __future__ import annotations

from ai_calls_router.routing.synthesis_openai import synthesize_chat_sse


def test_text_response_sse_matches_golden() -> None:
    body = {
        "id": "msg_1",
        "model": "deepseek-v4",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
    }
    expected = (
        'data: {"id": "msg_1", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {"role": "assistant"}, '
        '"finish_reason": null}]}\n\n'
        'data: {"id": "msg_1", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {"content": "hello"}, '
        '"finish_reason": null}]}\n\n'
        'data: {"id": "msg_1", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {}, '
        '"finish_reason": "stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    assert b"".join(synthesize_chat_sse(body)) == expected.encode("utf-8")


def test_tool_call_response_sse_matches_golden() -> None:
    body = {
        "id": "msg_2",
        "model": "deepseek-v4",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "terminal", "input": {"cmd": "ls"}}
        ],
        "stop_reason": "tool_use",
    }
    expected = (
        'data: {"id": "msg_2", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {"role": "assistant"}, '
        '"finish_reason": null}]}\n\n'
        'data: {"id": "msg_2", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, '
        '"id": "call_1", "type": "function", "function": {"name": "terminal", '
        '"arguments": "{\\"cmd\\": \\"ls\\"}"}}]}, "finish_reason": null}]}\n\n'
        'data: {"id": "msg_2", "object": "chat.completion.chunk", "created": 0, '
        '"model": "deepseek-v4", "choices": [{"index": 0, "delta": {}, '
        '"finish_reason": "tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    )
    assert b"".join(synthesize_chat_sse(body)) == expected.encode("utf-8")


def test_sse_stream_ends_with_done() -> None:
    body = {"id": "msg_3", "model": "m", "content": []}
    assert b"".join(synthesize_chat_sse(body)).endswith(b"data: [DONE]\n\n")
