"""Adversarial tests for wire-aware forward-body compression.

Covers the contract of ``forward_compression``: DeepSeek upstreams and
unparseable, non-object, or unknown-wire bodies relay byte-identical; any
compressor error fails open; a turn that shrinks nothing relays byte-identical;
and a real shrink is mapped back into the originating wire by tool-call id while
the input body is never mutated. The headroom-backed ``compress_litellm_messages``
wrapper is stubbed so the map-back and measurement logic can be asserted without
depending on headroom's near-limit gating heuristics.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ai_calls_router.accounting.shrink_stats import ShrinkStats
from ai_calls_router.routing import forward_compression

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject

_SHRUNK = "S"  # Stand-in compressed tool output, far shorter than any input.


def _shrink_tool_messages(
    messages: JsonArray, *, model: str, model_limit: int = 0
) -> tuple[JsonArray, ShrinkStats]:
    """Stub compressor: collapse every role=tool content to a 1-char string."""
    out: list[JsonObject] = []
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
        ):
            out.append({**msg, "content": _SHRUNK})
        else:
            out.append(msg)  # type: ignore[arg-type]
    return out, ShrinkStats("compress", 0, 0)


def _identity(
    messages: JsonArray, *, model: str, model_limit: int = 0
) -> tuple[JsonArray, ShrinkStats]:
    """Stub compressor that changes nothing (no tool output is shrinkable)."""
    return messages, ShrinkStats("none", 0, 0)


def _boom(*_args: object, **_kwargs: object) -> tuple[JsonArray, ShrinkStats]:
    """Stub compressor that raises, standing in for a headroom failure."""
    raise RuntimeError("headroom exploded")


def _patch_compressor(monkeypatch: pytest.MonkeyPatch, fn: object) -> None:
    monkeypatch.setattr(forward_compression, "compress_litellm_messages", fn)


def _anthropic_body(*tool_outputs: tuple[str, str]) -> JsonObject:
    """Build an Anthropic body with assistant tool_use + user tool_result blocks."""
    tool_use = [
        {"type": "tool_use", "id": tid, "name": "exec", "input": {}} for tid, _ in tool_outputs
    ]
    tool_result = [
        {"type": "tool_result", "tool_use_id": tid, "content": text} for tid, text in tool_outputs
    ]
    return {
        "model": "claude-test",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": tool_use},
            {"role": "user", "content": tool_result},
        ],
    }


def _responses_body(*tool_outputs: tuple[str, str]) -> JsonObject:
    """Build a Responses body with function_call + function_call_output items."""
    items: list[JsonObject] = [{"type": "message", "role": "user", "content": "go"}]
    for call_id, text in tool_outputs:
        items.append(
            {"type": "function_call", "call_id": call_id, "name": "exec", "arguments": "{}"}
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": text})
    return {"model": "gpt-test", "input": items}


# ── skip / fail-open contract (no compressor stub needed) ──────────────────


def test_deepseek_upstream_relays_byte_identical_without_compressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressor(monkeypatch, _boom)  # would raise if compression ran
    raw = json.dumps(_anthropic_body(("call_1", "x" * 5000))).encode()
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.deepseek.com/v1"
    )
    assert out is raw
    assert stats.chars_saved == 0
    assert stats.path == "none"


def test_unparseable_body_fails_open_to_original_bytes() -> None:
    raw = b"not json {{{"
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.anthropic.com"
    )
    assert out is raw
    assert stats.chars_saved == 0


def test_non_object_json_body_relays_unchanged() -> None:
    raw = b"[1, 2, 3]"
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.anthropic.com"
    )
    assert out is raw
    assert stats.path == "none"


def test_unknown_wire_path_relays_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    raw = json.dumps(_anthropic_body(("call_1", "x" * 5000))).encode()
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/embeddings", upstream="https://api.openai.com"
    )
    assert out is raw
    assert stats.chars_saved == 0


def test_compressor_exception_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _boom)
    raw = json.dumps(_anthropic_body(("call_1", "x" * 5000))).encode()
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.anthropic.com"
    )
    assert out is raw
    assert stats.path == "none"


def test_no_savings_returns_byte_identical_object(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _identity)
    raw = json.dumps(_anthropic_body(("call_1", "x" * 5000))).encode()
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.anthropic.com"
    )
    assert out is raw  # not re-serialized -> upstream prompt cache stays intact
    assert stats.chars_saved == 0


# ── realized compression + id-keyed map-back ───────────────────────────────


def test_anthropic_shrink_reserializes_and_reports_savings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    raw = json.dumps(_anthropic_body(("call_1", "x" * 5000))).encode()
    out, stats = forward_compression.compress_forward_body(
        raw, request_path="/v1/messages", upstream="https://api.anthropic.com"
    )
    assert out is not raw
    assert stats.chars_saved > 0
    body = json.loads(out)
    block = body["messages"][2]["content"][0]
    assert block["content"] == _SHRUNK


def test_anthropic_maps_each_output_to_its_own_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body = _anthropic_body(("call_1", "a" * 4000), ("call_2", "b" * 4000))
    new_body, stats = forward_compression.compress_anthropic(body)
    results = {b["tool_use_id"]: b["content"] for b in new_body["messages"][2]["content"]}
    assert results == {"call_1": _SHRUNK, "call_2": _SHRUNK}
    assert stats.chars_saved > 0


def test_anthropic_does_not_mutate_input_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body = _anthropic_body(("call_1", "x" * 4000))
    before = json.dumps(body, sort_keys=True)
    forward_compression.compress_anthropic(body)
    assert json.dumps(body, sort_keys=True) == before


def test_anthropic_without_tool_results_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body: JsonObject = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    new_body, stats = forward_compression.compress_anthropic(body)
    assert new_body is body
    assert stats.chars_saved == 0


def test_anthropic_non_list_messages_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _boom)  # malformed input must not reach the compressor
    body: JsonObject = {"model": "claude-test", "messages": "not-a-list"}
    new_body, stats = forward_compression.compress_anthropic(body)
    assert new_body is body
    assert stats.chars_saved == 0


def test_responses_non_list_input_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _boom)
    body: JsonObject = {"model": "gpt-test", "input": "not-a-list"}
    new_body, stats = forward_compression.compress_responses(body)
    assert new_body is body
    assert stats.chars_saved == 0


def test_responses_maps_output_back_by_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body = _responses_body(("call_1", "a" * 4000), ("call_2", "b" * 4000))
    new_body, stats = forward_compression.compress_responses(body)
    outputs = {
        i["call_id"]: i["output"] for i in new_body["input"] if i["type"] == "function_call_output"
    }
    assert outputs == {"call_1": _SHRUNK, "call_2": _SHRUNK}
    assert stats.chars_saved > 0


def test_responses_does_not_mutate_input_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body = _responses_body(("call_1", "x" * 4000))
    before = json.dumps(body, sort_keys=True)
    forward_compression.compress_responses(body)
    assert json.dumps(body, sort_keys=True) == before


def test_openai_chat_replaces_messages_when_shrunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_compressor(monkeypatch, _shrink_tool_messages)
    body: JsonObject = {
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "call_1", "content": "x" * 4000},
        ],
    }
    new_body, stats = forward_compression.compress_openai_chat(body)
    assert new_body["messages"][1]["content"] == _SHRUNK
    assert stats.chars_saved > 0


# ── pure helpers ───────────────────────────────────────────────────────────


def test_responses_input_to_openai_preserves_ids_and_skips_non_dict() -> None:
    items = [
        "garbage-non-dict-item",
        {"type": "message", "role": "user", "content": "go"},
        {"type": "function_call", "call_id": "call_9", "name": "exec", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_9", "output": "result"},
    ]
    msgs = forward_compression._responses_input_to_openai(items)
    assert msgs[0] == {"role": "user", "content": "go"}  # non-dict item dropped
    assert msgs[1]["tool_calls"][0]["id"] == "call_9"
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_9", "content": "result"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["bare", {"type": "text", "text": "y"}], "barey"),
        ([{"type": "image"}], ""),
        (42, ""),
        (None, ""),
    ],
)
def test_flatten_text(value: object, expected: str) -> None:
    assert forward_compression._flatten_text(value) == expected  # type: ignore[arg-type]
