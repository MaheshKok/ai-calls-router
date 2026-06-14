"""Spec-derived tests for the routing engine and its SSE synthesis.

Contract under test: _prepare_routed_body swaps in the tier model, clamps
max_tokens, and strips Claude-specific thinking blocks without mutating the
input; escalates flags routed responses that call premium tools; routed_call
serves a request on the tier provider with only the tier key, records honest
savings under the true routed model, masks the served body to the client's
requested model, and returns None on escalation or any provider failure
(invariants 1-4); synthesis.synthesize_sse renders a finished Anthropic response
as the Messages SSE stream Claude Code expects.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_calls_router.accounting import savings
from ai_calls_router.routing import engine as rc
from ai_calls_router.routing import synthesis
from tests.acr_testkit import FakeLitellm
from tests.acr_testkit import make_response as _fake_response

# Non-DeepSeek ids keep these tests on the LiteLLM path: DeepSeek tier models
# are diverted to the native direct path (see TestRoutedCallDeepSeekDirect).
CHEAP_MODEL = "groq/acr-test-cheap"
PREMIUM_MODEL = "groq/acr-test-premium"

PRICED_ROUTES: dict[str, Any] = {
    "tiers": {
        "fast": {
            "model": CHEAP_MODEL,
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 2.0,
        },
        "premium_stand_in": {
            "model": PREMIUM_MODEL,
            "input_cost_per_1m": 10.0,
            "output_cost_per_1m": 20.0,
        },
    }
}

TIER_CFG: dict[str, Any] = {"model": CHEAP_MODEL, "max_tokens": 8192}

SETTINGS: dict[str, Any] = {
    "premium_tools": ["Edit", "Write", "Task", "ExitPlanMode", "AskUserQuestion"],
    "escalate_on_premium_tools": True,
    "compress_routed": False,
}


@pytest.fixture(autouse=True)
def ledger_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register test prices and point the default ledger at a temp file."""
    savings.register_tier_prices(PRICED_ROUTES)
    path = tmp_path / "savings.jsonl"
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(path))
    return path


def _request_body(model: str = PREMIUM_MODEL) -> dict[str, Any]:
    """Build a tool-result-processing Anthropic request body."""
    return {
        "model": model,
        "stream": True,
        "max_tokens": 32000,
        "messages": [
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}],
            },
        ],
    }


def _fake_tool_call(*, call_id: str, name: str, arguments: str) -> Any:
    """Build an OpenAI-shaped tool call object."""
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _call(
    *,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLitellm,
    body: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    api_key: str = "test-tier-key",
) -> Any:
    """Run routed_call against the fake litellm module."""
    monkeypatch.setattr(rc, "load_litellm", lambda: fake)
    return asyncio.run(
        rc.routed_call(
            body=body if body is not None else _request_body(),
            tier_name="fast",
            tier_cfg=TIER_CFG,
            api_key=api_key,
            settings=settings if settings is not None else SETTINGS,
        )
    )


class TestPrepareRoutedBody:
    def test_model_swapped_to_tier_model(self) -> None:
        routed = rc._prepare_routed_body(_request_body(), TIER_CFG)
        assert routed["model"] == CHEAP_MODEL

    def test_stream_flag_removed(self) -> None:
        routed = rc._prepare_routed_body(_request_body(), TIER_CFG)
        assert "stream" not in routed

    def test_max_tokens_clamped_when_over_tier_limit(self) -> None:
        body = _request_body()
        body["max_tokens"] = 100_000
        routed = rc._prepare_routed_body(body, {"model": CHEAP_MODEL, "max_tokens": 4096})
        assert routed["max_tokens"] == 4096

    def test_max_tokens_set_when_missing(self) -> None:
        body = _request_body()
        del body["max_tokens"]
        routed = rc._prepare_routed_body(body, {"model": CHEAP_MODEL, "max_tokens": 4096})
        assert routed["max_tokens"] == 4096

    def test_max_tokens_below_tier_limit_preserved(self) -> None:
        body = _request_body()
        body["max_tokens"] = 100
        routed = rc._prepare_routed_body(body, {"model": CHEAP_MODEL, "max_tokens": 4096})
        assert routed["max_tokens"] == 100

    @pytest.mark.parametrize("tier_max", [None, 0, -1, "4096", True])
    def test_unusable_tier_max_never_clamps(self, tier_max: Any) -> None:
        body = _request_body()
        routed = rc._prepare_routed_body(body, {"model": CHEAP_MODEL, "max_tokens": tier_max})
        assert routed["max_tokens"] == 32000

    def test_thinking_blocks_stripped(self) -> None:
        body = _request_body()
        body["messages"][1]["content"].insert(
            0, {"type": "thinking", "thinking": "secret", "signature": "s"}
        )
        routed = rc._prepare_routed_body(body, TIER_CFG)
        types = [b["type"] for b in routed["messages"][1]["content"]]
        assert "thinking" not in types
        assert "tool_use" in types

    def test_redacted_thinking_blocks_stripped(self) -> None:
        body = _request_body()
        body["messages"][1]["content"].insert(0, {"type": "redacted_thinking", "data": "opaque"})
        routed = rc._prepare_routed_body(body, TIER_CFG)
        types = [b["type"] for b in routed["messages"][1]["content"]]
        assert "redacted_thinking" not in types

    def test_assistant_message_emptied_by_stripping_is_dropped(self) -> None:
        body = _request_body()
        body["messages"].insert(
            1,
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "only", "signature": "s"}],
            },
        )
        routed = rc._prepare_routed_body(body, TIER_CFG)
        assert len(routed["messages"]) == 3
        assert all(msg.get("content") for msg in routed["messages"] if msg["role"] == "assistant")

    def test_string_content_messages_pass_through(self) -> None:
        routed = rc._prepare_routed_body(_request_body(), TIER_CFG)
        assert routed["messages"][0] == {"role": "user", "content": "run ls"}

    def test_input_body_never_mutated(self) -> None:
        body = _request_body()
        body["messages"][1]["content"].insert(
            0, {"type": "thinking", "thinking": "secret", "signature": "s"}
        )
        snapshot = copy.deepcopy(body)
        rc._prepare_routed_body(body, TIER_CFG)
        assert body == snapshot


class TestEscalates:
    def _response_with_tool(self, name: str) -> dict[str, Any]:
        return {
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "x", "name": name, "input": {}},
            ]
        }

    def test_premium_tool_call_escalates(self) -> None:
        assert rc.escalates(self._response_with_tool("Edit"), SETTINGS) is True

    def test_non_premium_tool_call_does_not_escalate(self) -> None:
        assert rc.escalates(self._response_with_tool("Bash"), SETTINGS) is False

    def test_text_only_response_does_not_escalate(self) -> None:
        body = {"content": [{"type": "text", "text": "summary"}]}
        assert rc.escalates(body, SETTINGS) is False

    def test_disabled_guard_never_escalates(self) -> None:
        settings = {**SETTINGS, "escalate_on_premium_tools": False}
        assert rc.escalates(self._response_with_tool("Edit"), settings) is False

    def test_empty_premium_tools_never_escalates(self) -> None:
        settings = {**SETTINGS, "premium_tools": []}
        assert rc.escalates(self._response_with_tool("Edit"), settings) is False

    def test_missing_settings_keys_default_safe(self) -> None:
        assert rc.escalates(self._response_with_tool("Edit"), {}) is False

    def test_non_list_content_does_not_escalate(self) -> None:
        assert rc.escalates({"content": "plain"}, SETTINGS) is False


class TestRoutedCall:
    def test_success_masks_response_model_to_client_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Invariant 1: the served body claims the client-requested model.
        response = _call(monkeypatch=monkeypatch, fake=FakeLitellm(_fake_response()))
        assert response is not None
        assert response.status_code == 200
        assert response.body["model"] == PREMIUM_MODEL

    def test_ledger_records_true_routed_model(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        # Invariant 4: accounting reads the true routed model, pre-mask.
        _call(monkeypatch=monkeypatch, fake=FakeLitellm(_fake_response()))
        entries = [json.loads(line) for line in ledger_file.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["routed_model"] == CHEAP_MODEL
        assert entries[0]["premium_model"] == PREMIUM_MODEL
        assert entries[0]["saved_usd"] > 0

    def test_provider_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Invariant 3: every failure path falls back to passthrough.
        response = _call(monkeypatch=monkeypatch, fake=FakeLitellm(error=RuntimeError("boom")))
        assert response is None

    def test_no_ledger_entry_when_provider_fails(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        _call(monkeypatch=monkeypatch, fake=FakeLitellm(error=RuntimeError("boom")))
        assert not ledger_file.exists()

    def test_escalating_response_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool_calls = [_fake_tool_call(call_id="c1", name="Edit", arguments='{"file_path": "x"}')]
        fake = FakeLitellm(
            _fake_response(text=None, tool_calls=tool_calls, finish_reason="tool_calls")
        )
        assert _call(monkeypatch=monkeypatch, fake=fake) is None

    def test_no_ledger_entry_when_escalated(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        tool_calls = [_fake_tool_call(call_id="c1", name="Write", arguments='{"file_path": "x"}')]
        fake = FakeLitellm(
            _fake_response(text=None, tool_calls=tool_calls, finish_reason="tool_calls")
        )
        _call(monkeypatch=monkeypatch, fake=fake)
        assert not ledger_file.exists()

    def test_non_premium_tool_call_is_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tool_calls = [_fake_tool_call(call_id="c1", name="Bash", arguments='{"command": "ls"}')]
        fake = FakeLitellm(
            _fake_response(text=None, tool_calls=tool_calls, finish_reason="tool_calls")
        )
        response = _call(monkeypatch=monkeypatch, fake=fake)
        assert response is not None
        blocks = response.body["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["name"] == "Bash"
        assert blocks[0]["input"] == {"command": "ls"}

    def test_only_tier_key_sent_to_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Invariant 2: routed calls carry only the tier key.
        fake = FakeLitellm(_fake_response())
        _call(monkeypatch=monkeypatch, fake=fake, api_key="test-tier-key")
        kwargs = fake.calls[0]
        assert kwargs["api_key"] == "test-tier-key"
        assert not any("auth" in key.lower() for key in kwargs)

    def test_stream_flag_not_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeLitellm(_fake_response())
        _call(monkeypatch=monkeypatch, fake=fake)
        assert "stream" not in fake.calls[0]

    def test_max_tokens_clamped_in_provider_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeLitellm(_fake_response())
        _call(monkeypatch=monkeypatch, fake=fake)
        assert fake.calls[0]["max_tokens"] == 8192
        assert fake.calls[0]["model"] == CHEAP_MODEL

    def test_compression_applied_before_provider_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _request_body()
        body["messages"] = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 5000}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "short"}],
            },
        ]
        settings = {
            **SETTINGS,
            "compress_routed": True,
            "compression": {
                "keep_recent_messages": 1,
                "max_tool_result_chars": 50,
                "use_rtk": "never",
            },
        }
        fake = FakeLitellm(_fake_response())
        _call(monkeypatch=monkeypatch, fake=fake, body=body, settings=settings)
        sent = fake.calls[0]["messages"]
        old_result = next(m for m in sent if m.get("tool_call_id") == "t1")
        assert "truncated" in old_result["content"]
        assert len(old_result["content"]) < 200
        recent_result = next(m for m in sent if m.get("tool_call_id") == "t2")
        assert recent_result["content"] == "short"

    def test_zero_usage_serves_response_without_ledger_entry(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        # Invariant 5: no fabricated costs; zero-token usage writes nothing.
        fake = FakeLitellm(_fake_response(prompt_tokens=0, completion_tokens=0))
        response = _call(monkeypatch=monkeypatch, fake=fake)
        assert response is not None
        assert not ledger_file.exists()


# --- DeepSeek direct path -------------------------------------------------

DS_MODEL = "deepseek/deepseek-v4-pro"

# Cache-aware tier prices: misses cost 120x a hit, exercised by the savings math.
DS_TIER: dict[str, Any] = {
    "model": DS_MODEL,
    "max_tokens": 8192,
    "input_cost_per_1m": 0.435,
    "input_cached_cost_per_1m": 0.003625,
    "output_cost_per_1m": 0.87,
}


def _direct_body(
    *,
    text: str | None = "routed reply",
    input_tokens: int = 200,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_creation: int = 0,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Build a DeepSeek-native Anthropic response body with a cache usage block.

    The model field carries the native id (no provider prefix), mirroring what
    the DeepSeek endpoint actually returns, so masking has something to rewrite.
    """
    if tool_name is not None:
        content: list[dict[str, Any]] = [
            {"type": "tool_use", "id": "t9", "name": tool_name, "input": {}}
        ]
    else:
        content = [{"type": "text", "text": text or ""}]
    return {
        "id": "msg_ds",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-pro",
        "content": content,
        "stop_reason": "tool_use" if tool_name else "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }


def _call_direct(
    monkeypatch: pytest.MonkeyPatch,
    direct_response: dict[str, Any] | None | Exception,
    *,
    body: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    api_key: str = "ds-tier-key",
    tier_cfg: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], list[Any]]:
    """Run routed_call on the DeepSeek direct path with direct_call stubbed.

    Guards that the direct path never touches LiteLLM or the compressor, and
    captures the arguments handed to direct_call so the caller can assert
    credential isolation and body preparation.

    Returns:
        (routed_call result, captured direct_call args, compressor call log).
    """
    captured: dict[str, Any] = {}

    async def _fake_direct(*, body: dict[str, Any], tier_cfg: dict[str, Any], api_key: str) -> Any:
        captured["body"] = body
        captured["tier_cfg"] = tier_cfg
        captured["api_key"] = api_key
        if isinstance(direct_response, Exception):
            raise direct_response
        return direct_response

    def _no_litellm() -> Any:
        raise AssertionError("load_litellm must not run on the direct path")

    compress_calls: list[Any] = []

    def _spy_compress(b: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
        compress_calls.append(b)
        return b

    monkeypatch.setattr(rc.anthropic_direct, "direct_call", _fake_direct)
    monkeypatch.setattr(rc, "load_litellm", _no_litellm)
    monkeypatch.setattr(rc.compression, "compress_body", _spy_compress)

    result = asyncio.run(
        rc.routed_call(
            body=body if body is not None else _request_body(),
            tier_name="code",
            tier_cfg=tier_cfg if tier_cfg is not None else DS_TIER,
            api_key=api_key,
            settings=settings if settings is not None else SETTINGS,
        )
    )
    return result, captured, compress_calls


class TestRoutedCallDeepSeekDirect:
    def test_deepseek_tier_takes_direct_path_not_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The guard in _call_direct raises if load_litellm runs; reaching a
        # served response proves the DeepSeek tier bypassed LiteLLM entirely.
        result, captured, _ = _call_direct(monkeypatch, _direct_body())
        assert result is not None
        assert captured["api_key"] == "ds-tier-key"

    def test_direct_path_skips_compression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # DeepSeek's prefix cache outperforms our compression, so the direct
        # path must never compress -- compressing would break byte-identical
        # prefixes and defeat the cache.
        settings = {**SETTINGS, "compress_routed": True}
        _, _, compress_calls = _call_direct(monkeypatch, _direct_body(), settings=settings)
        assert compress_calls == []

    def test_direct_path_prepares_body_and_strips_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _prepare_routed_body still runs: model swapped to the tier id, the
        # stream flag dropped, max_tokens clamped to the tier ceiling.
        _, captured, _ = _call_direct(monkeypatch, _direct_body())
        sent = captured["body"]
        assert sent["model"] == DS_MODEL
        assert "stream" not in sent
        assert sent["max_tokens"] == 8192

    def test_direct_path_reduces_tool_result_noise_before_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Deterministic reduction must run before the body reaches the provider:
        # ANSI escapes and blank-line runs in tool_result content are stripped so
        # fewer tokens go on the wire, and because the reduction is a pure
        # function the prefix stays byte-stable across turns (cache-safe).
        noisy_request = {
            "model": PREMIUM_MODEL,
            "max_tokens": 32000,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "\x1b[32mok\x1b[0m\n\n\n\ntail",
                        }
                    ],
                },
            ],
        }
        _, captured, _ = _call_direct(monkeypatch, _direct_body(), body=noisy_request)
        sent_result = captured["body"]["messages"][-1]["content"][0]["content"]
        assert sent_result == "ok\n\ntail"

    def test_direct_path_forwards_only_tier_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Invariant 2: the client's credential never reaches the routed
        # provider; only the tier key is handed to direct_call.
        _, captured, _ = _call_direct(monkeypatch, _direct_body(), api_key="only-the-tier-key")
        assert captured["api_key"] == "only-the-tier-key"

    def test_direct_path_masks_response_to_client_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Invariant 1: the served body claims the client-requested model, not
        # the DeepSeek native id the endpoint returned.
        result, _, _ = _call_direct(monkeypatch, _direct_body())
        assert result.body["model"] == PREMIUM_MODEL

    def test_direct_path_escalates_on_premium_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _, _ = _call_direct(monkeypatch, _direct_body(text=None, tool_name="Edit"))
        assert result is None

    def test_direct_path_no_ledger_entry_when_escalated(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        _call_direct(monkeypatch, _direct_body(text=None, tool_name="Write"))
        assert not ledger_file.exists()

    def test_direct_path_none_response_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        # Invariant 3: a failed direct call (None) must fall back to passthrough
        # and record nothing.
        result, _, _ = _call_direct(monkeypatch, None)
        assert result is None
        assert not ledger_file.exists()

    def test_direct_path_exception_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        result, _, _ = _call_direct(monkeypatch, RuntimeError("transport down"))
        assert result is None
        assert not ledger_file.exists()

    def test_direct_path_records_cache_aware_savings(
        self, monkeypatch: pytest.MonkeyPatch, ledger_file: Path
    ) -> None:
        # 900k cache-read tokens billed at the hit rate, 100k creation at the
        # miss rate, 1k output. Premium counterfactual prices the full 1M prompt
        # at groq/acr-test-premium (10/20 per 1M).
        #   routed = 100_000*0.435/1e6 + 900_000*0.003625/1e6 + 1_000*0.87/1e6
        #          = 0.0435 + 0.0032625 + 0.00087 = 0.0476325
        #   premium = 1_000_000*10/1e6 + 1_000*20/1e6 = 10.00 + 0.02 = 10.02
        result, _, _ = _call_direct(
            monkeypatch,
            _direct_body(
                input_tokens=0,
                output_tokens=1_000,
                cache_read=900_000,
                cache_creation=100_000,
            ),
        )
        assert result is not None
        entries = [json.loads(line) for line in ledger_file.read_text().splitlines()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["routed_model"] == DS_MODEL
        assert entry["premium_model"] == PREMIUM_MODEL
        assert entry["input_tokens"] == 1_000_000
        assert entry["cache_read_input_tokens"] == 900_000
        assert entry["cache_creation_input_tokens"] == 100_000
        assert entry["routed_usd"] == pytest.approx(0.0476325, abs=1e-7)
        assert entry["premium_usd"] == pytest.approx(10.02, abs=1e-6)
        assert entry["saved_usd"] == pytest.approx(9.9723675, abs=1e-6)


def _parse_sse(payload: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Split an SSE payload into (event_name, data) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for chunk in payload.decode("utf-8").strip().split("\n\n"):
        lines = chunk.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append((lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :])))
    return events


class TestSynthesizeSse:
    def _response_body(self) -> dict[str, Any]:
        return {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            ],
            "model": "claude-fable-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 25},
        }

    def test_starts_with_message_start_and_ends_with_message_stop(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        assert events[0][0] == "message_start"
        assert events[-1][0] == "message_stop"

    def test_message_start_carries_id_model_and_input_tokens(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        message = events[0][1]["message"]
        assert message["id"] == "msg_test123"
        assert message["model"] == "claude-fable-5"
        assert message["usage"] == {"input_tokens": 100, "output_tokens": 0}
        assert message["content"] == []
        assert message["stop_reason"] is None

    def test_text_block_streamed_as_text_delta(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        starts = [d for n, d in events if n == "content_block_start"]
        deltas = [d for n, d in events if n == "content_block_delta"]
        assert starts[0]["content_block"] == {"type": "text", "text": ""}
        assert deltas[0]["delta"] == {"type": "text_delta", "text": "hello"}
        assert deltas[0]["index"] == 0

    def test_tool_use_block_streamed_as_input_json_delta(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        starts = [d for n, d in events if n == "content_block_start"]
        deltas = [d for n, d in events if n == "content_block_delta"]
        assert starts[1]["content_block"]["type"] == "tool_use"
        assert starts[1]["content_block"]["id"] == "toolu_1"
        assert starts[1]["content_block"]["name"] == "Bash"
        assert starts[1]["content_block"]["input"] == {}
        assert deltas[1]["delta"]["type"] == "input_json_delta"
        assert json.loads(deltas[1]["delta"]["partial_json"]) == {"command": "ls"}

    def test_every_block_gets_a_content_block_stop(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        stops = [d for n, d in events if n == "content_block_stop"]
        assert [s["index"] for s in stops] == [0, 1]

    def test_message_delta_carries_stop_reason_and_output_tokens(self) -> None:
        events = _parse_sse(synthesis.synthesize_sse(self._response_body()))
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["delta"]["stop_reason"] == "tool_use"
        assert delta["usage"] == {"output_tokens": 25}

    def test_empty_content_still_produces_valid_stream(self) -> None:
        body = self._response_body()
        body["content"] = []
        events = _parse_sse(synthesis.synthesize_sse(body))
        names = [n for n, _ in events]
        assert names == ["message_start", "message_delta", "message_stop"]

    def test_non_dict_blocks_skipped(self) -> None:
        body = self._response_body()
        body["content"] = [None, {"type": "text", "text": "kept"}]
        events = _parse_sse(synthesis.synthesize_sse(body))
        deltas = [d for n, d in events if n == "content_block_delta"]
        assert len(deltas) == 1
        assert deltas[0]["index"] == 1
        assert deltas[0]["delta"]["text"] == "kept"

    def test_missing_usage_defaults_to_zero(self) -> None:
        body = self._response_body()
        del body["usage"]
        events = _parse_sse(synthesis.synthesize_sse(body))
        assert events[0][1]["message"]["usage"]["input_tokens"] == 0
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["usage"] == {"output_tokens": 0}
