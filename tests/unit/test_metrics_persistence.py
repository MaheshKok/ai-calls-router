"""Tests for Metrics.bootstrap() that restores counters from savings.jsonl."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ai_calls_router.accounting.metrics import _Metrics


def _make_record(
    *,
    ts: int = 0,
    premium_model: str = "claude-sonnet-4-20250514",
    routed_model: str = "deepseek-chat",
    input_tokens: int = 500,
    output_tokens: int = 400,
    cache_read_input_tokens: int = 100,
    cache_creation_input_tokens: int = 50,
    routed_usd: float = 0.001,
    premium_usd: float = 0.005,
    saved_usd: float = 0.004,
    tier_name: str = "deepseek",
    tool_names: str = "",
    user_agent: str = "claude-code/2.0.0",
    agent: str = "claude-code-cli",
    session_id: str = "abc123def456",
    provider: str = "deepseek",
) -> dict[str, object]:
    return {
        "ts": ts,
        "premium_model": premium_model,
        "routed_model": routed_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "routed_usd": routed_usd,
        "premium_usd": premium_usd,
        "saved_usd": saved_usd,
        "tier_name": tier_name,
        "tool_names": tool_names,
        "user_agent": user_agent,
        "agent": agent,
        "session_id": session_id,
        "provider": provider,
    }


class TestMetricsBootstrapCounters:
    """Bootstrap should restore counters from a savings.jsonl file."""

    def test_bootstrap_restores_routed_tokens(self) -> None:
        rec1 = _make_record(
            ts=1718000000,
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            saved_usd=0.0,
        )
        rec2 = _make_record(
            ts=1718000001,
            input_tokens=200,
            output_tokens=80,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=10,
            saved_usd=0.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec1) + "\n")
            f.write(json.dumps(rec2) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=2)
            snap = m.snapshot()

            assert snap["requests"]["routed"] == 2
            assert snap["requests"]["total"] == 2  # total = routed + passthrough (2 + 0)
            assert snap["routed_tokens"]["input"] == 300
            assert snap["routed_tokens"]["output"] == 130
            assert snap["routed_tokens"]["cache_read"] == 30
            assert snap["routed_tokens"]["cache_creation"] == 10
            assert snap["costs"]["routed_usd"] == pytest.approx(0.002)
            assert snap["costs"]["premium_usd"] == pytest.approx(0.010)
            assert snap["costs"]["saved_usd"] == pytest.approx(0.000)
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_preserves_last_requests(self) -> None:
        rec1 = _make_record(
            ts=1718000000,
            session_id="sess-a",
            agent="claude-code-cli",
            provider="deepseek",
        )
        rec2 = _make_record(
            ts=1718000001,
            session_id="sess-b",
            agent="claude-desktop",
            provider="anthropic",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec1) + "\n")
            f.write(json.dumps(rec2) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=2)
            snap = m.snapshot()
            last = snap["last_requests"]
            assert len(last) == 2
            # Most recent first
            assert last[0]["agent"] == "claude-desktop"
            assert last[0]["session_id"] == "sess-b"
            assert last[0]["provider"] == "anthropic"
            assert last[1]["agent"] == "claude-code-cli"
            assert last[1]["session_id"] == "sess-a"
            assert last[1]["provider"] == "deepseek"
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_empty_ledger(self) -> None:
        m = _Metrics()
        m.bootstrap(ledger_path=None, max_recent=10)
        snap = m.snapshot()
        assert snap["requests"]["routed"] == 0
        assert snap["last_requests"] == []

    def test_bootstrap_ignores_non_savings_lines(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"not": "a savings entry"}\n')
            f.write('{"routed_model": "deepseek-chat", "tier_name": "deepseek"}\n')
            f.write(json.dumps(_make_record(ts=1718000000, saved_usd=0.0)) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=10)
            snap = m.snapshot()
            # Only the well-formed entry with premium_model counts
            assert snap["requests"]["routed"] == 1
        finally:
            ledger_path.unlink(missing_ok=True)


class TestMetricsBootstrapIntegration:
    """Check that total = routed + passthrough after bootstrap."""

    def test_total_is_routed_plus_passthrough(self) -> None:
        m = _Metrics()
        m._routed_requests = 5
        m._passthrough_requests = 3
        snap = m.snapshot()
        assert snap["requests"]["total"] == 8
