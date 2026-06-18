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


def _write_records(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    """Write records to a temporary savings ledger."""
    ledger_path = tmp_path / "savings.jsonl"
    ledger_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    return ledger_path


def _apply_live_record(m: _Metrics, record: dict[str, object]) -> None:
    """Apply one ledger-equivalent record through live metrics methods."""
    input_tokens = int(record["input_tokens"])
    output_tokens = int(record["output_tokens"])
    cache_read = int(record["cache_read_input_tokens"])
    cache_creation = int(record["cache_creation_input_tokens"])
    m.incr_routed()
    m.add_routed_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_creation=cache_creation,
    )
    m.add_savings(
        routed_usd=float(record["routed_usd"]),
        premium_usd=float(record["premium_usd"]),
        saved_usd=float(record["saved_usd"]),
    )
    m.add_shrink(
        chars_before=int(record.get("shrink_chars_before", 0)),
        chars_after=int(record.get("shrink_chars_after", 0)),
    )
    m.record_request(
        method="POST",
        path="/v1/messages",
        status=200,
        tier=str(record["tier_name"]),
        route="routed",
        model=str(record["routed_model"]),
        premium_model=str(record["premium_model"]),
        user_agent=str(record["user_agent"]),
        client_ip="",
        tool_names=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_creation=cache_creation,
        duration=0,
        agent=str(record["agent"]),
        session_id=str(record["session_id"]),
        provider=str(record["provider"]),
        decision_reason="routed",
        shrink_chars_before=int(record.get("shrink_chars_before", 0)),
        shrink_chars_after=int(record.get("shrink_chars_after", 0)),
    )


def _aggregate_view(snapshot: dict[str, object]) -> dict[str, object]:
    """Extract aggregate fields that ledger replay must reproduce."""
    return {
        "requests": snapshot["requests"],
        "routed_tokens": snapshot["routed_tokens"],
        "costs": snapshot["costs"],
        "compression": snapshot["compression"],
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

    def test_ledger_replay_equals_live_aggregate(self, tmp_path: Path) -> None:
        records = [
            _make_record(
                ts=1718000000,
                input_tokens=100,
                output_tokens=40,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=2,
                routed_usd=0.001,
                premium_usd=0.008,
                saved_usd=0.007,
            ),
            _make_record(
                ts=1718000001,
                input_tokens=300,
                output_tokens=80,
                cache_read_input_tokens=20,
                cache_creation_input_tokens=5,
                routed_usd=0.003,
                premium_usd=0.020,
                saved_usd=0.017,
            ),
        ]
        records[0]["shrink_chars_before"] = 8000
        records[0]["shrink_chars_after"] = 3000
        records[1]["shrink_chars_before"] = -9
        records[1]["shrink_chars_after"] = -3
        live = _Metrics()
        for record in records:
            _apply_live_record(live, record)

        replay = _Metrics()
        replay.bootstrap(ledger_path=_write_records(tmp_path, records), max_recent=10)

        assert _aggregate_view(replay.snapshot()) == _aggregate_view(live.snapshot())


class TestMetricsBootstrapIntegration:
    """Check that total = routed + passthrough after bootstrap."""

    def test_total_is_routed_plus_passthrough(self) -> None:
        m = _Metrics()
        m._routed_requests = 5
        m._passthrough_requests = 3
        snap = m.snapshot()
        assert snap["requests"]["total"] == 8


class TestMetricsCompression:
    """add_shrink accumulates per-turn tool_result character deltas, and the
    snapshot exposes a compression block with the derived savings figures. The
    character counts are authoritative; est_tokens_saved is a coarse divisor
    estimate (chars_saved / 3.5, truncated)."""

    def test_fresh_metrics_report_zero_compression(self) -> None:
        snap = _Metrics().snapshot()
        comp = snap["compression"]
        assert comp["chars_before"] == 0
        assert comp["chars_after"] == 0
        assert comp["chars_saved"] == 0
        assert comp["ratio"] == 0.0
        assert comp["est_tokens_saved"] == 0

    def test_add_shrink_accumulates_across_turns(self) -> None:
        # Two turns: 10000->4000 and 5000->2000. Totals derived independently:
        # before=15000, after=6000, saved=9000, ratio=9000/15000=0.6,
        # est=floor(9000/3.5)=2571.
        m = _Metrics()
        m.add_shrink(chars_before=10000, chars_after=4000)
        m.add_shrink(chars_before=5000, chars_after=2000)
        comp = m.snapshot()["compression"]
        assert comp["chars_before"] == 15000
        assert comp["chars_after"] == 6000
        assert comp["chars_saved"] == 9000
        assert comp["ratio"] == pytest.approx(0.6)
        assert comp["est_tokens_saved"] == 2571

    def test_add_shrink_clamps_negative_inputs(self) -> None:
        m = _Metrics()
        m.add_shrink(chars_before=-100, chars_after=-40)
        comp = m.snapshot()["compression"]
        assert comp["chars_before"] == 0
        assert comp["chars_after"] == 0

    def test_no_op_pass_reports_zero_savings_not_negative(self) -> None:
        # A pass that does not shrink (after == before) contributes no savings,
        # and chars_saved is floored at 0 even if a later turn grows the body.
        m = _Metrics()
        m.add_shrink(chars_before=3000, chars_after=3000)
        comp = m.snapshot()["compression"]
        assert comp["chars_saved"] == 0
        assert comp["ratio"] == 0.0

    def test_bootstrap_restores_shrink_chars_from_ledger(self) -> None:
        rec1 = _make_record(ts=1718000000, saved_usd=0.0)
        rec1["shrink_chars_before"] = 8000
        rec1["shrink_chars_after"] = 3000
        rec2 = _make_record(ts=1718000001, saved_usd=0.0)
        rec2["shrink_chars_before"] = 2000
        rec2["shrink_chars_after"] = 1000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec1) + "\n")
            f.write(json.dumps(rec2) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=2)
            comp = m.snapshot()["compression"]
            assert comp["chars_before"] == 10000
            assert comp["chars_after"] == 4000
            assert comp["chars_saved"] == 6000
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_missing_shrink_fields_default_to_zero(self) -> None:
        # Legacy ledger entries predate the shrink fields; replay must treat
        # their absence as zero rather than raising.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_record(ts=1718000000, saved_usd=0.0)) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=1)
            comp = m.snapshot()["compression"]
            assert comp["chars_before"] == 0
            assert comp["chars_after"] == 0
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_clamps_negative_shrink_fields(self) -> None:
        rec = _make_record(ts=1718000000, saved_usd=0.0)
        rec["shrink_chars_before"] = -500
        rec["shrink_chars_after"] = -9

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=1)
            comp = m.snapshot()["compression"]
            assert comp["chars_before"] == 0
            assert comp["chars_after"] == 0
        finally:
            ledger_path.unlink(missing_ok=True)


def _record_routed(
    m: _Metrics,
    *,
    shrink_chars_before: int = 0,
    shrink_chars_after: int = 0,
) -> None:
    """Record one routed request row with the given shrink measurement.

    Args:
        m: The metrics instance to record into.
        shrink_chars_before: tool_result characters before the shrink pass.
        shrink_chars_after: tool_result characters after the shrink pass.
    """
    m.record_request(
        method="POST",
        path="/v1/messages",
        status=200,
        tier="deepseek",
        route="direct",
        model="deepseek-chat",
        user_agent="claude-code/2.0.0",
        client_ip="",
        tool_names=[],
        input_tokens=500,
        output_tokens=400,
        cache_read=100,
        cache_creation=50,
        duration=0.5,
        shrink_chars_before=shrink_chars_before,
        shrink_chars_after=shrink_chars_after,
    )


class TestRecordRequestPerRowShrink:
    """A live recorded request carries its own tool_result shrink measurement
    on the recent-requests row, so the dashboard can show per-row shrinkage
    rather than only the cumulative total."""

    def test_recorded_row_carries_explicit_shrink_fields(self) -> None:
        m = _Metrics()
        _record_routed(m, shrink_chars_before=12000, shrink_chars_after=4500)
        row = m.snapshot()["last_requests"][0]
        assert row["shrink_chars_before"] == 12000
        assert row["shrink_chars_after"] == 4500

    def test_recorded_row_defaults_shrink_to_zero(self) -> None:
        # A path that does not pass shrink must produce neutral zeros, not a
        # missing key the dashboard would read as undefined.
        m = _Metrics()
        _record_routed(m)
        row = m.snapshot()["last_requests"][0]
        assert row["shrink_chars_before"] == 0
        assert row["shrink_chars_after"] == 0

    def test_recorded_row_clamps_negative_shrink_to_zero(self) -> None:
        m = _Metrics()
        _record_routed(m, shrink_chars_before=-7, shrink_chars_after=-3)
        row = m.snapshot()["last_requests"][0]
        assert row["shrink_chars_before"] == 0
        assert row["shrink_chars_after"] == 0

    def test_no_op_pass_records_equal_before_and_after(self) -> None:
        # When the shrink pass removed nothing (before == after) the row keeps
        # both counts verbatim, so a per-row "0 saved" stays distinguishable
        # from "no measurement". 77892 mirrors the observed real-traffic no-op.
        m = _Metrics()
        _record_routed(m, shrink_chars_before=77892, shrink_chars_after=77892)
        row = m.snapshot()["last_requests"][0]
        assert row["shrink_chars_before"] == 77892
        assert row["shrink_chars_after"] == 77892

    def test_snapshot_rows_are_detached_from_live_updates(self) -> None:
        m = _Metrics()
        m.record_request(
            method="POST",
            path="/v1/messages",
            status=0,
            tier="premium",
            route="passthrough",
            model="claude",
            user_agent="",
            client_ip="",
            tool_names=[],
            input_tokens=0,
            output_tokens=0,
            cache_read=0,
            cache_creation=0,
            duration=0,
            request_id="req-1",
        )
        row = m.snapshot()["last_requests"][0]
        m.update_request_usage(
            request_id="req-1",
            status=200,
            input_tokens=7,
            output_tokens=3,
            cache_read=2,
            cache_creation=1,
            duration=0.5,
        )
        assert row["status"] == 0
        assert row["input_tokens"] == 0

    def test_snapshot_aggregate_is_detached_from_live_updates(self) -> None:
        m = _Metrics()
        m.incr_routed()
        snap = m.snapshot()
        m.incr_routed()
        m.add_routed_tokens(input_tokens=7, output_tokens=3, cache_read=2, cache_creation=1)
        m.add_savings(routed_usd=0.01, premium_usd=0.05, saved_usd=0.04)
        m.add_shrink(chars_before=100, chars_after=25)
        assert snap["requests"]["routed"] == 1
        assert snap["routed_tokens"]["input"] == 0
        assert snap["costs"]["saved_usd"] == 0
        assert snap["compression"]["chars_before"] == 0


class TestBootstrapPerRowShrink:
    """Bootstrap rebuilds recent-request rows from the ledger, and each row
    carries the shrink measurement persisted with that record."""

    def test_bootstrap_row_carries_shrink_fields(self) -> None:
        rec = _make_record(ts=1718000000, saved_usd=0.0)
        rec["shrink_chars_before"] = 8000
        rec["shrink_chars_after"] = 3000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=1)
            row = m.snapshot()["last_requests"][0]
            assert row["shrink_chars_before"] == 8000
            assert row["shrink_chars_after"] == 3000
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_row_missing_shrink_defaults_to_zero(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_record(ts=1718000000, saved_usd=0.0)) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=1)
            row = m.snapshot()["last_requests"][0]
            assert row["shrink_chars_before"] == 0
            assert row["shrink_chars_after"] == 0
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_row_clamps_negative_shrink_to_zero(self) -> None:
        rec = _make_record(ts=1718000000, saved_usd=0.0)
        rec["shrink_chars_before"] = -500
        rec["shrink_chars_after"] = -9

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(rec) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=1)
            row = m.snapshot()["last_requests"][0]
            assert row["shrink_chars_before"] == 0
            assert row["shrink_chars_after"] == 0
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_row_shape_matches_record_request(self, tmp_path: Path) -> None:
        rec = _make_record(ts=1718000000, saved_usd=0.0)
        live = _Metrics()
        _apply_live_record(live, rec)

        replay = _Metrics()
        replay.bootstrap(ledger_path=_write_records(tmp_path, [rec]), max_recent=1)

        live_row = live.snapshot()["last_requests"][0]
        replay_row = replay.snapshot()["last_requests"][0]
        assert set(replay_row) == set(live_row)
