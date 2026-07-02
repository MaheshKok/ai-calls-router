"""Tests for Metrics.bootstrap() that restores counters from savings.jsonl."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ai_calls_router.accounting.metrics import (
    _ensure_request_db,
    _load_request_events,
    _Metrics,
)


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
            assert snap["requests"]["total"] == 2  # routed + passthrough + errors (2 + 0 + 0)
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

    def test_sqlite_history_restores_non_ledger_decisions(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        m = _Metrics(db_path=db_path)
        m.record_request(
            method="POST",
            path="/v1/messages",
            status=0,
            tier="premium",
            route="passthrough",
            model="claude-opus",
            user_agent="claude-code/2.0.0",
            client_ip="127.0.0.1",
            tool_names=["Edit"],
            input_tokens=0,
            output_tokens=0,
            cache_read=0,
            cache_creation=0,
            duration=0.0,
            premium_model="claude-opus",
            agent="claude-code-cli",
            session_id="sess-x",
            decision_reason="request_premium_guard",
            request_id="req-x",
        )

        replay = _Metrics(db_path=db_path)
        replay.bootstrap(ledger_path=None, max_recent=10)
        latest = replay.snapshot()["last_requests"][0]

        assert latest["route"] == "passthrough"
        assert latest["model"] == "claude-opus"
        assert latest["tool_names"] == ["Edit"]
        assert latest["decision_reason"] == "request_premium_guard"

    def test_sqlite_history_restores_usage_updates(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        m = _Metrics(db_path=db_path)
        m.record_request(
            method="POST",
            path="/v1/messages",
            status=0,
            tier="premium",
            route="passthrough",
            model="claude-opus",
            user_agent="claude-code/2.0.0",
            client_ip="127.0.0.1",
            tool_names=[],
            input_tokens=0,
            output_tokens=0,
            cache_read=0,
            cache_creation=0,
            duration=0.0,
            request_id="req-y",
        )
        m.update_request_usage(
            request_id="req-y",
            status=200,
            input_tokens=123,
            output_tokens=45,
            cache_read=6,
            cache_creation=7,
            duration=0.25,
        )

        replay = _Metrics(db_path=db_path)
        replay.bootstrap(ledger_path=None, max_recent=10)
        latest = replay.snapshot()["last_requests"][0]

        assert latest["status"] == 200
        assert latest["input_tokens"] == 123
        assert latest["output_tokens"] == 45
        assert latest["cache_read_tokens"] == 6
        assert latest["cache_creation_tokens"] == 7
        assert latest["duration_ms"] == 250

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
    snapshot exposes a compression block using token estimates as the single
    source of truth (chars / DEFAULT_CHARS_PER_TOKEN, truncated to int).
    The ratio is still derived from characters (chars_saved / chars_before)."""

    def test_fresh_metrics_report_zero_compression(self) -> None:
        snap = _Metrics().snapshot()
        comp = snap["compression"]
        assert comp["tokens_before"] == 0
        assert comp["tokens_after"] == 0
        assert comp["tokens_saved"] == 0
        assert comp["ratio"] == 0.0
        # chars_* keys must not leak into snapshot
        assert "chars_before" not in comp
        assert "chars_saved" not in comp

    def test_add_shrink_accumulates_across_turns(self) -> None:
        # Two turns: 10000->4000 and 5000->2000. Totals derived independently:
        # before=15000, after=6000, saved=9000, ratio=9000/15000=0.6,
        # tokens_before=floor(15000/3.5)=4285, tokens_after=floor(6000/3.5)=1714,
        # tokens_saved=floor(9000/3.5)=2571.
        m = _Metrics()
        m.add_shrink(chars_before=10000, chars_after=4000)
        m.add_shrink(chars_before=5000, chars_after=2000)
        comp = m.snapshot()["compression"]
        assert comp["tokens_before"] == 4285
        assert comp["tokens_after"] == 1714
        assert comp["tokens_saved"] == 2571
        assert comp["ratio"] == pytest.approx(0.6)

    def test_add_shrink_clamps_negative_inputs(self) -> None:
        m = _Metrics()
        m.add_shrink(chars_before=-100, chars_after=-40)
        comp = m.snapshot()["compression"]
        assert comp["tokens_before"] == 0
        assert comp["tokens_after"] == 0

    def test_no_op_pass_reports_zero_savings_not_negative(self) -> None:
        # A pass that does not shrink (after == before) contributes no savings.
        m = _Metrics()
        m.add_shrink(chars_before=3000, chars_after=3000)
        comp = m.snapshot()["compression"]
        assert comp["tokens_saved"] == 0
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
            # chars: before=10000, after=4000, saved=6000
            # tokens: floor(10000/3.5)=2857, floor(4000/3.5)=1142, floor(6000/3.5)=1714
            assert comp["tokens_before"] == 2857
            assert comp["tokens_after"] == 1142
            assert comp["tokens_saved"] == 1714
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
            assert comp["tokens_before"] == 0
            assert comp["tokens_after"] == 0
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
            assert comp["tokens_before"] == 0
            assert comp["tokens_after"] == 0
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
        assert snap["compression"]["tokens_before"] == 0


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


class TestRoutedByModelCounter:
    """incr_routed_model maintains a lifetime routed-call tally keyed by routed
    model. The snapshot exposes it under "routed_by_model"; bootstrap seeds it
    from the savings ledger so the tally survives restarts."""

    def test_fresh_metrics_report_empty_model_tally(self) -> None:
        assert _Metrics().snapshot()["routed_by_model"] == {}

    def test_increment_counts_per_model(self) -> None:
        m = _Metrics()
        m.incr_routed_model(model="deepseek-chat")
        m.incr_routed_model(model="deepseek-chat")
        m.incr_routed_model(model="gpt-5.4-mini")
        assert m.snapshot()["routed_by_model"] == {
            "deepseek-chat": 2,
            "gpt-5.4-mini": 1,
        }

    def test_blank_model_buckets_under_unknown(self) -> None:
        m = _Metrics()
        m.incr_routed_model(model="")
        m.incr_routed_model(model="   ")
        assert m.snapshot()["routed_by_model"] == {"unknown": 2}

    def test_bootstrap_seeds_tally_from_ledger(self) -> None:
        # Two records on deepseek-chat, one on gpt-5.4-mini -> {2, 1}. Counts
        # come from the ledger's per-model request totals, not row order.
        records = [
            _make_record(ts=1718000000, routed_model="deepseek-chat", saved_usd=0.0),
            _make_record(ts=1718000001, routed_model="deepseek-chat", saved_usd=0.0),
            _make_record(ts=1718000002, routed_model="gpt-5.4-mini", saved_usd=0.0),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=10)
            assert m.snapshot()["routed_by_model"] == {
                "deepseek-chat": 2,
                "gpt-5.4-mini": 1,
            }
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_bootstrap_then_live_increment_accumulates(self) -> None:
        # A live routed call after bootstrap must add to the seeded count, not
        # replace it: the restart-surviving tally keeps growing in-process.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(_make_record(ts=1718000000, routed_model="deepseek-chat", saved_usd=0.0))
                + "\n"
            )
            ledger_path = Path(f.name)

        try:
            m = _Metrics()
            m.bootstrap(ledger_path=ledger_path, max_recent=10)
            m.incr_routed_model(model="deepseek-chat")
            assert m.snapshot()["routed_by_model"]["deepseek-chat"] == 2
        finally:
            ledger_path.unlink(missing_ok=True)

    def test_snapshot_tally_is_detached_from_live_updates(self) -> None:
        m = _Metrics()
        m.incr_routed_model(model="deepseek-chat")
        snap = m.snapshot()
        m.incr_routed_model(model="deepseek-chat")
        assert snap["routed_by_model"] == {"deepseek-chat": 1}


def _record_passthrough(
    m: _Metrics,
    *,
    route: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    request_id: str,
) -> None:
    """Persist one premium passthrough row to the metrics DB."""
    m.record_request(
        method="POST",
        path="/v1/messages",
        status=200,
        tier="premium",
        route=route,
        model="claude-opus-4-8",
        user_agent="claude-cli/2.1.0",
        client_ip="",
        tool_names=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_creation=cache_creation,
        duration=0,
        premium_model="claude-opus-4-8",
        request_id=request_id,
    )


def _premium_view(snapshot: dict[str, object]) -> dict[str, int]:
    """Extract the premium_tokens block from a snapshot."""
    return dict(snapshot["premium_tokens"])  # type: ignore[arg-type]


class TestPremiumTokenBootstrap:
    """Bootstrap should restore premium token totals from the request DB.

    Premium passthrough turns never reach the savings ledger, so their token
    usage is summed from the persisted request_events rows on restart.
    """

    def test_fresh_metrics_report_zero_premium_tokens(self, tmp_path: Path) -> None:
        m = _Metrics(db_path=tmp_path / "events.db")
        assert _premium_view(m.snapshot()) == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

    def test_bootstrap_sums_passthrough_rows_into_premium_tokens(self, tmp_path: Path) -> None:
        db_path = tmp_path / "events.db"
        writer = _Metrics(db_path=db_path)
        _record_passthrough(
            writer,
            route="passthrough",
            input_tokens=100,
            output_tokens=40,
            cache_read=200,
            cache_creation=10,
            request_id="r1",
        )
        _record_passthrough(
            writer,
            route="passthrough",
            input_tokens=50,
            output_tokens=20,
            cache_read=100,
            cache_creation=5,
            request_id="r2",
        )

        restarted = _Metrics(db_path=db_path)
        restarted.bootstrap(ledger_path=None, max_recent=10)

        assert _premium_view(restarted.snapshot()) == {
            "input": 150,
            "output": 60,
            "cache_read": 300,
            "cache_creation": 15,
        }

    def test_bootstrap_counts_premium_guard_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "events.db"
        writer = _Metrics(db_path=db_path)
        _record_passthrough(
            writer,
            route="premium_guard",
            input_tokens=70,
            output_tokens=30,
            cache_read=0,
            cache_creation=0,
            request_id="g1",
        )

        restarted = _Metrics(db_path=db_path)
        restarted.bootstrap(ledger_path=None, max_recent=10)

        assert _premium_view(restarted.snapshot())["input"] == 70
        assert _premium_view(restarted.snapshot())["output"] == 30

    def test_bootstrap_excludes_routed_rows_from_premium(self, tmp_path: Path) -> None:
        db_path = tmp_path / "events.db"
        writer = _Metrics(db_path=db_path)
        for route in ("litellm", "direct", "anthropic_oauth"):
            _record_passthrough(
                writer,
                route=route,
                input_tokens=999,
                output_tokens=999,
                cache_read=999,
                cache_creation=999,
                request_id=f"routed-{route}",
            )

        restarted = _Metrics(db_path=db_path)
        restarted.bootstrap(ledger_path=None, max_recent=10)

        assert _premium_view(restarted.snapshot()) == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

    def test_bootstrap_restores_premium_even_with_ledger_present(self, tmp_path: Path) -> None:
        db_path = tmp_path / "events.db"
        writer = _Metrics(db_path=db_path)
        _record_passthrough(
            writer,
            route="passthrough",
            input_tokens=80,
            output_tokens=25,
            cache_read=0,
            cache_creation=0,
            request_id="p1",
        )
        ledger_path = _write_records(tmp_path, [_make_record(ts=1)])

        restarted = _Metrics(db_path=db_path)
        restarted.bootstrap(ledger_path=ledger_path, max_recent=10)

        # Premium tokens come from the DB; routed tokens from the ledger. The DB
        # passthrough row must still be counted when a ledger is present.
        premium = _premium_view(restarted.snapshot())
        assert premium["input"] == 80
        assert premium["output"] == 25

    def test_premium_tokens_survive_record_then_update_then_restart(self, tmp_path: Path) -> None:
        db_path = tmp_path / "events.db"
        writer = _Metrics(db_path=db_path)
        # Real flow: row persisted with zero usage, filled by the async callback.
        _record_passthrough(
            writer,
            route="passthrough",
            input_tokens=0,
            output_tokens=0,
            cache_read=0,
            cache_creation=0,
            request_id="late",
        )
        writer.update_request_usage(
            request_id="late",
            status=200,
            input_tokens=120,
            output_tokens=45,
            cache_read=300,
            cache_creation=12,
            duration=1.0,
        )

        restarted = _Metrics(db_path=db_path)
        restarted.bootstrap(ledger_path=None, max_recent=10)

        assert _premium_view(restarted.snapshot()) == {
            "input": 120,
            "output": 45,
            "cache_read": 300,
            "cache_creation": 12,
        }

    def test_bootstrap_premium_noop_without_db_path(self) -> None:
        m = _Metrics()
        m.bootstrap(ledger_path=None, max_recent=10)
        assert _premium_view(m.snapshot()) == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }


class TestRequestsTotal:
    """requests.total must equal routed + passthrough + errors."""

    def test_total_is_zero_on_fresh_metrics(self) -> None:
        m = _Metrics()
        assert m.snapshot()["requests"]["total"] == 0

    def test_total_includes_routed_only(self) -> None:
        m = _Metrics()
        m.incr_routed()
        m.incr_routed()
        snap = m.snapshot()
        assert snap["requests"]["total"] == 2
        assert snap["requests"]["routed"] == 2
        assert snap["requests"]["errors"] == 0

    def test_total_includes_passthrough_only(self) -> None:
        m = _Metrics()
        m.incr_passthrough()
        snap = m.snapshot()
        assert snap["requests"]["total"] == 1
        assert snap["requests"]["passthrough"] == 1

    def test_total_includes_errors(self) -> None:
        m = _Metrics()
        m.incr_error()
        snap = m.snapshot()
        assert snap["requests"]["total"] == 1
        assert snap["requests"]["errors"] == 1
        assert snap["requests"]["routed"] == 0
        assert snap["requests"]["passthrough"] == 0

    def test_total_sums_all_three_categories(self) -> None:
        m = _Metrics()
        m.incr_routed()
        m.incr_routed()
        m.incr_passthrough()
        m.incr_error()
        snap = m.snapshot()
        assert snap["requests"]["total"] == 4
        assert snap["requests"]["routed"] == 2
        assert snap["requests"]["passthrough"] == 1
        assert snap["requests"]["errors"] == 1


# ── tool_output_type: migration + persistence ───────────────────────────────

# The request_events schema exactly as it shipped BEFORE tool_output_type was
# added (23 data columns). Written by hand rather than derived from the current
# constant so the test pins the real legacy shape a live metrics.db would have.
_LEGACY_REQUEST_EVENTS_DDL = """
CREATE TABLE request_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status INTEGER NOT NULL,
    tier TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    premium_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    agent TEXT NOT NULL,
    session_id TEXT NOT NULL,
    decision_reason TEXT NOT NULL,
    client_ip TEXT NOT NULL,
    tool_names TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    shrink_chars_before INTEGER NOT NULL,
    shrink_chars_after INTEGER NOT NULL
)
"""

_LEGACY_INSERT = """
INSERT INTO request_events (
    ts, request_id, method, path, status, tier, route, model, premium_model,
    provider, user_agent, agent, session_id, decision_reason, client_ip,
    tool_names, input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, duration_ms, shrink_chars_before, shrink_chars_after
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _seed_legacy_db(db_path: Path) -> None:
    """Create a pre-migration request_events table with one row, no column."""
    with sqlite3.connect(db_path) as db:
        db.execute(_LEGACY_REQUEST_EVENTS_DDL)
        db.execute(
            _LEGACY_INSERT,
            (
                1718000000,
                "legacy-req",
                "POST",
                "/v1/messages",
                200,
                "deepseek",
                "routed",
                "deepseek-chat",
                "claude-opus",
                "deepseek",
                "claude-code/2.0.0",
                "claude-code-cli",
                "sess-legacy",
                "routed",
                "127.0.0.1",
                "[]",
                100,
                40,
                0,
                0,
                0,
                0,
                0,
            ),
        )


def _table_columns(db_path: Path) -> list[str]:
    """Return current request_events column names in declaration order."""
    with sqlite3.connect(db_path) as db:
        return [row[1] for row in db.execute("PRAGMA table_info(request_events)")]


class TestToolOutputTypeMigration:
    """A DB predating tool_output_type must gain the column without data loss."""

    def test_legacy_db_lacks_column_before_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        _seed_legacy_db(db_path)
        assert "tool_output_type" not in _table_columns(db_path)

    def test_ensure_request_db_adds_missing_column(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        _seed_legacy_db(db_path)
        _ensure_request_db(db_path)
        assert "tool_output_type" in _table_columns(db_path)

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        # A second connect must not raise "duplicate column name".
        db_path = tmp_path / "metrics.db"
        _seed_legacy_db(db_path)
        _ensure_request_db(db_path)
        _ensure_request_db(db_path)
        assert _table_columns(db_path).count("tool_output_type") == 1

    def test_legacy_row_survives_and_reads_back_blank(self, tmp_path: Path) -> None:
        # The pre-existing row must remain and default to an empty label.
        db_path = tmp_path / "metrics.db"
        _seed_legacy_db(db_path)
        rows = _load_request_events(db_path, max_recent=10)
        assert len(rows) == 1
        assert rows[0]["request_id"] == "legacy-req"
        assert rows[0]["tool_output_type"] == ""


class TestToolOutputTypePersistence:
    """record_request must persist and reload the classification label."""

    def _record(self, m: _Metrics, *, request_id: str, **kwargs: object) -> None:
        m.record_request(
            method="POST",
            path="/v1/messages",
            status=200,
            tier="deepseek",
            route="routed",
            model="deepseek-chat",
            user_agent="claude-code/2.0.0",
            client_ip="127.0.0.1",
            tool_names=[],
            input_tokens=10,
            output_tokens=5,
            cache_read=0,
            cache_creation=0,
            duration=0.0,
            request_id=request_id,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_label_visible_in_live_snapshot(self, tmp_path: Path) -> None:
        m = _Metrics(db_path=tmp_path / "metrics.db")
        self._record(m, request_id="req-live", tool_output_type="excluded, smart_crusher")
        assert m.snapshot()["last_requests"][0]["tool_output_type"] == "excluded, smart_crusher"

    def test_label_survives_restart(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        m = _Metrics(db_path=db_path)
        self._record(m, request_id="req-restart", tool_output_type="smart_crusher")

        replay = _Metrics(db_path=db_path)
        replay.bootstrap(ledger_path=None, max_recent=10)
        assert replay.snapshot()["last_requests"][0]["tool_output_type"] == "smart_crusher"

    def test_defaults_to_blank_when_omitted(self, tmp_path: Path) -> None:
        db_path = tmp_path / "metrics.db"
        m = _Metrics(db_path=db_path)
        self._record(m, request_id="req-blank")

        replay = _Metrics(db_path=db_path)
        replay.bootstrap(ledger_path=None, max_recent=10)
        assert replay.snapshot()["last_requests"][0]["tool_output_type"] == ""
