"""Spec-derived tests for ai_calls_router.ledger.

Contract under test: load_entries reads a JSONL savings ledger tolerantly
(missing file, blank lines, corrupt lines, and non-dict JSON all skipped);
aggregate rolls entries up into totals plus a per-routed-model breakdown
with a zero-division-safe savings percentage; format_report renders a
human-readable summary for the acr savings command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_calls_router.accounting import ledger


def _entry(**overrides: Any) -> dict[str, Any]:
    """Build a complete ledger entry with overridable fields."""
    base: dict[str, Any] = {
        "ts": 1_770_000_000,
        "premium_model": "claude-fable-5",
        "routed_model": "deepseek/deepseek-v4-flash",
        "input_tokens": 1000,
        "output_tokens": 200,
        "routed_usd": 0.001,
        "premium_usd": 0.02,
        "saved_usd": 0.019,
    }
    base.update(overrides)
    return base


class TestLoadEntries:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert ledger.load_entries(tmp_path / "absent.jsonl") == []

    def test_parses_one_dict_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert ledger.load_entries(path) == [{"a": 1}, {"b": 2}]

    def test_blank_and_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        path.write_text('{"a": 1}\n\n   \nnot json{\n{"b": 2}\n', encoding="utf-8")
        assert ledger.load_entries(path) == [{"a": 1}, {"b": 2}]

    def test_non_dict_json_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "ledger.jsonl"
        path.write_text('[1, 2]\n"string"\n42\nnull\n{"keep": true}\n', encoding="utf-8")
        assert ledger.load_entries(path) == [{"keep": True}]

    def test_default_path_honors_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "env-ledger.jsonl"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(path))
        assert ledger.load_entries() == [{"a": 1}]


class TestAggregate:
    def test_empty_entries_aggregate_to_zeros(self) -> None:
        summary = ledger.aggregate([])
        assert summary["totals"]["requests"] == 0
        assert summary["totals"]["saved_usd"] == 0.0
        assert summary["totals"]["savings_pct"] == 0.0
        assert summary["by_model"] == {}

    def test_single_entry_totals(self) -> None:
        summary = ledger.aggregate(
            [
                _entry(
                    input_tokens=1000,
                    output_tokens=200,
                    routed_usd=0.5,
                    premium_usd=5.0,
                    saved_usd=4.5,
                )
            ]
        )
        totals = summary["totals"]
        assert totals["requests"] == 1
        assert totals["input_tokens"] == 1000
        assert totals["output_tokens"] == 200
        assert totals["routed_usd"] == pytest.approx(0.5)
        assert totals["premium_usd"] == pytest.approx(5.0)
        assert totals["saved_usd"] == pytest.approx(4.5)
        # 4.5 / 5.0 * 100 = 90.0, rounded to one decimal.
        assert totals["savings_pct"] == 90.0

    def test_multiple_models_bucketed_separately(self) -> None:
        entries = [
            _entry(routed_model="deepseek/a", saved_usd=1.0, premium_usd=2.0, routed_usd=1.0),
            _entry(routed_model="deepseek/a", saved_usd=1.0, premium_usd=2.0, routed_usd=1.0),
            _entry(routed_model="groq/b", saved_usd=3.0, premium_usd=4.0, routed_usd=1.0),
        ]
        summary = ledger.aggregate(entries)
        assert summary["totals"]["requests"] == 3
        assert summary["by_model"]["deepseek/a"]["requests"] == 2
        assert summary["by_model"]["deepseek/a"]["saved_usd"] == pytest.approx(2.0)
        assert summary["by_model"]["groq/b"]["requests"] == 1
        assert summary["by_model"]["groq/b"]["saved_usd"] == pytest.approx(3.0)

    def test_missing_routed_model_buckets_under_question_mark(self) -> None:
        entry = _entry()
        del entry["routed_model"]
        summary = ledger.aggregate([entry])
        assert "?" in summary["by_model"]

    def test_missing_and_null_fields_default_to_zero(self) -> None:
        summary = ledger.aggregate(
            [{"routed_model": "deepseek/a", "input_tokens": None, "saved_usd": None}]
        )
        totals = summary["totals"]
        assert totals["requests"] == 1
        assert totals["input_tokens"] == 0
        assert totals["saved_usd"] == 0.0
        assert totals["savings_pct"] == 0.0

    def test_savings_pct_rounded_to_one_decimal(self) -> None:
        summary = ledger.aggregate(
            [_entry(saved_usd=1.0, premium_usd=3.0, routed_usd=2.0)]
        )
        # 1/3 * 100 = 33.333... -> 33.3
        assert summary["totals"]["savings_pct"] == 33.3

    def test_zero_premium_yields_zero_pct(self) -> None:
        summary = ledger.aggregate([_entry(saved_usd=0.0, premium_usd=0.0, routed_usd=0.0)])
        assert summary["totals"]["savings_pct"] == 0.0


class TestFormatReport:
    def test_empty_ledger_message(self) -> None:
        report = ledger.format_report(ledger.aggregate([]))
        assert "No routed calls recorded" in report

    def test_report_contains_totals_and_models(self) -> None:
        entries = [
            _entry(routed_model="deepseek/a", saved_usd=1.5, premium_usd=2.0, routed_usd=0.5),
            _entry(routed_model="groq/b", saved_usd=3.0, premium_usd=4.0, routed_usd=1.0),
        ]
        report = ledger.format_report(ledger.aggregate(entries))
        assert "2" in report  # request count
        assert "deepseek/a" in report
        assert "groq/b" in report
        assert "4.50" in report  # total saved

    def test_report_is_provider_neutral(self) -> None:
        # The standalone proxy routes to any LiteLLM provider; the report
        # must not hardcode a provider name outside the per-model lines.
        report = ledger.format_report(
            ledger.aggregate([_entry(routed_model="groq/kimi-k2")])
        )
        assert "DeepSeek" not in report
