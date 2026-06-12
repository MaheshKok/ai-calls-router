"""Spec-derived tests for ai_calls_router.savings.

Contract under test: register_tier_prices teaches LiteLLM per-1M prices from
the tiers: section (never raising); record_routing_savings appends one honest
JSONL entry per routed call (saved_usd = premium_usd - routed_usd) and skips
silently whenever a side is unpriced, premium is missing, or premium equals
routed -- a missing price must never produce a fabricated savings figure
(invariant 5), and accounting must never break a served response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_calls_router import savings
from ai_calls_router.litellm_guard import load_litellm

CHEAP_MODEL = "deepseek/acr-test-cheap"
PREMIUM_MODEL = "deepseek/acr-test-premium"

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


@pytest.fixture(autouse=True)
def _register_test_prices() -> None:
    """Register the deterministic test prices before every test."""
    savings.register_tier_prices(PRICED_ROUTES)


def _read_entries(ledger: Path) -> list[dict[str, Any]]:
    """Parse all JSONL entries from a ledger file."""
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestRegisterTierPrices:
    def test_registered_price_is_usable_by_cost_per_token(self) -> None:
        litellm = load_litellm()
        in_usd, out_usd = litellm.cost_per_token(
            model=CHEAP_MODEL, prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        assert in_usd == pytest.approx(1.0)
        assert out_usd == pytest.approx(2.0)

    def test_tier_without_prices_is_not_registered(self) -> None:
        savings.register_tier_prices(
            {"tiers": {"x": {"model": "deepseek/acr-test-no-price"}}}
        )
        litellm = load_litellm()
        with pytest.raises(Exception):
            litellm.cost_per_token(
                model="deepseek/acr-test-no-price", prompt_tokens=1, completion_tokens=1
            )

    def test_tier_with_partial_prices_is_not_registered(self) -> None:
        # Both input AND output prices are required; one alone is unpriceable.
        savings.register_tier_prices(
            {
                "tiers": {
                    "x": {
                        "model": "deepseek/acr-test-partial-price",
                        "input_cost_per_1m": 1.0,
                    }
                }
            }
        )
        litellm = load_litellm()
        with pytest.raises(Exception):
            litellm.cost_per_token(
                model="deepseek/acr-test-partial-price", prompt_tokens=1, completion_tokens=1
            )

    def test_non_numeric_price_string_is_rejected(self) -> None:
        savings.register_tier_prices(
            {
                "tiers": {
                    "x": {
                        "model": "deepseek/acr-test-str-price",
                        "input_cost_per_1m": "0.28",
                        "output_cost_per_1m": 0.42,
                    }
                }
            }
        )
        litellm = load_litellm()
        with pytest.raises(Exception):
            litellm.cost_per_token(
                model="deepseek/acr-test-str-price", prompt_tokens=1, completion_tokens=1
            )

    def test_bool_prices_are_rejected(self) -> None:
        savings.register_tier_prices(
            {
                "tiers": {
                    "x": {
                        "model": "deepseek/acr-test-bool-price",
                        "input_cost_per_1m": True,
                        "output_cost_per_1m": 1.0,
                    }
                }
            }
        )
        litellm = load_litellm()
        with pytest.raises(Exception):
            litellm.cost_per_token(
                model="deepseek/acr-test-bool-price", prompt_tokens=1, completion_tokens=1
            )

    @pytest.mark.parametrize(
        "routes",
        [
            {},
            {"tiers": None},
            {"tiers": "nope"},
            {"tiers": {"x": 5}},
            {"tiers": {"x": {"model": 42, "input_cost_per_1m": 1, "output_cost_per_1m": 1}}},
        ],
        ids=["empty", "none-tiers", "str-tiers", "int-tier", "non-str-model"],
    )
    def test_malformed_routes_never_raise(self, routes: dict[str, Any]) -> None:
        savings.register_tier_prices(routes)

    def test_re_registration_is_idempotent(self) -> None:
        savings.register_tier_prices(PRICED_ROUTES)
        savings.register_tier_prices(PRICED_ROUTES)
        litellm = load_litellm()
        in_usd, _ = litellm.cost_per_token(
            model=CHEAP_MODEL, prompt_tokens=1_000_000, completion_tokens=0
        )
        assert in_usd == pytest.approx(1.0)


class TestRecordRoutingSavings:
    def test_writes_entry_with_independently_derived_math(self, tmp_path: Path) -> None:
        # 1M input at $1/1M = $1; 500k output at $2/1M = $1 -> routed $2.
        # Same volume at $10/$20 per 1M -> premium $20. Saved = $18.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 1_000_000, 500_000, ledger)
        entries = _read_entries(ledger)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["premium_model"] == PREMIUM_MODEL
        assert entry["routed_model"] == CHEAP_MODEL
        assert entry["input_tokens"] == 1_000_000
        assert entry["output_tokens"] == 500_000
        assert entry["routed_usd"] == pytest.approx(2.0, abs=1e-6)
        assert entry["premium_usd"] == pytest.approx(20.0, abs=1e-6)
        assert entry["saved_usd"] == pytest.approx(18.0, abs=1e-6)
        assert isinstance(entry["ts"], int)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        ledger = tmp_path / "nested" / "dir" / "savings.jsonl"
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 1_000_000, 0, ledger)
        assert len(_read_entries(ledger)) == 1

    def test_appends_one_line_per_call(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 1_000_000, 0, ledger)
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 2_000_000, 0, ledger)
        entries = _read_entries(ledger)
        assert [e["input_tokens"] for e in entries] == [1_000_000, 2_000_000]

    def test_skips_when_premium_model_missing(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(None, CHEAP_MODEL, 1_000_000, 0, ledger)
        savings.record_routing_savings("", CHEAP_MODEL, 1_000_000, 0, ledger)
        assert not ledger.exists()

    def test_skips_when_premium_equals_routed(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(CHEAP_MODEL, CHEAP_MODEL, 1_000_000, 0, ledger)
        assert not ledger.exists()

    def test_skips_when_premium_unpriced(self, tmp_path: Path) -> None:
        # Invariant 5: never fabricate cost numbers.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            "acr-unpriced/premium-xyz", CHEAP_MODEL, 1_000_000, 0, ledger
        )
        assert not ledger.exists()

    def test_skips_when_routed_unpriced(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL, "acr-unpriced/routed-xyz", 1_000_000, 0, ledger
        )
        assert not ledger.exists()

    def test_skips_when_zero_tokens(self, tmp_path: Path) -> None:
        # Zero volume prices to $0 premium -> no honest savings figure.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 0, 0, ledger)
        assert not ledger.exists()

    def test_default_ledger_honors_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger = tmp_path / "env-ledger.jsonl"
        monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(ledger))
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 1_000_000, 0)
        assert len(_read_entries(ledger)) == 1

    def test_unwritable_ledger_never_raises(self, tmp_path: Path) -> None:
        # A directory path cannot be opened for append; must be swallowed.
        savings.record_routing_savings(PREMIUM_MODEL, CHEAP_MODEL, 1_000_000, 0, tmp_path)


class TestRecordSavingsFromResponse:
    def test_extracts_usage_tokens(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        response = {"usage": {"input_tokens": 1_000_000, "output_tokens": 500_000}}
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, response, ledger)
        entries = _read_entries(ledger)
        assert len(entries) == 1
        assert entries[0]["input_tokens"] == 1_000_000
        assert entries[0]["output_tokens"] == 500_000

    def test_missing_usage_writes_nothing(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, {}, ledger)
        assert not ledger.exists()

    def test_non_dict_response_never_raises(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, None, ledger)
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, "junk", ledger)
        assert not ledger.exists()

    def test_null_usage_values_treated_as_zero(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        response = {"usage": {"input_tokens": None, "output_tokens": None}}
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, response, ledger)
        assert not ledger.exists()


class TestLitellmGuard:
    def test_returns_module_with_pricing_api(self) -> None:
        litellm = load_litellm()
        assert hasattr(litellm, "cost_per_token")
        assert hasattr(litellm, "register_model")

    def test_repeated_calls_return_same_object(self) -> None:
        assert load_litellm() is load_litellm()
