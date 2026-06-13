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

from ai_calls_router._lib.litellm_guard import load_litellm
from ai_calls_router.accounting import savings

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
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
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
        savings.register_tier_prices({"tiers": {"x": {"model": "deepseek/acr-test-no-price"}}})
        litellm = load_litellm()
        with pytest.raises(Exception, match="isn't mapped yet"):
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
        with pytest.raises(Exception, match="isn't mapped yet"):
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
        with pytest.raises(Exception, match="isn't mapped yet"):
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
        with pytest.raises(Exception, match="isn't mapped yet"):
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


class TestRoutedPricesFromTier:
    """_routed_prices_from_tier derives (miss, cached, output) per-token rates
    from a tier config, falling back to the miss rate when no cache-read price
    is declared, and returning None whenever the tier is unpriceable."""

    @pytest.mark.parametrize(
        "tier_cfg",
        [None, "deepseek/x", 5, 1.5, ["model"], {"model": "deepseek/x"}],
        ids=["none", "str", "int", "float", "list", "no-prices"],
    )
    def test_returns_none_for_unpriceable_tier(self, tier_cfg: Any) -> None:
        assert savings._routed_prices_from_tier(tier_cfg) is None

    def test_returns_none_when_output_price_missing(self) -> None:
        assert savings._routed_prices_from_tier({"input_cost_per_1m": 0.5}) is None

    @pytest.mark.parametrize("bad", ["0.5", True, None], ids=["str", "bool", "none"])
    def test_returns_none_when_input_price_not_a_number(self, bad: Any) -> None:
        tier = {"input_cost_per_1m": bad, "output_cost_per_1m": 1.0}
        assert savings._routed_prices_from_tier(tier) is None

    def test_full_tier_returns_three_distinct_per_token_rates(self) -> None:
        tier = {
            "input_cost_per_1m": 0.435,
            "input_cached_cost_per_1m": 0.003625,
            "output_cost_per_1m": 0.87,
        }
        miss, cached, out = savings._routed_prices_from_tier(tier)
        assert miss == pytest.approx(0.435 / 1_000_000)
        assert cached == pytest.approx(0.003625 / 1_000_000)
        assert out == pytest.approx(0.87 / 1_000_000)

    def test_cached_rate_falls_back_to_miss_rate_when_absent(self) -> None:
        # No cache-read price declared: hits priced at the miss rate, a
        # conservative upper bound that never overstates savings.
        miss, cached, _ = savings._routed_prices_from_tier(
            {"input_cost_per_1m": 0.5, "output_cost_per_1m": 1.0}
        )
        assert cached == miss == pytest.approx(0.5 / 1_000_000)

    @pytest.mark.parametrize("bad", ["x", True, None], ids=["str", "bool", "none"])
    def test_cached_rate_falls_back_when_not_a_number(self, bad: Any) -> None:
        miss, cached, _ = savings._routed_prices_from_tier(
            {
                "input_cost_per_1m": 0.5,
                "input_cached_cost_per_1m": bad,
                "output_cost_per_1m": 1.0,
            }
        )
        assert cached == miss == pytest.approx(0.5 / 1_000_000)


# DeepSeek-pro per-token rates: a cache hit costs 1/120th of a miss.
DS_ROUTED_PRICES = (0.435 / 1_000_000, 0.003625 / 1_000_000, 0.87 / 1_000_000)


class TestCacheAwareSavings:
    """record_routing_savings reconciles cache tokens against the routed and
    premium sides: hits bill at the cheap cached rate, misses (reported input
    plus cache_creation) at the miss rate, and the premium counterfactual
    always prices the entire prompt (hit + miss) at the premium standard rate."""

    def test_splits_hit_miss_output_at_independent_rates(self, tmp_path: Path) -> None:
        # hit=700k @ 0.003625/1M, miss=200k+100k=300k @ 0.435/1M, out=50k @ 0.87/1M
        #   routed  = 0.1305 + 0.0025375 + 0.0435           = 0.1765375
        # premium prices the full 1M prompt + 50k out at 10/20 per 1M
        #   premium = 10.0 + 1.0                            = 11.0
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            200_000,
            50_000,
            ledger,
            cache_read_tokens=700_000,
            cache_creation_tokens=100_000,
            routed_prices=DS_ROUTED_PRICES,
        )
        entry = _read_entries(ledger)[0]
        assert entry["input_tokens"] == 1_000_000
        assert entry["cache_read_input_tokens"] == 700_000
        assert entry["cache_creation_input_tokens"] == 100_000
        assert entry["routed_usd"] == pytest.approx(0.1765375, abs=1e-7)
        assert entry["premium_usd"] == pytest.approx(11.0, abs=1e-6)
        assert entry["saved_usd"] == pytest.approx(10.8234625, abs=1e-6)

    def test_all_hit_is_far_cheaper_than_all_miss(self, tmp_path: Path) -> None:
        # The same 1M prompt priced as all-cache-hit vs all-miss must differ by
        # the hit/miss ratio -- catching a swapped hit/miss rate.
        hit_ledger = tmp_path / "hit.jsonl"
        miss_ledger = tmp_path / "miss.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            0,
            0,
            hit_ledger,
            cache_read_tokens=1_000_000,
            routed_prices=DS_ROUTED_PRICES,
        )
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            1_000_000,
            0,
            miss_ledger,
            routed_prices=DS_ROUTED_PRICES,
        )
        hit_usd = _read_entries(hit_ledger)[0]["routed_usd"]
        miss_usd = _read_entries(miss_ledger)[0]["routed_usd"]
        assert hit_usd == pytest.approx(0.003625, abs=1e-7)
        assert miss_usd == pytest.approx(0.435, abs=1e-7)

    def test_cache_creation_bills_at_miss_rate_not_cached_rate(self, tmp_path: Path) -> None:
        # cache_creation tokens are written-not-read, so they cost a full miss.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            0,
            0,
            ledger,
            cache_creation_tokens=1_000_000,
            routed_prices=DS_ROUTED_PRICES,
        )
        entry = _read_entries(ledger)[0]
        assert entry["cache_creation_input_tokens"] == 1_000_000
        assert entry["cache_read_input_tokens"] == 0
        assert entry["routed_usd"] == pytest.approx(0.435, abs=1e-7)

    def test_premium_counterfactual_prices_full_prompt_including_hits(self, tmp_path: Path) -> None:
        # Premium has no prefix cache: it would re-read all 1M tokens. The
        # counterfactual must price hit+miss, not misses alone.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            0,
            0,
            ledger,
            cache_read_tokens=1_000_000,
            routed_prices=DS_ROUTED_PRICES,
        )
        # 1M @ $10/1M = $10 even though every routed token was a cheap cache hit.
        assert _read_entries(ledger)[0]["premium_usd"] == pytest.approx(10.0)

    def test_cache_tokens_fold_into_total_on_litellm_path(self, tmp_path: Path) -> None:
        # Without routed_prices the routed side is priced by LiteLLM, but cache
        # tokens must still count toward the prompt total on both sides.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            0,
            0,
            ledger,
            cache_read_tokens=600_000,
            cache_creation_tokens=400_000,
        )
        entry = _read_entries(ledger)[0]
        assert entry["input_tokens"] == 1_000_000
        # CHEAP_MODEL @ $1/1M over the full 1M prompt.
        assert entry["routed_usd"] == pytest.approx(1.0, abs=1e-6)
        assert entry["premium_usd"] == pytest.approx(10.0, abs=1e-6)

    def test_negative_cache_counts_are_clamped_to_zero(self, tmp_path: Path) -> None:
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            1_000_000,
            0,
            ledger,
            cache_read_tokens=-5,
            cache_creation_tokens=-9,
            routed_prices=DS_ROUTED_PRICES,
        )
        entry = _read_entries(ledger)[0]
        assert entry["cache_read_input_tokens"] == 0
        assert entry["cache_creation_input_tokens"] == 0
        assert entry["input_tokens"] == 1_000_000

    def test_negative_output_tokens_is_clamped_to_zero(self, tmp_path: Path) -> None:
        # output_tokens comes straight off the provider's usage block; a
        # malformed negative count must be clamped like the input/cache counts,
        # never credited as a discount against the routed cost.
        ledger = tmp_path / "savings.jsonl"
        savings.record_routing_savings(
            PREMIUM_MODEL,
            CHEAP_MODEL,
            1_000_000,
            -100,
            ledger,
            routed_prices=DS_ROUTED_PRICES,
        )
        entry = _read_entries(ledger)[0]
        assert entry["output_tokens"] == 0
        # 1M miss @ 0.435/1M, zero output -> the negative count contributes nothing.
        assert entry["routed_usd"] == pytest.approx(0.435, abs=1e-7)
        # premium prices 1M input @ $10/1M with zero output.
        assert entry["premium_usd"] == pytest.approx(10.0, abs=1e-6)

    def test_round_trip_from_response_with_cache_usage_and_tier_cfg(self, tmp_path: Path) -> None:
        # The DeepSeek direct path hands a usage block with cache fields plus the
        # tier config; record_savings_from_response must price it cache-aware.
        ledger = tmp_path / "savings.jsonl"
        response = {
            "usage": {
                "input_tokens": 200_000,
                "output_tokens": 50_000,
                "cache_read_input_tokens": 700_000,
                "cache_creation_input_tokens": 100_000,
            }
        }
        tier_cfg = {
            "model": CHEAP_MODEL,
            "input_cost_per_1m": 0.435,
            "input_cached_cost_per_1m": 0.003625,
            "output_cost_per_1m": 0.87,
        }
        savings.record_savings_from_response(
            PREMIUM_MODEL, CHEAP_MODEL, response, ledger, tier_cfg=tier_cfg
        )
        entry = _read_entries(ledger)[0]
        assert entry["cache_read_input_tokens"] == 700_000
        assert entry["cache_creation_input_tokens"] == 100_000
        assert entry["routed_usd"] == pytest.approx(0.1765375, abs=1e-7)
        assert entry["premium_usd"] == pytest.approx(11.0, abs=1e-6)

    def test_without_tier_cfg_response_path_stays_single_rate(self, tmp_path: Path) -> None:
        # No tier_cfg -> routed_prices is None -> LiteLLM single-rate pricing,
        # and cache fields default to zero.
        ledger = tmp_path / "savings.jsonl"
        response = {"usage": {"input_tokens": 1_000_000, "output_tokens": 0}}
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, response, ledger)
        entry = _read_entries(ledger)[0]
        assert entry["cache_read_input_tokens"] == 0
        assert entry["routed_usd"] == pytest.approx(1.0, abs=1e-6)


class TestFailOpen:
    """Accounting must never break a served turn: registration and usage
    extraction swallow every error rather than propagating it."""

    def test_register_model_failure_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A LiteLLM register_model error during price registration must not
        propagate (savings.py:93-94). The model+price below is unique so the
        loop reaches register_model rather than short-circuiting on an
        already-registered signature.
        """
        litellm = load_litellm()

        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(litellm, "register_model", _boom)
        fresh_routes = {
            "tiers": {
                "fast": {
                    "model": "deepseek/acr-test-register-raise",
                    "input_cost_per_1m": 3.0,
                    "output_cost_per_1m": 4.0,
                }
            }
        }
        # Must return normally despite the underlying registry error.
        savings.register_tier_prices(fresh_routes)

    def test_non_numeric_usage_writes_no_ledger_entry(self, tmp_path: Path) -> None:
        """A non-coercible token count raises inside extraction; the error is
        swallowed and no ledger entry is written (savings.py:174-176).

        "abc" is truthy, so the ``or 0`` fallback does not apply and int("abc")
        raises ValueError -- the path the fail-open guard exists to protect.
        """
        ledger = tmp_path / "savings.jsonl"
        response = {"usage": {"input_tokens": "abc", "output_tokens": 5}}
        savings.record_savings_from_response(PREMIUM_MODEL, CHEAP_MODEL, response, ledger)
        assert not ledger.exists()
