"""Routing-savings accounting: tier price registration and the JSONL ledger.

Ported from Headroom's tool-router usage.py (record_routed_usage and the
Headroom metrics funnel were dropped by design). register_tier_prices teaches
LiteLLM the per-token prices declared in the tiers: config section so
litellm.cost_per_token can price routed calls; record_routing_savings appends
one honest entry per routed call to the savings ledger. Accounting is strictly
best-effort: nothing in this module ever raises into a serving path, and a
missing price never produces a fabricated savings figure.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ai_calls_router._lib import config, jsonnum
from ai_calls_router._lib.litellm_guard import load_litellm
from ai_calls_router.accounting.metrics import get_metrics, identify_provider

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue
    from ai_calls_router.accounting.shrink_stats import ShrinkStats

logger = logging.getLogger("acr.savings")

_ledger_lock = threading.Lock()
_register_lock = threading.Lock()
_registered: set[tuple[str, float, float]] = set()


class _PricingLiteLLM(Protocol):
    """LiteLLM pricing API used by savings accounting."""

    def register_model(self, model_cost: dict[str, dict[str, float | str]]) -> None: ...

    def cost_per_token(
        self, *, model: str, prompt_tokens: int, completion_tokens: int
    ) -> tuple[float, float]: ...


def register_tier_prices(routes: JsonObject) -> None:
    """Teach LiteLLM the per-1M token prices declared on each tier.

    Tiers must carry a string model plus numeric input_cost_per_1m AND
    output_cost_per_1m to be registered; anything else is skipped. Identical
    re-registration is deduplicated; a changed price re-registers. Never
    raises -- pricing is best-effort and must not block routing.

    Args:
        routes: Full routes/config mapping containing a "tiers" section.
    """
    try:
        tiers = routes.get("tiers")
        if not isinstance(tiers, dict):
            return
        litellm = cast("_PricingLiteLLM", load_litellm())
        for tier_cfg in tiers.values():
            if not isinstance(tier_cfg, dict):
                continue
            model = tier_cfg.get("model")
            input_per_1m = tier_cfg.get("input_cost_per_1m")
            output_per_1m = tier_cfg.get("output_cost_per_1m")
            if not isinstance(model, str) or not model:
                continue
            input_price = jsonnum.optional_float_value(input_per_1m)
            output_price = jsonnum.optional_float_value(output_per_1m)
            if input_price is None or output_price is None:
                continue
            input_per_token = input_price / 1_000_000
            output_per_token = output_price / 1_000_000
            signature = (model, input_per_token, output_per_token)
            with _register_lock:
                if signature in _registered:
                    continue
                provider = model.split("/", 1)[0] if "/" in model else "deepseek"
                litellm.register_model(
                    {
                        model: {
                            "input_cost_per_token": input_per_token,
                            "output_cost_per_token": output_per_token,
                            "litellm_provider": provider,
                            "mode": "chat",
                        }
                    }
                )
                _registered.add(signature)
                logger.info(
                    "registered prices for %s (in=%s out=%s per 1M)",
                    model,
                    input_per_1m,
                    output_per_1m,
                )
    except Exception as exc:
        logger.warning("tier price registration failed: %s", exc, exc_info=True)


def _routed_prices_from_tier(
    tier_cfg: JsonValue,
) -> tuple[float, float, float] | None:
    """Derive per-token (miss, cached, output) rates from a tier config.

    The DeepSeek direct path bills cache hits at a separate, far cheaper read
    rate than cache misses, which LiteLLM's single input price cannot express.
    When the tier declares numeric input and output prices, this returns the
    three per-token rates so cache hits are priced honestly; a cache-read price
    is used when declared, otherwise the miss rate (a conservative upper bound
    that never overstates savings). Returns None when the tier lacks numeric
    prices, signalling the LiteLLM single-rate path instead.

    Args:
        tier_cfg: Tier config mapping, or None on the LiteLLM path.

    Returns:
        (miss_per_token, cached_per_token, output_per_token) or None.
    """
    if not isinstance(tier_cfg, dict):
        return None
    input_per_1m = jsonnum.optional_float_value(tier_cfg.get("input_cost_per_1m"))
    output_per_1m = jsonnum.optional_float_value(tier_cfg.get("output_cost_per_1m"))
    if input_per_1m is None or output_per_1m is None:
        return None
    cached_per_1m = jsonnum.optional_float_value(tier_cfg.get("input_cached_cost_per_1m"))
    if cached_per_1m is None:
        cached_per_1m = input_per_1m
    return (
        input_per_1m / 1_000_000,
        cached_per_1m / 1_000_000,
        output_per_1m / 1_000_000,
    )


def routed_prices_from_tier(tier_cfg: JsonValue) -> tuple[float, float, float] | None:
    """Return cache-aware routed prices from a tier config."""
    return _routed_prices_from_tier(tier_cfg)


def _compute_routed_usd(
    litellm: _PricingLiteLLM,
    routed_model: str,
    total_input: int,
    hit_tokens: int,
    miss_tokens: int,
    output_tokens: int,
    routed_prices: tuple[float, float, float] | None,
) -> float | None:
    """Compute the routed-side cost in USD; None when unpriced."""
    if routed_prices is not None:
        miss_rate, cached_rate, out_rate = routed_prices
        return miss_tokens * miss_rate + hit_tokens * cached_rate + output_tokens * out_rate
    routed_in, routed_out = litellm.cost_per_token(
        model=routed_model, prompt_tokens=total_input, completion_tokens=output_tokens
    )
    if routed_in < 0:
        return None
    return routed_in + routed_out


def record_routing_savings(
    *,
    premium_model: str | None,
    routed_model: str,
    input_tokens: int,
    output_tokens: int,
    ledger: Path | None = None,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    routed_prices: tuple[float, float, float] | None = None,
    tier_name: str = "",
    tool_names: str = "",
    user_agent: str = "",
    agent: str = "",
    session_id: str = "",
    shrink_path: str = "",
    shrink_chars_before: int = 0,
    shrink_chars_after: int = 0,
) -> None:
    """Append one savings entry comparing routed cost against premium cost.

    Skips silently when the premium model is missing or equal to the routed
    model (no substitution happened), or when either side cannot be priced --
    a savings figure is only written when both costs are real and the premium
    cost is positive. Token counts are reconciled cache-aware: cache_read
    tokens are billed at the cheap cached rate, while reported input plus
    cache_creation tokens are billed at the miss rate; the premium
    counterfactual always prices the entire prompt at the premium standard
    rate. When routed_prices is given those per-token rates bill the routed
    side; otherwise LiteLLM prices the full prompt at its single input rate.
    Never raises.

    Args:
        premium_model: Model the client originally requested.
        routed_model: Model that actually served the call (true id, unmasked).
        input_tokens: Non-cached prompt tokens reported by the routed response.
        output_tokens: Completion tokens reported by the routed response.
        ledger: Ledger path override; defaults to config.ledger_path().
        cache_read_tokens: Tokens served from the provider's prefix cache.
        cache_creation_tokens: Tokens written into the prefix cache this turn.
        routed_prices: Optional (miss, cached, output) per-token rates for the
            routed side; when None, LiteLLM prices the full prompt instead.
        tier_name: Tier label for dashboards and reports.
        tool_names: Comma-separated tool names from the request.
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label (e.g. ``claude-code-cli``).
        session_id: Session fingerprint hex string.
        shrink_path: Shrink pass that ran on this turn (``reduce``/``compress``).
        shrink_chars_before: tool_result characters before the shrink pass.
        shrink_chars_after: tool_result characters after the shrink pass.
    """
    if not premium_model or premium_model == routed_model:
        return
    try:
        hit_tokens = max(int(cache_read_tokens), 0)
        miss_tokens = max(int(input_tokens), 0) + max(int(cache_creation_tokens), 0)
        total_input = hit_tokens + miss_tokens
        output_tokens = max(int(output_tokens), 0)
        litellm = cast("_PricingLiteLLM", load_litellm())
        premium_in, premium_out = litellm.cost_per_token(
            model=premium_model,
            prompt_tokens=total_input,
            completion_tokens=output_tokens,
        )
        premium_usd = premium_in + premium_out
        if premium_usd <= 0:
            return
        routed_usd = _compute_routed_usd(
            litellm,
            routed_model,
            total_input,
            hit_tokens,
            miss_tokens,
            output_tokens,
            routed_prices,
        )
        if routed_usd is None:
            return
        routed_usd_value = round(routed_usd, 8)
        premium_usd_value = round(premium_usd, 8)
        saved_usd_value = round(premium_usd - routed_usd, 8)
        entry: JsonObject = {
            "ts": int(time.time()),
            "premium_model": premium_model,
            "routed_model": routed_model,
            "input_tokens": total_input,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": hit_tokens,
            "cache_creation_input_tokens": max(int(cache_creation_tokens), 0),
            "routed_usd": routed_usd_value,
            "premium_usd": premium_usd_value,
            "saved_usd": saved_usd_value,
            "tier_name": tier_name,
            "tool_names": tool_names,
            "user_agent": user_agent[:200],
            "agent": agent,
            "session_id": session_id,
            "provider": identify_provider(routed_model),
            "shrink_path": shrink_path,
            "shrink_chars_before": max(int(shrink_chars_before), 0),
            "shrink_chars_after": max(int(shrink_chars_after), 0),
        }
        ledger = ledger if ledger is not None else config.ledger_path()
        with _ledger_lock:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            ledger.chmod(0o600)
        get_metrics().add_savings(
            routed_usd=routed_usd_value,
            premium_usd=premium_usd_value,
            saved_usd=saved_usd_value,
        )
    except Exception as exc:
        logger.warning("savings recording failed: %s", exc, exc_info=True)


def record_savings_from_response(
    *,
    premium_model: str | None,
    routed_model: str,
    response_body: JsonValue,
    ledger: Path | None = None,
    tier_cfg: JsonValue = None,
    tier_name: str = "",
    tool_names: list[str] | None = None,
    user_agent: str = "",
    agent: str = "",
    session_id: str = "",
    shrink: ShrinkStats | None = None,
) -> None:
    """Record savings using token counts taken from a routed response body.

    Extracts the Anthropic usage breakdown -- including cache_read and
    cache_creation tokens emitted by the DeepSeek direct path -- and prices the
    routed side cache-aware when the tier declares numeric prices.

    Args:
        premium_model: Model the client originally requested.
        routed_model: Model that actually served the call.
        response_body: Anthropic-format response body carrying a usage block.
        ledger: Ledger path override; defaults to config.ledger_path().
        tier_cfg: Tier config; supplies cache-aware per-token rates when present.
        tier_name: Tier label for dashboards and reports.
        tool_names: Tool names extracted from the request body.
        user_agent: Raw User-Agent header from the client.
        agent: Identified agent label (e.g. ``claude-code-cli``).
        session_id: Session fingerprint hex string.
        shrink: Read-only tool_result shrink measurement for this turn; its
            character counts ride along on the ledger entry so the dashboard's
            cumulative compression survives a proxy restart.
    """
    try:
        usage = response_body.get("usage") if isinstance(response_body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = jsonnum.int_value(usage.get("input_tokens", 0), strict=True)
        output_tokens = jsonnum.int_value(usage.get("output_tokens", 0), strict=True)
        cache_read = jsonnum.int_value(usage.get("cache_read_input_tokens", 0), strict=True)
        cache_creation = jsonnum.int_value(usage.get("cache_creation_input_tokens", 0), strict=True)
    except Exception as exc:
        logger.warning("usage extraction failed: %s", exc, exc_info=True)
        return
    record_routing_savings(
        premium_model=premium_model,
        routed_model=routed_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ledger=ledger,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        routed_prices=_routed_prices_from_tier(tier_cfg),
        tier_name=tier_name,
        tool_names=",".join(tool_names) if tool_names else "",
        user_agent=user_agent,
        agent=agent,
        session_id=session_id,
        shrink_path=shrink.path if shrink else "",
        shrink_chars_before=shrink.chars_before if shrink else 0,
        shrink_chars_after=shrink.chars_after if shrink else 0,
    )
