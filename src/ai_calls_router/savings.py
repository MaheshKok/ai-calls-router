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
from typing import Any

from ai_calls_router import config
from ai_calls_router.litellm_guard import load_litellm

logger = logging.getLogger("acr.savings")

_ledger_lock = threading.Lock()
_register_lock = threading.Lock()
_registered: set[tuple[str, float, float]] = set()


def _is_number(value: Any) -> bool:
    """Check whether a value is a real number (bool excluded).

    Args:
        value: Candidate price value from config.

    Returns:
        True for int/float values that are not bool.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def register_tier_prices(routes: dict[str, Any]) -> None:
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
        litellm = load_litellm()
        for tier_cfg in tiers.values():
            if not isinstance(tier_cfg, dict):
                continue
            model = tier_cfg.get("model")
            input_per_1m = tier_cfg.get("input_cost_per_1m")
            output_per_1m = tier_cfg.get("output_cost_per_1m")
            if not isinstance(model, str) or not model:
                continue
            if not _is_number(input_per_1m) or not _is_number(output_per_1m):
                continue
            input_per_token = float(input_per_1m) / 1_000_000
            output_per_token = float(output_per_1m) / 1_000_000
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
        logger.warning("tier price registration failed: %s", exc)


def record_routing_savings(
    premium_model: str | None,
    routed_model: str,
    input_tokens: int,
    output_tokens: int,
    ledger: Path | None = None,
) -> None:
    """Append one savings entry comparing routed cost against premium cost.

    Skips silently when the premium model is missing or equal to the routed
    model (no substitution happened), or when either side cannot be priced
    by LiteLLM -- a savings figure is only written when both costs are real
    and the premium cost is positive. Never raises.

    Args:
        premium_model: Model the client originally requested.
        routed_model: Model that actually served the call (true id, unmasked).
        input_tokens: Prompt tokens reported by the routed response.
        output_tokens: Completion tokens reported by the routed response.
        ledger: Ledger path override; defaults to config.ledger_path().
    """
    if not premium_model or premium_model == routed_model:
        return
    try:
        litellm = load_litellm()
        routed_in, routed_out = litellm.cost_per_token(
            model=routed_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        premium_in, premium_out = litellm.cost_per_token(
            model=premium_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        routed_usd = routed_in + routed_out
        premium_usd = premium_in + premium_out
        if premium_usd <= 0:
            return
        entry = {
            "ts": int(time.time()),
            "premium_model": premium_model,
            "routed_model": routed_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "routed_usd": round(routed_usd, 8),
            "premium_usd": round(premium_usd, 8),
            "saved_usd": round(premium_usd - routed_usd, 8),
        }
        ledger = ledger if ledger is not None else config.ledger_path()
        with _ledger_lock:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("savings recording failed: %s", exc)


def record_savings_from_response(
    premium_model: str | None,
    routed_model: str,
    response_body: Any,
    ledger: Path | None = None,
) -> None:
    """Record savings using token counts taken from a routed response body.

    Args:
        premium_model: Model the client originally requested.
        routed_model: Model that actually served the call.
        response_body: Anthropic-format response body carrying a usage block.
        ledger: Ledger path override; defaults to config.ledger_path().
    """
    try:
        usage = response_body.get("usage") if isinstance(response_body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
    except Exception as exc:
        logger.warning("usage extraction failed: %s", exc)
        return
    record_routing_savings(premium_model, routed_model, input_tokens, output_tokens, ledger)
