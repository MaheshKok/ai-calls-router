"""Guarded lazy import of litellm.

litellm calls dotenv's load_dotenv() at import time, which silently injects
any project .env keys into the proxy process environment -- where they could
later be picked up by key_env resolution. This module imports litellm once,
on first use, and deletes every environment variable that import leaked
(Headroom's proven guard, backends/litellm.py). Importing lazily also keeps
the ~1s litellm import cost out of CLI commands that never route.
"""

from __future__ import annotations

import os
import sys
import threading
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonValue

_lock = threading.Lock()
_litellm: ModuleType | None = None


class _PricingFallback(ModuleType):
    """Pricing-only LiteLLM fallback used when the import is unavailable."""

    def __init__(self) -> None:
        super().__init__("litellm_pricing_fallback")
        self._prices: dict[str, tuple[float, float]] = {}

    def register_model(self, model_cost: dict[str, dict[str, float | str]]) -> None:
        """Register per-token prices using LiteLLM's model-cost shape."""
        for model, prices in model_cost.items():
            input_price = prices.get("input_cost_per_token")
            output_price = prices.get("output_cost_per_token")
            if isinstance(input_price, int | float) and isinstance(output_price, int | float):
                self._prices[model] = (float(input_price), float(output_price))

    def cost_per_token(
        self, *, model: str, prompt_tokens: int, completion_tokens: int
    ) -> tuple[float, float]:
        """Return the registered prompt and completion cost for a model."""
        try:
            input_price, output_price = self._prices[model]
        except KeyError as exc:
            raise ValueError(f"Model {model} isn't mapped yet") from exc
        return prompt_tokens * input_price, completion_tokens * output_price

    async def acompletion(self, **_: JsonValue) -> None:
        """Fail open through the routed-call error path when LiteLLM is absent."""
        raise ImportError("litellm import failed; pricing fallback cannot serve completions")


def load_litellm() -> ModuleType:
    """Import litellm with the dotenv-leak guard and cache the module.

    Returns:
        The litellm module.

    Raises:
        ImportError: If litellm is not installed.
    """
    global _litellm
    with _lock:
        if _litellm is None:
            env_before = set(os.environ)
            try:
                import litellm
            except Exception:
                sys.modules.pop("litellm", None)
                _litellm = _PricingFallback()
                return _litellm

            for leaked in set(os.environ) - env_before:
                del os.environ[leaked]
            _litellm = litellm
    return _litellm
