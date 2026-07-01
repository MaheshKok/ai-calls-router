"""Tests for wiring the context-window guard into dispatch and passthrough.

Covers route_dispatch._context_overflow_attempt (the decision-time guard that
forces premium for a session projected to overflow the tier window) and
orchestrator._premium_usage_callback (which teaches the guard the real premium
size after a passthrough completes). The store is a module global, cleared per
test.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from ai_calls_router.accounting import metrics
from ai_calls_router.proxy import orchestrator, route_dispatch
from ai_calls_router.routing import context_budget
from ai_calls_router.routing import decide as routing


@pytest.fixture(autouse=True)
def _clear_store() -> None:
    """Reset the module-global store around every test for isolation."""
    context_budget._last_context_tokens.clear()
    yield
    context_budget._last_context_tokens.clear()


def _decision(**overrides: object) -> route_dispatch.RouteDecision:
    """Return a routable Anthropic-OAuth decision with a 200K-window tier."""
    base: dict[str, object] = {
        "body": {"model": "claude-opus-4-8"},
        "anthropic_body": {"model": "claude-opus-4-8", "messages": []},
        "requested_model": "claude-opus-4-8",
        "streaming": False,
        "names": ["Bash"],
        "tier": "fast",
        "tier_cfg": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-5",
            "context_window": 200000,
            "max_tokens": 8192,
        },
        "credential": routing.TierCredential(value="oauth", auth_mode="oauth"),
        "settings": {},
        "premium_tools": ["Edit"],
    }
    base.update(overrides)
    return route_dispatch.RouteDecision(**base)  # type: ignore[arg-type]


def test_overflow_session_forces_premium_guard() -> None:
    """A session recorded above the ceiling yields a context_window_guard attempt."""
    # ceiling = 200000 - 8192 - 4096 = 187712
    context_budget.record_context_size("s", 187712, 0)
    attempt = route_dispatch._context_overflow_attempt(_decision(), "s")
    assert attempt is not None
    assert attempt.response is None  # forces passthrough
    assert attempt.reason == "context_window_guard"
    assert attempt.model == "claude-opus-4-8"
    assert attempt.tier == "fast"
    assert attempt.tool_names == ["Bash"]


def test_under_ceiling_routes_normally() -> None:
    """A session under the ceiling returns None (route as usual)."""
    context_budget.record_context_size("s", 100000, 0)
    assert route_dispatch._context_overflow_attempt(_decision(), "s") is None


def test_no_context_window_disables_guard() -> None:
    """A tier without context_window never guards, even for a huge stored size."""
    context_budget.record_context_size("s", 999999, 0)
    decision = _decision(
        tier_cfg={"provider": "anthropic", "model": "m", "max_tokens": 8192},
    )
    assert route_dispatch._context_overflow_attempt(decision, "s") is None


def test_unknown_session_routes_normally() -> None:
    """A first-seen session gets its fail-open routed attempt."""
    assert route_dispatch._context_overflow_attempt(_decision(), "unseen") is None


def test_empty_session_routes_normally() -> None:
    """An empty session key skips the guard."""
    context_budget.record_context_size("", 999999, 0)
    assert route_dispatch._context_overflow_attempt(_decision(), "") is None


def test_missing_max_tokens_uses_default_reserve() -> None:
    """Without a tier max_tokens the guard reserves the default output budget."""
    # ceiling = 200000 - _DEFAULT_OUTPUT_RESERVE - 4096 = 200000 - 32000 - 4096 = 163904
    decision = _decision(
        tier_cfg={"provider": "anthropic", "model": "m", "context_window": 200000},
    )
    assert route_dispatch._DEFAULT_OUTPUT_RESERVE == 32000
    context_budget.record_context_size("s", 163903, 0)
    assert route_dispatch._context_overflow_attempt(decision, "s") is None
    context_budget.record_context_size("s", 163904, 0)
    assert route_dispatch._context_overflow_attempt(decision, "s") is not None


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    """Build a passthrough usage dict as the callback receives it."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def test_premium_callback_records_session_size() -> None:
    """A completed passthrough teaches the guard the real premium size."""
    m = cast("metrics.Metrics", MagicMock())
    callback = orchestrator._premium_usage_callback(m=m, request_id="r", session="s")
    callback(200, _usage(187712, 0), 1.0)
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=8192) is True


def test_premium_callback_none_session_records_nothing() -> None:
    """A passthrough with no session fingerprint records nothing."""
    m = cast("metrics.Metrics", MagicMock())
    callback = orchestrator._premium_usage_callback(m=m, request_id="r", session=None)
    callback(200, _usage(187712, 0), 1.0)
    assert len(context_budget._last_context_tokens) == 0
