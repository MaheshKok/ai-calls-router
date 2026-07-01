"""Tests for the per-session context-window guard store.

context_budget remembers each session's last observed conversation size and
projects whether the next turn would overflow a tier's context window. These
tests derive the trip point independently from the documented contract
(ceiling = context_window - output_reserve - safety_margin), cover the boundary
and clamping rules, and exercise the bounded-LRU eviction. State is a module
global, so each test clears it first.
"""

from __future__ import annotations

import pytest

from ai_calls_router.routing import context_budget

# Independently restated from the module contract (do not import the constant so
# a silent change to it fails these tests).
_SAFETY = 4096


@pytest.fixture(autouse=True)
def _clear_store() -> None:
    """Reset the module-global store around every test for isolation."""
    context_budget._last_context_tokens.clear()
    yield
    context_budget._last_context_tokens.clear()


def test_records_and_trips_at_exact_ceiling() -> None:
    """last_total == ceiling trips (>= boundary)."""
    # ceiling = 200000 - 8192 - 4096 = 187712
    context_budget.record_context_size("s", 100000, 87712)
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=8192) is True


def test_one_below_ceiling_routes() -> None:
    """last_total one under the ceiling does not trip."""
    context_budget.record_context_size("s", 100000, 87711)  # 187711 < 187712
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=8192) is False


def test_one_above_ceiling_trips() -> None:
    """last_total above the ceiling trips."""
    context_budget.record_context_size("s", 187713, 0)
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=8192) is True


def test_unknown_session_routes() -> None:
    """A session with no record never trips (first turn still gets fail-open)."""
    assert context_budget.would_overflow("nope", context_window=200000, output_reserve=0) is False


def test_none_session_never_records_or_trips() -> None:
    """A missing session key is a no-op on both write and read."""
    context_budget.record_context_size(None, 999999, 999999)
    assert len(context_budget._last_context_tokens) == 0
    assert context_budget.would_overflow(None, context_window=200000, output_reserve=0) is False


def test_empty_session_string_is_noop() -> None:
    """An empty session string is falsy and skipped."""
    context_budget.record_context_size("", 999999, 0)
    assert "" not in context_budget._last_context_tokens
    assert context_budget.would_overflow("", context_window=200000, output_reserve=0) is False


def test_all_negative_or_zero_total_not_recorded() -> None:
    """A turn with no usable size records nothing."""
    context_budget.record_context_size("z", 0, 0)
    context_budget.record_context_size("n", -5, -5)
    assert "z" not in context_budget._last_context_tokens
    assert "n" not in context_budget._last_context_tokens


def test_negative_input_clamped_in_sum() -> None:
    """A negative component is clamped to zero, not subtracted from the total."""
    # max(-100, 0) + 187712 = 187712 == ceiling -> trips.
    context_budget.record_context_size("s", -100, 187712)
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=8192) is True


def test_context_window_nonpositive_disables_guard() -> None:
    """A non-positive window disables the guard regardless of stored size."""
    context_budget.record_context_size("s", 999999, 0)
    assert context_budget.would_overflow("s", context_window=0, output_reserve=0) is False
    assert context_budget.would_overflow("s", context_window=-5, output_reserve=0) is False


def test_latest_record_overwrites_not_max() -> None:
    """A newer, smaller size replaces the old one (the store is not a max)."""
    context_budget.record_context_size("s", 999999, 0)
    context_budget.record_context_size("s", 10, 0)
    # ceiling = 200000 - 0 - 4096 = 195904; 10 < 195904 -> routes.
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=0) is False


def test_output_reserve_negative_clamped_to_zero() -> None:
    """A negative reserve is clamped so it cannot raise the ceiling above the window."""
    context_budget.record_context_size("s", 195904, 0)  # == 200000 - 0 - 4096
    assert context_budget.would_overflow("s", context_window=200000, output_reserve=-100) is True


def test_lru_evicts_oldest_session() -> None:
    """Filling past the cap evicts the oldest, un-accessed session."""
    context_budget.record_context_size("victim", 999999, 0)
    for i in range(context_budget._MAX_SESSIONS):
        context_budget.record_context_size(f"s{i}", 1, 0)
    assert "victim" not in context_budget._last_context_tokens
    assert context_budget.would_overflow("victim", context_window=200000, output_reserve=0) is False


def test_access_moves_to_end_and_survives_eviction() -> None:
    """Reading a session via would_overflow refreshes it so it is not evicted next."""
    context_budget.record_context_size("hot", 999999, 0)
    for i in range(context_budget._MAX_SESSIONS - 1):
        context_budget.record_context_size(f"s{i}", 1, 0)
    # "hot" is currently the oldest; reading it moves it to the most-recent end.
    assert context_budget.would_overflow("hot", context_window=200000, output_reserve=0) is True
    context_budget.record_context_size("extra", 1, 0)  # evicts the now-oldest (s0), not hot
    assert "hot" in context_budget._last_context_tokens
    assert context_budget.would_overflow("hot", context_window=200000, output_reserve=0) is True
