"""Spec-derived tests for the tool_result shrink measurement module.

Tests are written from the contracts in ai_calls_router.accounting.shrink_stats
(docstrings and types), not from the implementation: tool_result_chars counts
only tool_result text, ShrinkStats derives savings/ratio/token estimates from a
before/after pair, and compute_shrink packages the two counts. Adversarial cases
cover empty/missing/malformed bodies, both content shapes, boundary ratios, and
the token-estimate divisor.
"""

from __future__ import annotations

import dataclasses

import pytest

from ai_calls_router.accounting.shrink_stats import (
    DEFAULT_CHARS_PER_TOKEN,
    ShrinkStats,
    compute_shrink,
    tool_result_chars,
)


def _tool_result_msg(content: object) -> dict[str, object]:
    """Build a user message wrapping one tool_result block with given content."""
    return {"role": "user", "content": [{"type": "tool_result", "content": content}]}


# ── tool_result_chars ──────────────────────────────────────────────────────


def test_tool_result_chars_returns_zero_for_empty_body() -> None:
    assert tool_result_chars({}) == 0


def test_tool_result_chars_returns_zero_when_messages_not_list() -> None:
    assert tool_result_chars({"messages": "nope"}) == 0
    assert tool_result_chars({"messages": None}) == 0


def test_tool_result_chars_returns_zero_for_empty_message_list() -> None:
    assert tool_result_chars({"messages": []}) == 0


def test_tool_result_chars_skips_non_dict_messages() -> None:
    assert tool_result_chars({"messages": ["x", 5, None]}) == 0


def test_tool_result_chars_counts_string_content() -> None:
    body = {"messages": [_tool_result_msg("hello")]}
    assert tool_result_chars(body) == 5


def test_tool_result_chars_counts_text_blocks_in_list_content() -> None:
    content = [
        {"type": "text", "text": "abc"},
        {"type": "text", "text": "de"},
    ]
    body = {"messages": [_tool_result_msg(content)]}
    assert tool_result_chars(body) == 5


def test_tool_result_chars_ignores_non_text_blocks_in_list() -> None:
    content = [
        {"type": "text", "text": "keep"},
        {"type": "image", "source": "ignored-large-payload"},
        {"type": "text"},  # missing text key
        {"type": "text", "text": 123},  # non-string text
        "not-a-dict",
    ]
    body = {"messages": [_tool_result_msg(content)]}
    assert tool_result_chars(body) == 4


def test_tool_result_chars_ignores_non_tool_result_blocks() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "user prompt not counted"},
            {"type": "tool_result", "content": "counted"},
        ],
    }
    assert tool_result_chars({"messages": [msg]}) == len("counted")


def test_tool_result_chars_returns_zero_when_message_content_not_list() -> None:
    msg = {"role": "user", "content": "plain string content"}
    assert tool_result_chars({"messages": [msg]}) == 0


def test_tool_result_chars_returns_zero_for_unhandled_content_shape() -> None:
    body = {"messages": [_tool_result_msg({"unexpected": "dict"})]}
    assert tool_result_chars(body) == 0


def test_tool_result_chars_sums_across_multiple_messages() -> None:
    body = {
        "messages": [
            _tool_result_msg("aa"),
            _tool_result_msg([{"type": "text", "text": "bbb"}]),
            {"role": "assistant", "content": [{"type": "text", "text": "ignored"}]},
        ]
    }
    assert tool_result_chars(body) == 5


def test_tool_result_chars_counts_unicode_by_code_points() -> None:
    body = {"messages": [_tool_result_msg("héllo😀")]}
    assert tool_result_chars(body) == 6


# ── ShrinkStats.chars_saved ─────────────────────────────────────────────────


def test_chars_saved_is_difference() -> None:
    assert ShrinkStats(path="reduce", chars_before=100, chars_after=40).chars_saved == 60


def test_chars_saved_is_zero_for_no_op() -> None:
    assert ShrinkStats(path="none", chars_before=50, chars_after=50).chars_saved == 0


def test_chars_saved_floors_at_zero_when_after_exceeds_before() -> None:
    assert ShrinkStats(path="compress", chars_before=10, chars_after=25).chars_saved == 0


# ── ShrinkStats.ratio ───────────────────────────────────────────────────────


def test_ratio_is_fraction_removed() -> None:
    assert ShrinkStats(path="reduce", chars_before=200, chars_after=50).ratio == 0.75


def test_ratio_zero_when_before_zero() -> None:
    assert ShrinkStats(path="none", chars_before=0, chars_after=0).ratio == 0.0


def test_ratio_zero_when_before_negative() -> None:
    # Defensive: a non-positive baseline must not divide.
    assert ShrinkStats(path="none", chars_before=-5, chars_after=0).ratio == 0.0


def test_ratio_is_zero_for_no_op() -> None:
    assert ShrinkStats(path="none", chars_before=80, chars_after=80).ratio == 0.0


# ── ShrinkStats.est_tokens_saved ────────────────────────────────────────────


def test_est_tokens_saved_uses_default_divisor() -> None:
    stats = ShrinkStats(path="reduce", chars_before=400, chars_after=0)
    assert stats.est_tokens_saved() == int(400 / DEFAULT_CHARS_PER_TOKEN)


def test_est_tokens_saved_truncates_toward_zero() -> None:
    stats = ShrinkStats(path="reduce", chars_before=10, chars_after=0)
    # 10 / 3.5 == 2.857..., truncated to 2.
    assert stats.est_tokens_saved() == 2


def test_est_tokens_saved_custom_divisor() -> None:
    stats = ShrinkStats(path="compress", chars_before=100, chars_after=0)
    assert stats.est_tokens_saved(chars_per_token=4.0) == 25


def test_est_tokens_saved_zero_when_divisor_zero() -> None:
    stats = ShrinkStats(path="reduce", chars_before=100, chars_after=0)
    assert stats.est_tokens_saved(chars_per_token=0) == 0


def test_est_tokens_saved_zero_when_divisor_negative() -> None:
    stats = ShrinkStats(path="reduce", chars_before=100, chars_after=0)
    assert stats.est_tokens_saved(chars_per_token=-2.0) == 0


def test_est_tokens_saved_zero_when_nothing_saved() -> None:
    stats = ShrinkStats(path="none", chars_before=50, chars_after=50)
    assert stats.est_tokens_saved() == 0


# ── ShrinkStats immutability ────────────────────────────────────────────────


def test_shrink_stats_is_frozen() -> None:
    stats = ShrinkStats(path="reduce", chars_before=10, chars_after=5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.chars_before = 99  # type: ignore[misc]


# ── compute_shrink ──────────────────────────────────────────────────────────


def test_compute_shrink_measures_before_and_after() -> None:
    before = {"messages": [_tool_result_msg("aaaaaaaaaa")]}  # 10 chars
    after = {"messages": [_tool_result_msg("aaa")]}  # 3 chars
    stats = compute_shrink(path="compress", before=before, after=after)
    assert stats.path == "compress"
    assert stats.chars_before == 10
    assert stats.chars_after == 3
    assert stats.chars_saved == 7


def test_compute_shrink_counts_responses_function_outputs() -> None:
    before = {
        "input": [
            {"type": "function_call_output", "call_id": "call_1", "output": "abcdef"},
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "xyz"},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "ignore"}],
            },
        ]
    }
    after = {
        "input": [
            {"type": "function_call_output", "call_id": "call_1", "output": "abc"},
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "x"},
        ]
    }
    stats = compute_shrink(path="compress", before=before, after=after)
    assert stats.chars_before == 9
    assert stats.chars_after == 4
    assert stats.chars_saved == 5


def test_compute_shrink_no_op_identity_gives_zero_saved() -> None:
    body = {"messages": [_tool_result_msg("unchanged")]}
    stats = compute_shrink(path="reduce", before=body, after=body)
    assert stats.chars_before == stats.chars_after
    assert stats.chars_saved == 0
    assert stats.ratio == 0.0


def test_compute_shrink_handles_bodies_without_tool_results() -> None:
    before = {"messages": [{"role": "user", "content": "hi"}]}
    after = {"messages": [{"role": "user", "content": "hi"}]}
    stats = compute_shrink(path="none", before=before, after=after)
    assert stats == ShrinkStats(path="none", chars_before=0, chars_after=0)
