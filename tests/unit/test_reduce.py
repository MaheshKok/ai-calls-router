"""Spec-derived tests for ai_calls_router.routing.reduce.

Contract under test: reduce_text strips ANSI CSI escapes and collapses runs of
blank lines (both lossless and always on), and optionally drops consecutive
duplicate lines; it is a pure function, so identical input yields byte-identical
output. reduce_tool_results applies that reduction to every tool_result in a
request body independent of message position, never mutates the input, and
returns the original body object unchanged when nothing is stripped. The
position independence is what keeps a provider's prefix cache hitting: the same
tool_result must reduce to the same bytes whether it is the newest message or an
old one.
"""

from __future__ import annotations

import copy
from typing import Any

from ai_calls_router.routing import reduce


def _tool_result_body(
    content: Any, *, trailing: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build a minimal body with one tool_result, optional trailing messages."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
        },
    ]
    if trailing:
        messages.extend(trailing)
    return {"model": "claude", "messages": messages}


def _result_content(body: dict[str, Any], index: int) -> Any:
    """Pull the content of the tool_result block in message at index."""
    return body["messages"][index]["content"][0]["content"]


# --- ANSI stripping -------------------------------------------------------


def test_reduce_text_strips_sgr_colour_codes() -> None:
    assert reduce_text_only("\x1b[31mred\x1b[0m") == "red"


def test_reduce_text_strips_cursor_and_clear_sequences() -> None:
    # \x1b[2K (erase line) and \x1b[1G (cursor column) are CSI codes too.
    assert reduce_text_only("a\x1b[2K\x1b[1Gb") == "ab"


def test_reduce_text_strips_multi_param_sgr() -> None:
    assert reduce_text_only("\x1b[1;32;40mx\x1b[0m") == "x"


def test_reduce_text_string_of_only_ansi_becomes_empty() -> None:
    assert reduce_text_only("\x1b[31m\x1b[0m") == ""


# --- blank-line collapse --------------------------------------------------


def test_reduce_text_collapses_blank_line_run_to_single_blank() -> None:
    assert reduce_text_only("a\n\n\n\nb") == "a\n\nb"


def test_reduce_text_collapses_whitespace_only_line_runs() -> None:
    # Whitespace-only lines count as blank for run collapsing, but the kept
    # blank line's bytes are preserved -- reduction removes redundant lines,
    # it never rewrites the content of a line it keeps.
    assert reduce_text_only("a\n   \n\t\nb") == "a\n   \nb"


def test_reduce_text_keeps_single_blank_line() -> None:
    assert reduce_text_only("a\n\nb") == "a\n\nb"


def test_reduce_text_keeps_separated_repeats_across_blank() -> None:
    # Identical lines separated by a blank are not consecutive duplicates.
    assert reduce_text_only("x\n\nx") == "x\n\nx"


# --- duplicate-line drop (opt-in) -----------------------------------------


def test_reduce_text_drops_consecutive_duplicate_lines_when_enabled() -> None:
    assert reduce.reduce_text("x\nx\nx\ny", drop_duplicate_lines=True) == "x\ny"


def test_reduce_text_keeps_duplicate_lines_by_default() -> None:
    # Default must preserve adjacent identical lines (e.g. nested braces in code).
    assert reduce_text_only("}\n}\n}") == "}\n}\n}"


def test_reduce_text_duplicate_drop_does_not_merge_nonadjacent() -> None:
    assert reduce.reduce_text("a\nb\na", drop_duplicate_lines=True) == "a\nb\na"


# --- no-op when nothing to strip ------------------------------------------


def test_reduce_text_returns_equal_string_for_clean_input() -> None:
    clean = "line one\nline two\nline three"
    assert reduce_text_only(clean) == clean


def test_reduce_text_empty_string_is_empty() -> None:
    assert reduce_text_only("") == ""


def test_reduce_tool_results_returns_same_object_when_clean() -> None:
    body = _tool_result_body("already clean\noutput")
    assert reduce.reduce_tool_results(body) is body


def test_reduce_tool_results_returns_same_object_without_messages() -> None:
    body: dict[str, Any] = {"model": "claude"}
    assert reduce.reduce_tool_results(body) is body


# --- byte-stability across two simulated turns ----------------------------


def test_same_tool_result_reduces_identically_regardless_of_position() -> None:
    # The exact bytes a routed turn would re-send for the SAME tool_result must
    # match whether that result is the newest message (turn one) or an older one
    # followed by a fresh round (turn two) -- otherwise the prefix cache misses.
    noisy = "\x1b[32mok\x1b[0m\nresult\n\n\n\nresult-tail"
    turn_one = _tool_result_body(noisy)
    turn_two = _tool_result_body(
        noisy,
        trailing=[
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "fresh"}],
            },
        ],
    )

    reduced_one = reduce.reduce_tool_results(turn_one)
    reduced_two = reduce.reduce_tool_results(turn_two)

    # The shared tool_result sits at index 1 in both bodies.
    assert _result_content(reduced_one, 1) == _result_content(reduced_two, 1)
    assert _result_content(reduced_one, 1) == "ok\nresult\n\nresult-tail"


# --- body-level behaviour & immutability ----------------------------------


def test_reduce_tool_results_reduces_string_content() -> None:
    body = _tool_result_body("\x1b[31merr\x1b[0m\n\n\n\ntail")
    reduced = reduce.reduce_tool_results(body)
    assert _result_content(reduced, 1) == "err\n\ntail"


def test_reduce_tool_results_reduces_text_block_list_content() -> None:
    body = _tool_result_body([{"type": "text", "text": "\x1b[31mred\x1b[0m"}])
    reduced = reduce.reduce_tool_results(body)
    assert _result_content(reduced, 1) == [{"type": "text", "text": "red"}]


def test_reduce_tool_results_never_mutates_input() -> None:
    body = _tool_result_body("\x1b[31mdirty\x1b[0m")
    snapshot = copy.deepcopy(body)
    reduce.reduce_tool_results(body)
    assert body == snapshot


def test_reduce_tool_results_leaves_non_text_blocks_untouched() -> None:
    image_block = {"type": "image", "source": {"type": "base64", "data": "AAAA"}}
    body = _tool_result_body([image_block, {"type": "text", "text": "\x1b[31mx\x1b[0m"}])
    reduced = reduce.reduce_tool_results(body)
    assert _result_content(reduced, 1) == [image_block, {"type": "text", "text": "x"}]


def test_reduce_tool_results_ignores_non_tool_result_blocks() -> None:
    # A plain text block in a user message must pass through untouched.
    body: dict[str, Any] = {
        "model": "claude",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "\x1b[31mhi\x1b[0m"}]}],
    }
    assert reduce.reduce_tool_results(body) is body


def test_reduce_tool_results_survives_malformed_content() -> None:
    body: dict[str, Any] = {
        "model": "claude",
        "messages": [
            "not-a-dict",
            {"role": "user", "content": "plain string"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": 123}],
            },
        ],
    }
    # Integer tool_result content is left as-is; nothing raises.
    result = reduce.reduce_tool_results(body)
    assert result is body


def reduce_text_only(text: str) -> str:
    """Call reduce_text with the default (duplicate-line drop disabled)."""
    return reduce.reduce_text(text)
