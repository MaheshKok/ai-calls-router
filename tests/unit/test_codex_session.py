"""Tests for the per-socket Codex conversation store (Option B P1).

These are written from the CodexSession contract, not its implementation: the
store reconstructs the full stateless input a routed Codex turn needs by
accumulating prior delta inputs and streamed output items, and it virtualizes
``previous_response_id`` by issuing its own response ids. The decisive case is
the function_call/function_call_output pairing that the stateless backend
rejects with 400 when history is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_calls_router.proxy.codex_session import CodexSession

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray


def _msg(role: str, text: str) -> dict[str, object]:
    """Build a minimal Responses message item."""
    return {"type": "message", "role": role, "content": text}


def _call(call_id: str, name: str) -> dict[str, object]:
    """Build a function_call output item."""
    return {"type": "function_call", "name": name, "call_id": call_id}


def _call_output(call_id: str, output: str) -> dict[str, object]:
    """Build a function_call_output input item."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def test_reconstruct_first_turn_returns_delta_when_no_previous_id() -> None:
    session = CodexSession()
    delta: JsonArray = [_msg("user", "hello")]

    assert session.reconstruct_input(None, delta) == [_msg("user", "hello")]


def test_reconstruct_first_turn_with_empty_delta_returns_empty() -> None:
    session = CodexSession()

    assert session.reconstruct_input(None, []) == []


def test_reconstruct_appends_delta_after_recorded_history_in_order() -> None:
    session = CodexSession()
    turn1_input: JsonArray = [_msg("user", "first")]
    turn1_output: JsonArray = [_msg("assistant", "reply")]
    response_id = session.record_response(full_input=turn1_input, output=turn1_output)

    turn2_delta: JsonArray = [_msg("user", "second")]
    result = session.reconstruct_input(response_id, turn2_delta)

    assert result == [
        _msg("user", "first"),
        _msg("assistant", "reply"),
        _msg("user", "second"),
    ]


def test_reconstruct_pairs_function_call_with_later_output() -> None:
    # Regression for the original 400 "No tool output found for function call":
    # turn 1's output holds the function_call; turn 2's delta holds its output.
    # The reconstructed input must contain BOTH so the stateless backend pairs them.
    session = CodexSession()
    turn1_input: JsonArray = [_msg("user", "do a thing")]
    turn1_output: JsonArray = [_call("call_X", "exec_command")]
    response_id = session.record_response(full_input=turn1_input, output=turn1_output)

    turn2_delta: JsonArray = [_call_output("call_X", "done")]
    result = session.reconstruct_input(response_id, turn2_delta)

    assert _call("call_X", "exec_command") in result
    assert _call_output("call_X", "done") in result
    assert result.index(_call("call_X", "exec_command")) < result.index(
        _call_output("call_X", "done")
    )


def test_reconstruct_accumulates_across_three_turns() -> None:
    session = CodexSession()
    id1 = session.record_response(full_input=[_msg("user", "a")], output=[_msg("assistant", "b")])
    full2 = session.reconstruct_input(id1, [_call_output("c1", "x")])
    id2 = session.record_response(full_input=full2, output=[_call("c2", "search")])
    full3 = session.reconstruct_input(id2, [_call_output("c2", "y")])

    assert len(full2) == 3
    assert len(full3) == 5
    assert full3[-1] == _call_output("c2", "y")


def test_reconstruct_with_unknown_previous_id_returns_delta_only() -> None:
    session = CodexSession()
    delta: JsonArray = [_msg("user", "orphan")]

    assert session.reconstruct_input("resp_never_issued", delta) == [_msg("user", "orphan")]


def test_knows_is_false_for_unissued_id_and_true_after_record() -> None:
    session = CodexSession()
    assert session.knows("resp_anything") is False

    response_id = session.record_response(full_input=[], output=[])

    assert session.knows(response_id) is True


def test_record_returns_unique_response_ids() -> None:
    session = CodexSession()

    ids = {session.record_response(full_input=[], output=[]) for _ in range(50)}

    assert len(ids) == 50


def test_record_returns_response_shaped_id() -> None:
    session = CodexSession()

    response_id = session.record_response(full_input=[], output=[])

    assert isinstance(response_id, str)
    assert response_id.startswith("resp_")


def test_reconstruct_result_is_a_fresh_list_not_aliasing_storage() -> None:
    session = CodexSession()
    response_id = session.record_response(full_input=[_msg("user", "x")], output=[])

    first = session.reconstruct_input(response_id, [])
    first.append(_msg("user", "mutation"))
    second = session.reconstruct_input(response_id, [])

    assert second == [_msg("user", "x")]


def test_record_does_not_alias_caller_lists() -> None:
    session = CodexSession()
    full_input: JsonArray = [_msg("user", "x")]
    output: JsonArray = [_msg("assistant", "y")]
    response_id = session.record_response(full_input=full_input, output=output)

    full_input.append(_msg("user", "late mutation"))
    output.append(_msg("assistant", "late mutation"))

    assert session.reconstruct_input(response_id, []) == [
        _msg("user", "x"),
        _msg("assistant", "y"),
    ]


def test_sessions_are_isolated_from_each_other() -> None:
    session_a = CodexSession()
    session_b = CodexSession()
    id_a = session_a.record_response(full_input=[_msg("user", "a")], output=[])

    assert session_b.knows(id_a) is False
