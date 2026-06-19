"""Tests for the hybrid sticky-route Codex WebSocket relay.

These exercise the routing layer's contract, not the HTTP transport (which is
covered by test_codex_direct): the direct Codex call and shared accounting are
mocked at their boundaries so the tests assert relay behavior -- frame parsing,
the per-turn tier decision over reconstructed input, passthrough observation
that records turns under the upstream's real id, and the sticky switch to the
HTTP bridge that synthesizes frames back to the client.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from starlette.websockets import WebSocketDisconnect

from ai_calls_router.accounting import shrink_stats
from ai_calls_router.proxy import codex_ws_router

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonArray, JsonObject

_ROUTES: JsonObject = {
    "settings": {"tier_precedence": ["premium", "structured", "code", "fast", "crud"]},
    "agents": {
        "codex": {
            "tools": {"exec_command": "fast", "apply_patch": "premium"},
            "premium_tools": ["apply_patch"],
        }
    },
    "tiers": {"fast": {"model": "codex/gpt-5.4-mini", "provider": "codex", "key_env": "oauth"}},
}


def _routes() -> JsonObject:
    return _ROUTES


def _create_frame(
    input_items: JsonArray,
    *,
    previous_response_id: str | None = None,
    model: str = "gpt-5.5-codex",
) -> str:
    frame: JsonObject = {"type": "response.create", "model": model, "input": input_items}
    if previous_response_id is not None:
        frame["previous_response_id"] = previous_response_id
    return json.dumps(frame)


def _item_done(item: JsonObject) -> str:
    return json.dumps({"type": "response.output_item.done", "item": item})


def _completed(response_id: str) -> str:
    return json.dumps(
        {"type": "response.completed", "response": {"id": response_id, "output": [], "usage": None}}
    )


def _function_call(call_id: str, name: str) -> JsonObject:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _call_output(call_id: str) -> JsonObject:
    return {"type": "function_call_output", "call_id": call_id, "output": "ok"}


def _exec_result_turn() -> JsonArray:
    """Return a routable delta: the exec_command call paired with its output."""
    return [_function_call("c1", "exec_command"), _call_output("c1")]


class _FakeClient:
    """A scripted client WebSocket that disconnects when its script is exhausted."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = iter(incoming)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def receive_text(self) -> str:
        try:
            return next(self._incoming)
        except StopIteration:
            raise WebSocketDisconnect(code=1000) from None

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


class _FakeUpstream:
    """A persistent upstream iterator; each ``async for`` resumes where it left off."""

    def __init__(self, frames: list[str | bytes]) -> None:
        self._frames = frames
        self._index = 0
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def __aiter__(self) -> _FakeUpstream:
        return self

    async def __anext__(self) -> str | bytes:
        if self._index >= len(self._frames):
            raise StopAsyncIteration
        frame = self._frames[self._index]
        self._index += 1
        return frame


def _routed_response(response_id: str, text: str) -> JsonObject:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


def _install_mocks(monkeypatch: pytest.MonkeyPatch, responses: list[JsonObject | None]):
    """Stub the HTTP Codex call and accounting; return the captured-bodies list."""
    bodies: list[JsonObject] = []
    queue = iter(responses)

    async def fake_responses_call(*, body, **_kwargs):  # type: ignore[no-untyped-def]
        bodies.append(body)
        response = next(queue)
        if response is None:
            return None
        usage = (5, 2, 0, 0)
        shrink = shrink_stats.compute_shrink(path="none", before={}, after={})
        return response, usage, shrink

    outcomes: list[object] = []

    async def fake_record(outcome) -> None:  # type: ignore[no-untyped-def]
        outcomes.append(outcome)

    monkeypatch.setattr(codex_ws_router.codex_direct, "responses_call", fake_responses_call)
    monkeypatch.setattr(codex_ws_router.routed_call, "record_route_outcome", fake_record)
    return bodies, outcomes


@pytest.fixture(autouse=True)
def _clear_sessions() -> Iterator[None]:
    """Reset the cross-connection session store so tests stay independent."""
    codex_ws_router._SESSIONS.clear()
    yield
    codex_ws_router._SESSIONS.clear()


def test_parse_response_create_reads_flat_and_nested_and_rejects_others() -> None:
    flat = _create_frame([{"type": "message"}], previous_response_id="resp_x")
    assert codex_ws_router.parse_response_create(flat) == {
        "model": "gpt-5.5-codex",
        "input": [{"type": "message"}],
        "previous_response_id": "resp_x",
    }
    nested = json.dumps({"type": "response.create", "response": {"model": "m", "input": []}})
    assert codex_ws_router.parse_response_create(nested) == {"model": "m", "input": []}
    assert codex_ws_router.parse_response_create(json.dumps({"type": "response.cancel"})) is None
    assert codex_ws_router.parse_response_create("{not json") is None


def test_decide_ws_turn_premium_when_no_pending_tools() -> None:
    decision = codex_ws_router.decide_ws_turn([], routes=_routes(), group="codex")

    assert decision.routable is False
    assert decision.tier == "premium"


def test_decide_ws_turn_routable_resolves_tier_and_credential_from_history() -> None:
    # The delta carries only function_call_output; the name comes from the
    # function_call recorded in history, so the decision must run on full input.
    full_input: JsonArray = [_function_call("c1", "exec_command"), _call_output("c1")]

    decision = codex_ws_router.decide_ws_turn(full_input, routes=_routes(), group="codex")

    assert decision.routable is True
    assert decision.tier == "fast"
    assert decision.tier_cfg is not None
    assert decision.credential is not None
    assert decision.names == ["exec_command"]


def test_decide_ws_turn_premium_tool_is_not_routable() -> None:
    full_input: JsonArray = [_function_call("c1", "apply_patch"), _call_output("c1")]

    decision = codex_ws_router.decide_ws_turn(full_input, routes=_routes(), group="codex")

    assert decision.routable is False


class _NullRecorder:
    def note_request(self, raw_msg: str) -> None: ...
    def note_response(self, raw_msg: str) -> None: ...


@pytest.mark.asyncio
async def test_first_turn_passes_through_then_routes_with_paired_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Turn 1 has no pending tools -> passthrough; its streamed function_call is
    # observed and recorded under the upstream's real id. Turn 2 answers that
    # call -> routable; the reconstructed input handed to the HTTP call must
    # contain BOTH the function_call and its output (the original 400 regression).
    bodies, outcomes = _install_mocks(monkeypatch, [_routed_response("resp_routed_1", "done")])
    client = _FakeClient(
        [
            _create_frame([{"type": "message", "role": "user", "content": "go"}]),
            _create_frame([_call_output("c1")], previous_response_id="resp_up_1"),
        ]
    )
    upstream = _FakeUpstream(
        [_item_done(_function_call("c1", "exec_command")), _completed("resp_up_1")]
    )

    await codex_ws_router.run_hybrid_relay(
        client,
        upstream,
        recorder=_NullRecorder(),
        chatgpt_headers=[("Authorization", "Bearer t")],
        routes_loader=_routes,
    )

    # Only turn 1 reached the upstream; turn 2 was routed over HTTP.
    assert upstream.sent == [_create_frame([{"type": "message", "role": "user", "content": "go"}])]
    assert len(bodies) == 1
    routed_input = bodies[0]["input"]
    assert _function_call("c1", "exec_command") in routed_input
    assert _call_output("c1") in routed_input
    # Client saw passthrough frames for turn 1 and synthesized frames for turn 2.
    types = [json.loads(text)["type"] for text in client.sent_text]
    assert types[:2] == ["response.output_item.done", "response.completed"]
    assert types[2] == "response.created"
    assert types[-1] == "response.completed"
    assert len(outcomes) == 1


@pytest.mark.asyncio
async def test_routed_turn_synthesizes_router_id_for_next_chaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The id in the synthesized completed frame is what the client echoes next.
    # The router issues its OWN virtual id (not the cheap provider's id, which the
    # upstream also does not know), so the next turn chaining from it is recognized
    # as router-served and never forwarded upstream.
    _install_mocks(monkeypatch, [_routed_response("resp_routed_1", "ok")])
    client = _FakeClient([_create_frame(_exec_result_turn())])
    upstream = _FakeUpstream([])

    await codex_ws_router.run_hybrid_relay(
        client,
        upstream,
        recorder=_NullRecorder(),
        chatgpt_headers=None,
        routes_loader=_routes,
    )

    completed = json.loads(client.sent_text[-1])
    assert completed["type"] == "response.completed"
    synth_id = completed["response"]["id"]
    assert synth_id.startswith("resp_acr_")
    assert synth_id != "resp_routed_1"
    assert upstream.sent == []


@pytest.mark.asyncio
async def test_sticky_route_serves_later_premium_turn_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Once routed, the session cannot return to the upstream WS. A later turn
    # with no routable tools that chains from a router-issued (virtual) id is
    # still served over HTTP (premium fallback), never forwarded upstream.
    # Stickiness is driven by the virtual-id chain, not a per-connection flag.
    bodies, _ = _install_mocks(
        monkeypatch,
        [_routed_response("resp_routed_1", "a"), _routed_response("resp_routed_2", "b")],
    )
    client = _FakeClient(
        [
            _create_frame(_exec_result_turn()),
            _create_frame(
                [{"type": "message", "role": "user", "content": "more"}],
                previous_response_id="resp_acr_prior",
            ),
        ]
    )
    upstream = _FakeUpstream([])

    await codex_ws_router.run_hybrid_relay(
        client,
        upstream,
        recorder=_NullRecorder(),
        chatgpt_headers=None,
        routes_loader=_routes,
    )

    assert upstream.sent == []
    assert len(bodies) == 2


@pytest.mark.asyncio
async def test_cheap_failure_falls_back_to_premium_http(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cheap tier returns None; the session is committed to HTTP, so the turn
    # is retried against the client's requested model rather than dropped.
    bodies, outcomes = _install_mocks(monkeypatch, [None, _routed_response("resp_p", "p")])
    client = _FakeClient([_create_frame(_exec_result_turn())])
    upstream = _FakeUpstream([])

    await codex_ws_router.run_hybrid_relay(
        client, upstream, recorder=_NullRecorder(), chatgpt_headers=None, routes_loader=_routes
    )

    assert len(bodies) == 2
    assert json.loads(client.sent_text[-1])["type"] == "response.completed"
    assert len(outcomes) == 1


@pytest.mark.asyncio
async def test_control_frame_is_forwarded_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mocks(monkeypatch, [])
    cancel = json.dumps({"type": "response.cancel"})
    client = _FakeClient([cancel])
    upstream = _FakeUpstream([])

    await codex_ws_router.run_hybrid_relay(
        client, upstream, recorder=_NullRecorder(), chatgpt_headers=None, routes_loader=_routes
    )

    assert upstream.sent == [cancel]


@pytest.mark.asyncio
async def test_passthrough_forwards_binary_upstream_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mocks(monkeypatch, [])
    client = _FakeClient([_create_frame([{"type": "message", "role": "user", "content": "hi"}])])
    upstream = _FakeUpstream([b"\x00\x01", _completed("resp_up_b")])

    await codex_ws_router.run_hybrid_relay(
        client, upstream, recorder=_NullRecorder(), chatgpt_headers=None, routes_loader=_routes
    )

    assert client.sent_bytes == [b"\x00\x01"]


def test_decide_ws_turn_handles_routes_without_settings_block() -> None:
    routes: JsonObject = {
        "agents": {"codex": {"tools": {"exec_command": "fast"}}},
        "tiers": {"fast": {"model": "codex/gpt-5.4-mini", "provider": "codex", "key_env": "oauth"}},
    }

    decision = codex_ws_router.decide_ws_turn(_exec_result_turn(), routes=routes, group="codex")

    assert decision.routable is True


def test_response_id_reads_envelope_then_top_level_then_none() -> None:
    assert codex_ws_router._response_id({"response": {"id": "resp_a"}}) == "resp_a"
    assert codex_ws_router._response_id({"id": "resp_b"}) == "resp_b"
    assert codex_ws_router._response_id({"response": "not-a-dict", "id": ""}) is None
    assert codex_ws_router._response_id({}) is None


@pytest.mark.asyncio
async def test_routed_turn_raises_when_dispatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both cheap and premium HTTP attempts fail; a sticky session cannot fall
    # back to the upstream, so the relay raises to close the socket.
    _install_mocks(monkeypatch, [None, None])
    client = _FakeClient([_create_frame(_exec_result_turn())])
    upstream = _FakeUpstream([])

    with pytest.raises(codex_ws_router.RoutedTurnError):
        await codex_ws_router.run_hybrid_relay(
            client,
            upstream,
            recorder=_NullRecorder(),
            chatgpt_headers=None,
            routes_loader=_routes,
        )


_ACCOUNT_HEADERS: list[tuple[str, str]] = [("ChatGPT-Account-ID", "acct-123")]


@pytest.mark.asyncio
async def test_routable_turn_reconstructs_history_from_prior_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Codex pools WebSocket connections: a decision turn streaming a function_call
    # can arrive on connection 1, and the turn answering it on connection 2. The
    # session is keyed by account, so connection 2 must reconstruct the FULL input
    # (function_call from connection 1 + its output) -- otherwise the stateless
    # backend 400s with "No tool output found" (the regression, across the pool).
    bodies, _ = _install_mocks(monkeypatch, [_routed_response("resp_routed_1", "done")])

    client1 = _FakeClient([_create_frame([{"type": "message", "role": "user", "content": "go"}])])
    upstream1 = _FakeUpstream(
        [_item_done(_function_call("c1", "exec_command")), _completed("resp_up_1")]
    )
    await codex_ws_router.run_hybrid_relay(
        client1,
        upstream1,
        recorder=_NullRecorder(),
        chatgpt_headers=_ACCOUNT_HEADERS,
        routes_loader=_routes,
    )

    client2 = _FakeClient([_create_frame([_call_output("c1")], previous_response_id="resp_up_1")])
    upstream2 = _FakeUpstream([])
    await codex_ws_router.run_hybrid_relay(
        client2,
        upstream2,
        recorder=_NullRecorder(),
        chatgpt_headers=_ACCOUNT_HEADERS,
        routes_loader=_routes,
    )

    # Connection 1 passed through; connection 2 routed over HTTP with paired input.
    assert upstream2.sent == []
    assert len(bodies) == 1
    routed_input = bodies[0]["input"]
    assert _function_call("c1", "exec_command") in routed_input
    assert _call_output("c1") in routed_input


@pytest.mark.asyncio
async def test_virtual_chain_turn_never_forwarded_upstream_across_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Connection 1 routes a turn and hands the client a router-issued (virtual) id.
    # A later decision turn on connection 2 chains from that id. The upstream does
    # not know virtual ids, so it must be served over HTTP and NEVER forwarded --
    # forwarding it is what made Codex hang ("Unsupported parameter:
    # previous_response_id") under the per-connection store.
    bodies, _ = _install_mocks(
        monkeypatch,
        [_routed_response("resp_routed_1", "a"), _routed_response("resp_routed_2", "b")],
    )

    client1 = _FakeClient([_create_frame(_exec_result_turn())])
    upstream1 = _FakeUpstream([])
    await codex_ws_router.run_hybrid_relay(
        client1,
        upstream1,
        recorder=_NullRecorder(),
        chatgpt_headers=_ACCOUNT_HEADERS,
        routes_loader=_routes,
    )
    virtual_id = json.loads(client1.sent_text[-1])["response"]["id"]
    assert virtual_id.startswith("resp_acr_")

    client2 = _FakeClient(
        [
            _create_frame(
                [{"type": "message", "role": "user", "content": "more"}],
                previous_response_id=virtual_id,
            )
        ]
    )
    upstream2 = _FakeUpstream([])
    await codex_ws_router.run_hybrid_relay(
        client2,
        upstream2,
        recorder=_NullRecorder(),
        chatgpt_headers=_ACCOUNT_HEADERS,
        routes_loader=_routes,
    )

    assert upstream2.sent == []
    assert len(bodies) == 2


@pytest.mark.asyncio
async def test_missing_account_id_isolates_sessions_per_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without an account id the session cannot be shared safely, so each relay
    # gets an ephemeral session and the module store stays empty.
    _install_mocks(monkeypatch, [_routed_response("resp_routed_1", "a")])
    client = _FakeClient([_create_frame(_exec_result_turn())])
    upstream = _FakeUpstream([])

    await codex_ws_router.run_hybrid_relay(
        client,
        upstream,
        recorder=_NullRecorder(),
        chatgpt_headers=[("Authorization", "Bearer t")],
        routes_loader=_routes,
    )

    assert codex_ws_router._SESSIONS == {}
