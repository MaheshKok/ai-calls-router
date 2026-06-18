"""Tests for Codex ChatGPT-auth WebSocket passthrough."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import pytest
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from ai_calls_router._lib.types import JsonObject
from ai_calls_router.accounting import metrics
from ai_calls_router.proxy import server, websocket_passthrough
from ai_calls_router.routing.adapters.base import ClientAdapter

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


class _FakeUpstream:
    def __init__(self, response: str | bytes = '{"type":"response.completed"}') -> None:
        self.sent: list[str] = []
        self.response = response
        self.queue: asyncio.Queue[str | bytes | None] = asyncio.Queue()

    def __aiter__(self) -> _FakeUpstream:
        return self

    async def __anext__(self) -> str | bytes:
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, message: str) -> None:
        self.sent.append(message)
        await self.queue.put(self.response)

    async def close(self) -> None:
        await self.queue.put(None)


class _FakeConnect:
    def __init__(self, upstream: _FakeUpstream) -> None:
        self.upstream = upstream

    async def __aenter__(self) -> _FakeUpstream:
        return self.upstream

    async def __aexit__(self, *exc: object) -> None:
        await self.upstream.close()


class _SendOnlyWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, message: str) -> None:
        self.sent.append(message)


class _HangingWebSocket:
    async def iter_text(self) -> object:
        while True:
            await asyncio.sleep(60)
            yield "never"


class _ClosableUpstream(_FakeUpstream):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


def test_response_create_to_http_body_unwraps_envelope_without_mutating() -> None:
    frame = {
        "type": "response.create",
        "response": {
            "model": "gpt-5",
            "input": "hello",
            "type": "response",
        },
    }

    body = websocket_passthrough.response_create_to_http_body(json.dumps(frame))

    assert body == {"model": "gpt-5", "input": "hello", "stream": True}
    assert frame["response"]["type"] == "response"


def test_response_create_to_http_body_handles_flat_shape() -> None:
    body = websocket_passthrough.response_create_to_http_body(
        json.dumps({"type": "response.create", "model": "gpt-5", "input": "hello"})
    )

    assert body == {"model": "gpt-5", "input": "hello", "stream": True}


@pytest.mark.parametrize("raw_msg", ["not-json", '{"type":"ping"}', "[]"])
def test_response_create_to_http_body_ignores_non_create_frames(raw_msg: str) -> None:
    assert websocket_passthrough.response_create_to_http_body(raw_msg) is None


def test_codex_websocket_routed_response_short_circuits_upstream(monkeypatch) -> None:
    captures: list[JsonObject] = []
    m = metrics.Metrics()

    async def _fake_try_route(
        body_bytes: bytes,
        *,
        adapter: ClientAdapter,
        group: str,
        request_path: str,
        request_headers: Mapping[str, str],
        user_agent: str = "",
        agent: str = "",
        session: str | None = None,
    ) -> server._RouteAttempt:
        captures.append(json.loads(body_bytes))
        sse = b'event: response.completed\ndata: {"type":"response.completed"}\n\ndata: [DONE]\n\n'
        return server._RouteAttempt(
            response=Response(sse, media_type="text/event-stream"),
            tier="codex_fast",
            reason="routed",
            model="gpt-5",
            tool_names=["exec_command"],
            input_tokens=11,
            output_tokens=3,
            cache_read_tokens=7,
            duration=0.25,
        )

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        pytest.fail("routed response.create should not open upstream websocket")

    monkeypatch.setattr(server, "_load_assembled_routes", lambda: {})
    monkeypatch.setattr(server, "_try_route", _fake_try_route)
    monkeypatch.setattr(server.metrics, "get_metrics", lambda: m)
    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={
                "chatgpt-account-id": "acct_123",
                "user-agent": "codex_cli_rs/0.0.0",
            },
        ) as websocket,
    ):
        websocket.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "model": "gpt-5",
                        "input": "hello",
                    },
                }
            )
        )
        assert websocket.receive_text() == '{"type":"response.completed"}'

    assert captures == [{"model": "gpt-5", "input": "hello", "stream": True}]
    snapshot = m.snapshot()
    assert snapshot["requests"] == {
        "total": 1,
        "routed": 1,
        "passthrough": 0,
        "errors": 0,
        "escalated": 0,
        "fallback": 0,
    }
    latest = snapshot["last_requests"][0]
    assert latest["method"] == "WS"
    assert latest["path"] == "/v1/responses"
    assert latest["status"] == 101
    assert latest["route"] == "routed"
    assert latest["tier"] == "codex_fast"
    assert latest["model"] == "gpt-5"
    assert latest["tool_names"] == ["exec_command"]
    assert latest["input_tokens"] == 11
    assert latest["output_tokens"] == 3
    assert latest["cache_read_tokens"] == 7
    assert latest["duration_ms"] == 250


def test_codex_websocket_routes_later_output_with_cached_call(monkeypatch) -> None:
    upstream = _FakeUpstream(
        json.dumps(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_exec",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
            }
        )
    )
    captures: list[JsonObject] = []

    async def _fake_try_route(
        body_bytes: bytes,
        *,
        adapter: ClientAdapter,
        group: str,
        request_path: str,
        request_headers: Mapping[str, str],
        user_agent: str = "",
        agent: str = "",
        session: str | None = None,
    ) -> server._RouteAttempt:
        captures.append(json.loads(body_bytes))
        sse = b'event: response.completed\ndata: {"type":"response.completed"}\n\ndata: [DONE]\n\n'
        return server._RouteAttempt(response=Response(sse, media_type="text/event-stream"))

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        return _FakeConnect(upstream)

    monkeypatch.setattr(server, "_load_assembled_routes", lambda: {})
    monkeypatch.setattr(server, "_try_route", _fake_try_route)
    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={
                "chatgpt-account-id": "acct_123",
                "user-agent": "codex_cli_rs/0.0.0",
            },
        ) as websocket,
    ):
        websocket.send_text('{"type":"ping"}')
        assert json.loads(websocket.receive_text())["type"] == "response.output_item.done"
        websocket.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "model": "gpt-5",
                        "input": [
                            {
                                "type": "function_call_output",
                                "call_id": "call_exec",
                                "output": "/tmp/repo",
                            }
                        ],
                    },
                }
            )
        )
        assert websocket.receive_text() == '{"type":"response.completed"}'

    assert upstream.sent == ['{"type":"ping"}']
    assert captures == [
        {
            "model": "gpt-5",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_exec",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_exec",
                    "output": "/tmp/repo",
                },
            ],
            "stream": True,
        }
    ]


async def test_try_send_routed_response_declines_without_routed_messages() -> None:
    websocket = _SendOnlyWebSocket()
    call_items: dict[str, JsonObject] = {}

    async def _decline(raw_msg: str, headers: dict[str, str]) -> list[str] | None:
        return None

    assert not await websocket_passthrough._try_send_routed_response(
        cast("WebSocket", websocket), "{}", None, {}, call_items
    )
    assert not await websocket_passthrough._try_send_routed_response(
        cast("WebSocket", websocket), "{}", _decline, {}, call_items
    )
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_relay_awaits_cancelled_client_task_and_closes_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _finish_immediately(*args: object) -> None:
        del args

    upstream = _ClosableUpstream()
    monkeypatch.setattr(websocket_passthrough, "_upstream_to_client", _finish_immediately)

    await websocket_passthrough._relay_both(
        cast("WebSocket", _HangingWebSocket()),
        cast("ClientConnection", upstream),
        route_frame=None,
        headers={},
        call_items={},
    )

    assert upstream.closed


def test_cached_call_helpers_ignore_non_matching_shapes() -> None:
    call_items: dict[str, JsonObject] = {}
    websocket_passthrough._remember_call_items("not-json", call_items)
    websocket_passthrough._remember_call_items(
        json.dumps({"item": {"type": "function_call", "call_id": "missing-name"}}),
        call_items,
    )
    assert call_items == {}

    websocket_passthrough._remember_call_items(
        json.dumps(
            {
                "item": {
                    "type": "function_call",
                    "call_id": "call_exec",
                    "name": "exec_command",
                    "arguments": "{}",
                }
            }
        ),
        call_items,
    )

    assert websocket_passthrough._with_cached_call_items('{"type":"ping"}', call_items) == (
        '{"type":"ping"}'
    )
    assert websocket_passthrough._with_cached_call_items(
        json.dumps({"type": "response.create", "response": {"input": "hi"}}),
        call_items,
    ) == json.dumps({"type": "response.create", "response": {"input": "hi"}})

    existing = json.dumps(
        {
            "type": "response.create",
            "response": {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_exec",
                        "name": "exec_command",
                        "arguments": "{}",
                    },
                    {"type": "function_call_output", "call_id": "call_exec", "output": "ok"},
                ]
            },
        }
    )
    assert websocket_passthrough._with_cached_call_items(existing, call_items) == existing


async def test_ws_route_returns_none_when_routed_core_declines(monkeypatch) -> None:
    m = metrics.Metrics()

    async def _fake_try_route(
        body_bytes: bytes,
        *,
        adapter: ClientAdapter,
        group: str,
        request_path: str,
        request_headers: Mapping[str, str],
        user_agent: str = "",
        agent: str = "",
        session: str | None = None,
    ) -> server._RouteAttempt:
        return server._RouteAttempt(
            tier="codex_fast",
            reason="routed_fallback",
            model="gpt-5",
            tool_names=["exec_command"],
        )

    monkeypatch.setattr(server, "_load_assembled_routes", lambda: {})
    monkeypatch.setattr(server, "_try_route", _fake_try_route)
    monkeypatch.setattr(server.metrics, "get_metrics", lambda: m)

    result = await server._try_route_ws_response_create(
        json.dumps({"type": "response.create", "response": {"model": "gpt-5", "input": "hi"}}),
        {"chatgpt-account-id": "acct_123"},
    )

    assert result is None
    snapshot = m.snapshot()
    assert snapshot["requests"]["fallback"] == 1
    latest = snapshot["last_requests"][0]
    assert latest["method"] == "WS"
    assert latest["path"] == "/v1/responses"
    assert latest["status"] == 0
    assert latest["route"] == "passthrough"
    assert latest["decision_reason"] == "routed_fallback"


async def test_ws_route_errors_fail_open(monkeypatch) -> None:
    def _fail_load_routes() -> JsonObject:
        raise RuntimeError("broken config")

    monkeypatch.setattr(server, "_load_assembled_routes", _fail_load_routes)

    result = await server._try_route_ws_response_create(
        json.dumps({"type": "response.create", "response": {"model": "gpt-5", "input": "hi"}}),
        {"chatgpt-account-id": "acct_123"},
    )

    assert result is None


def test_codex_websocket_relays_chatgpt_oauth(monkeypatch) -> None:
    upstream = _FakeUpstream()
    captures: list[dict[str, object]] = []

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        captures.append({"uri": uri, **kwargs})
        return _FakeConnect(upstream)

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={
                "authorization": f"Bearer {_jwt('acct_123')}",
                "user-agent": "codex_cli_rs/0.0.0",
            },
        ) as websocket,
    ):
        websocket.send_text('{"type":"ping"}')
        assert websocket.receive_text() == '{"type":"response.completed"}'

    assert upstream.sent == ['{"type":"ping"}']
    assert captures[0]["uri"] == websocket_passthrough.CHATGPT_CODEX_RESPONSES_WS
    header_pairs = captures[0]["additional_headers"]
    assert isinstance(header_pairs, list)
    headers = {key.lower(): value for key, value in header_pairs}
    assert headers["authorization"].startswith("Bearer ")
    assert headers["chatgpt-account-id"] == "acct_123"
    assert headers["originator"] == "codex_cli_rs"
    assert captures[0]["user_agent_header"] == "codex_cli_rs/0.0.0"


def test_codex_websocket_alias_preserves_subpath_and_query(monkeypatch) -> None:
    upstream = _FakeUpstream()
    captures: list[dict[str, object]] = []

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        captures.append({"uri": uri, **kwargs})
        return _FakeConnect(upstream)

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(
        routes=[WebSocketRoute("/v1/codex/responses/{sub_path:path}", server.responses_ws_sub)]
    )

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/codex/responses/resp_123/events?stream=true",
            headers={"chatgpt-account-id": "acct_123"},
        ) as websocket,
    ):
        websocket.send_text('{"type":"ping"}')
        assert websocket.receive_text() == '{"type":"response.completed"}'

    assert captures[0]["uri"] == (
        f"{websocket_passthrough.CHATGPT_CODEX_RESPONSES_WS}/resp_123/events?stream=true"
    )


def test_codex_websocket_relays_binary_upstream_message(monkeypatch) -> None:
    upstream = _FakeUpstream(b"binary-ok")

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        return _FakeConnect(upstream)

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={"chatgpt-account-id": "acct_123"},
        ) as websocket,
    ):
        websocket.send_text('{"type":"ping"}')
        assert websocket.receive_bytes() == b"binary-ok"


def test_codex_websocket_upstream_failure_closes_with_1011(monkeypatch) -> None:
    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        raise OSError("dial failed")

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses", headers={"chatgpt-account-id": "acct_123"}
        ) as websocket,
    ):
        websocket.send_text('{"type":"ping"}')
        with pytest.raises(WebSocketDisconnect) as excinfo:
            websocket.receive_text()

    assert excinfo.value.code == 1011


def test_codex_header_detection_preserves_existing_chatgpt_headers() -> None:
    headers = websocket_passthrough.codex_chatgpt_headers(
        {
            "authorization": f"Bearer {_jwt('acct_from_token')}",
            "chatgpt-account-id": "acct_header",
            "originator": "custom_origin",
            "sec-websocket-key": "drop-me",
        }
    )

    assert headers is not None
    result = {key.lower(): value for key, value in headers}
    assert result["chatgpt-account-id"] == "acct_header"
    assert result["originator"] == "custom_origin"
    assert "sec-websocket-key" not in result


@pytest.mark.parametrize(
    "authorization",
    [
        "",
        "Bearer not-a-jwt",
        "Bearer header.@@@.sig",
        f"Bearer header.{base64.urlsafe_b64encode(b'[]').decode('ascii').rstrip('=')}.sig",
    ],
)
def test_invalid_bearer_token_is_not_chatgpt_auth(authorization: str) -> None:
    assert websocket_passthrough.codex_chatgpt_headers({"authorization": authorization}) is None


def test_codex_websocket_without_chatgpt_oauth_closes(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        websocket_passthrough,
        "connect",
        lambda *args, **kwargs: calls.append(True),
    )
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as excinfo,
        client.websocket_connect("/v1/responses"),
    ):
        pass

    assert excinfo.value.code == 1008
    assert calls == []


def _jwt(account_id: str) -> str:
    payload = json.dumps(
        {"https://api.openai.com/auth.chatgpt_account_id": account_id},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"header.{encoded}.signature"
