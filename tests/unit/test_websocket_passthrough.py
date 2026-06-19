"""Tests for Codex ChatGPT-auth WebSocket passthrough."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, cast

import pytest
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from ai_calls_router.accounting import metrics
from ai_calls_router.proxy import server, websocket_passthrough

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


def test_usage_from_response_reports_cache_miss_separately_from_cache_read() -> None:
    usage = {
        "input_tokens": 11702,
        "output_tokens": 165,
        "input_tokens_details": {"cached_tokens": 2432},
    }

    cache_miss, output, cache_read, cache_create = websocket_passthrough._usage_from_response(
        {"usage": usage}
    )

    assert cache_miss == 9270
    assert output == 165
    assert cache_read == 2432
    assert cache_create == 0


def test_usage_from_response_returns_zeros_when_usage_absent() -> None:
    assert websocket_passthrough._usage_from_response({}) == (0, 0, 0, 0)
    assert websocket_passthrough._usage_from_response({"usage": None}) == (0, 0, 0, 0)


def test_usage_from_response_clamps_cache_read_above_total() -> None:
    usage = {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 9}}

    cache_miss, _output, cache_read, _create = websocket_passthrough._usage_from_response(
        {"usage": usage}
    )

    assert cache_miss == 0
    assert cache_read == 9


@pytest.mark.parametrize("value", [True, False, -3, "7", None, 1.5])
def test_non_negative_int_rejects_non_int_and_negative(value: object) -> None:
    assert websocket_passthrough._non_negative_int(cast("object", value)) == 0


def test_non_negative_int_passes_through_positive_int() -> None:
    assert websocket_passthrough._non_negative_int(42) == 42


def test_tool_names_from_items_dedupes_and_keeps_order() -> None:
    items = [
        {"type": "function_call", "name": "exec_command", "call_id": "a"},
        {"type": "reasoning", "name": "ignored"},
        {"type": "custom_tool_call", "name": "search_graph", "call_id": "b"},
        {"type": "function_call", "name": "exec_command", "call_id": "c"},
        {"type": "function_call", "call_id": "no-name"},
    ]

    assert websocket_passthrough._tool_names_from_items(items) == ["exec_command", "search_graph"]


def test_recorder_ignores_non_terminal_frames() -> None:
    m = metrics.Metrics()
    recorder = websocket_passthrough._UsageRecorder("codex-tui/0.141.0")
    recorder._record = lambda *a, **k: pytest.fail(  # type: ignore[method-assign]
        "non-terminal frame must not record"
    )

    recorder.note_response('{"type":"response.output_item.done"}')
    recorder.note_response("not-json")
    recorder.note_response("[]")

    assert m.snapshot()["requests"]["total"] == 0


def test_passthrough_records_real_usage_from_completed_frame(monkeypatch) -> None:
    m = metrics.Metrics()
    completed = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "model": "gpt-5.4-mini",
                "usage": {
                    "input_tokens": 11702,
                    "output_tokens": 165,
                    "input_tokens_details": {"cached_tokens": 2432},
                },
                "output": [
                    {
                        "type": "function_call",
                        "name": "exec_command",
                        "call_id": "call_1",
                        "arguments": "{}",
                    }
                ],
            },
        }
    )
    upstream = _FakeUpstream(completed)

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        return _FakeConnect(upstream)

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    monkeypatch.setattr(websocket_passthrough.metrics, "get_metrics", lambda: m)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={"chatgpt-account-id": "acct_123", "user-agent": "codex-tui/0.141.0"},
        ) as websocket,
    ):
        websocket.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"model": "gpt-5.4-mini", "input": "hello"},
                }
            )
        )
        assert json.loads(websocket.receive_text())["type"] == "response.completed"

    assert upstream.sent == [
        json.dumps(
            {"type": "response.create", "response": {"model": "gpt-5.4-mini", "input": "hello"}}
        )
    ]
    snapshot = m.snapshot()
    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["passthrough"] == 1
    assert snapshot["requests"]["routed"] == 0
    latest = snapshot["last_requests"][0]
    assert latest["method"] == "WS"
    assert latest["path"] == "/v1/responses"
    assert latest["status"] == 200
    assert latest["route"] == "passthrough"
    assert latest["model"] == "gpt-5.4-mini"
    assert latest["tool_names"] == ["exec_command"]
    assert latest["input_tokens"] == 9270
    assert latest["cache_read_tokens"] == 2432
    assert latest["output_tokens"] == 165
    assert latest["agent"] == metrics.identify_agent("codex-tui/0.141.0")
    assert latest["decision_reason"] == "codex_passthrough"


def test_passthrough_records_failed_response_as_error_status(monkeypatch) -> None:
    m = metrics.Metrics()
    upstream = _FakeUpstream(
        json.dumps({"type": "response.failed", "response": {"model": "gpt-5.4-mini"}})
    )

    def _connect(uri: str, **kwargs: object) -> _FakeConnect:
        return _FakeConnect(upstream)

    monkeypatch.setattr(websocket_passthrough, "connect", _connect)
    monkeypatch.setattr(websocket_passthrough.metrics, "get_metrics", lambda: m)
    app = Starlette(routes=[WebSocketRoute("/v1/responses", server.responses_ws)])

    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/responses",
            headers={"chatgpt-account-id": "acct_123", "user-agent": "codex-tui/0.141.0"},
        ) as websocket,
    ):
        websocket.send_text(
            json.dumps({"type": "response.create", "response": {"model": "gpt-5.4-mini"}})
        )
        assert json.loads(websocket.receive_text())["type"] == "response.failed"

    latest = m.snapshot()["last_requests"][0]
    assert latest["status"] == 400
    assert latest["route"] == "passthrough"


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
        recorder=websocket_passthrough._UsageRecorder(""),
    )

    assert upstream.closed


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
