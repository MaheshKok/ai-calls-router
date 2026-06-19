"""Spec-derived tests for ai_calls_router.server.

Contract under test: the Starlette app answers GET /health locally, decides
POST /v1/messages per request -- tool-result turns mapped to a configured
cheap tier with a resolvable key go to routed_call, everything else (turn
openers, premium or unmapped tools, missing keys, malformed JSON, routing
errors, routed_call returning None) streams through to the premium upstream
with the client's headers intact -- and proxies every other path and method
to the upstream unchanged (invariant 3: routing never breaks a turn).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from ai_calls_router._lib import config
from ai_calls_router._lib.conversion import BackendResponse
from ai_calls_router._lib.types import JsonObject
from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.proxy import server as server_mod
from ai_calls_router.proxy.server import create_app
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as rc
from ai_calls_router.routing import synthesis

CONFIG_YAML = """
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
  premium_tools: [Edit]
tiers:
  fast:
    model: deepseek/test-model
    key_env: ACR_TEST_KEY
    max_tokens: 1000
tools:
  Bash: fast
  Edit: premium
"""

ROUTED_BODY: JsonObject = {
    "id": "msg_routed",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "routed answer"}],
    "model": "claude-fable-5",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def test_route_table_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route registration order and handlers must survive server splits."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    routes = [
        (
            route.path,
            sorted(route.methods) if hasattr(route, "methods") and route.methods else [],
            route.endpoint.__name__,
        )
        for route in create_app().routes
    ]

    assert routes == [
        ("/health", ["GET", "HEAD"], "health"),
        ("/metrics", ["GET", "HEAD"], "metrics_endpoint"),
        ("/dashboard", ["GET", "HEAD"], "dashboard"),
        ("/v1/messages", ["POST"], "messages"),
        ("/v1/chat/completions", ["POST"], "chat_completions"),
        ("/v1/responses", ["POST"], "responses"),
        (
            "/{path:path}",
            ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
            "proxy",
        ),
    ]


class _Upstream:
    """Mock premium upstream recording every proxied request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status_code = 200
        self.content = b'{"marker": "upstream"}'
        self.headers = {"content-type": "application/json"}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status_code,
            content=self.content,
            headers=self.headers,
        )


class _FakeRoutedCall:
    """routed_call stand-in recording invocations."""

    def __init__(
        self, result: BackendResponse | None, *, guard_tools: list[str] | None = None
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._guard_tools = guard_tools or []

    async def __call__(self, **kwargs: object) -> BackendResponse | None:
        self.calls.append(kwargs)
        if self._guard_tools:
            guard = kwargs.get("on_premium_guard")
            if callable(guard):
                guard(self._guard_tools)
        return self._result


@pytest.fixture
def upstream() -> _Upstream:
    return _Upstream()


def test_public_bind_host_warns_for_unauthenticated_dashboard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-loopback bind stays allowed but visible in startup logs."""
    caplog.set_level("WARNING", logger="acr.server")
    server_mod._warn_if_public_bind(
        config.ServerSettings(
            host="0.0.0.0",
            port=8747,
            upstream="https://api.anthropic.com",
        )
    )
    assert "not loopback" in caplog.text
    assert "/metrics and /dashboard are unauthenticated" in caplog.text


def test_loopback_bind_host_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Loopback-only binds are the expected default for local telemetry."""
    caplog.set_level("WARNING", logger="acr.server")
    server_mod._warn_if_public_bind(
        config.ServerSettings(
            host="127.0.0.1",
            port=8747,
            upstream="https://api.anthropic.com",
        )
    )
    assert not caplog.records


def test_loopback_host_classifier_handles_named_and_public_hosts() -> None:
    """Named loopback is safe; arbitrary hostnames are treated as public."""
    assert server_mod._is_loopback_host("localhost")
    assert server_mod._is_loopback_host("::1")
    assert not server_mod._is_loopback_host("public.example")


@pytest.fixture
def client(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: _Upstream) -> TestClient:
    metrics_mod._metrics_singleton = None
    config_file = tmp_path / "config.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    with TestClient(app) as test_client:
        yield test_client
    metrics_mod._metrics_singleton = None


def _tool_result_body(tool_name: str = "Bash", stream: bool = False) -> JsonObject:
    """Build a request body processing one pending tool result."""
    body: JsonObject = {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": tool_name, "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "output"}],
            },
        ],
    }
    if stream:
        body["stream"] = True
    return body


def _opener_body() -> JsonObject:
    """Build a turn-opener request body (no pending tool results)."""
    return {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hello"}],
    }


class TestHealth:
    def test_health_answers_locally(self, client: TestClient, upstream: _Upstream) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert upstream.requests == []


def test_dashboard_shows_date_time_column(client: TestClient) -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "<th>date/time</th>" in response.text
    assert "function requestTime(ts)" in response.text


class TestMessagesPassthrough:
    def test_opener_passes_through_to_upstream(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        response = client.post("/v1/messages", json=_opener_body())
        assert response.json() == {"marker": "upstream"}
        assert upstream.requests[0].url.path == "/v1/messages"
        assert json.loads(upstream.requests[0].content) == _opener_body()

    def test_client_auth_headers_reach_upstream(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        client.post(
            "/v1/messages",
            json=_opener_body(),
            headers={"authorization": "Bearer oauth", "anthropic-version": "2023-06-01"},
        )
        request = upstream.requests[0]
        assert request.headers["authorization"] == "Bearer oauth"
        assert request.headers["anthropic-version"] == "2023-06-01"

    def test_premium_tool_result_passes_through(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        response = client.post("/v1/messages", json=_tool_result_body("Edit"))
        assert response.json() == {"marker": "upstream"}
        assert len(upstream.requests) == 1
        latest = client.get("/metrics").json()["last_requests"][0]
        assert latest["route"] == "premium_guard"
        assert latest["tier"] == "premium"
        assert latest["model"] == "claude-fable-5"
        assert latest["premium_model"] == "claude-fable-5"
        assert latest["tool_names"] == ["Edit"]
        assert latest["decision_reason"] == "request_premium_guard"

    def test_unmapped_tool_result_passes_through(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        response = client.post("/v1/messages", json=_tool_result_body("Mystery"))
        assert response.json() == {"marker": "upstream"}
        assert len(upstream.requests) == 1

    def test_invalid_json_passes_through_verbatim(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        response = client.post(
            "/v1/messages",
            content=b"not json at all",
            headers={"content-type": "application/json"},
        )
        assert response.json() == {"marker": "upstream"}
        assert upstream.requests[0].content == b"not json at all"

    def test_passthrough_usage_updates_recent_request(
        self, client: TestClient, upstream: _Upstream
    ) -> None:
        upstream.headers = {"content-type": "text/event-stream"}
        upstream.content = (
            b"event: message_start\n"
            b'data: {"message": {"usage": {"input_tokens": 321, '
            b'"cache_read_input_tokens": 100}}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage": {"output_tokens": 9}}\n\n'
        )
        response = client.post("/v1/messages", json=_opener_body())
        assert response.content == upstream.content

        snapshot = client.get("/metrics").json()
        assert snapshot["premium_tokens"]["input"] == 321
        assert snapshot["premium_tokens"]["output"] == 9
        assert snapshot["premium_tokens"]["cache_read"] == 100
        latest = snapshot["last_requests"][0]
        assert latest["route"] == "passthrough"
        assert latest["decision_reason"] == "no_pending_tools"
        assert latest["status"] == 200
        assert latest["input_tokens"] == 321
        assert latest["output_tokens"] == 9
        assert latest["cache_read_tokens"] == 100

    def test_missing_api_key_passes_through_without_routing(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ACR_TEST_KEY")
        fake = _FakeRoutedCall(BackendResponse(body=ROUTED_BODY))
        monkeypatch.setattr(rc, "routed_call", fake)
        response = client.post("/v1/messages", json=_tool_result_body())
        assert response.json() == {"marker": "upstream"}
        assert fake.calls == []

    def test_routing_decision_error_passes_through(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(body: JsonObject) -> list[str]:
            raise RuntimeError("decision exploded")

        monkeypatch.setattr(routing, "pending_tool_names", _boom)
        response = client.post("/v1/messages", json=_tool_result_body())
        assert response.json() == {"marker": "upstream"}
        assert len(upstream.requests) == 1
        latest = client.get("/metrics").json()["last_requests"][0]
        assert latest["decision_reason"] == "routing_error"
        assert latest["model"] == "claude-fable-5"
        assert latest["premium_model"] == "claude-fable-5"


class TestMessagesRouted:
    def test_cheap_tool_result_served_by_routed_call(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(BackendResponse(body=ROUTED_BODY))
        monkeypatch.setattr(rc, "routed_call", fake)
        response = client.post("/v1/messages", json=_tool_result_body())
        assert response.status_code == 200
        assert response.json() == ROUTED_BODY
        assert upstream.requests == []

    def test_system_role_message_tool_result_routes_after_normalization(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(BackendResponse(body=ROUTED_BODY))
        monkeypatch.setattr(rc, "routed_call", fake)
        body = _tool_result_body()
        messages = body["messages"]
        assert isinstance(messages, list)
        messages.insert(0, {"role": "system", "content": "runtime"})

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert response.json() == ROUTED_BODY
        assert upstream.requests == []
        assert len(fake.calls) == 1
        routed_body = fake.calls[0]["body"]
        assert isinstance(routed_body, dict)
        assert routed_body["system"] == "runtime"
        assert routed_body["messages"] == _tool_result_body()["messages"]

    def test_routed_call_receives_tier_and_key(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(BackendResponse(body=ROUTED_BODY))
        monkeypatch.setattr(rc, "routed_call", fake)
        client.post("/v1/messages", json=_tool_result_body())
        call = fake.calls[0]
        assert call["tier_name"] == "fast"
        assert call["tier_cfg"]["model"] == "deepseek/test-model"
        assert call["api_key"] == "tier-key"
        assert call["premium_tools"] == ["Edit"]
        assert call["body"] == _tool_result_body()

    def test_streaming_request_served_as_sse(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(BackendResponse(body=ROUTED_BODY))
        monkeypatch.setattr(rc, "routed_call", fake)
        response = client.post("/v1/messages", json=_tool_result_body(stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.content == synthesis.synthesize_sse(ROUTED_BODY)
        assert "event: message_start" in response.text
        assert "event: message_stop" in response.text
        assert "routed answer" in response.text
        assert upstream.requests == []

    def test_routed_call_none_falls_back_to_passthrough(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(None)
        monkeypatch.setattr(rc, "routed_call", fake)
        response = client.post("/v1/messages", json=_tool_result_body())
        assert response.json() == {"marker": "upstream"}
        assert len(fake.calls) == 1
        assert len(upstream.requests) == 1

    def test_routed_premium_guard_records_escalation_reason(
        self,
        *,
        client: TestClient,
        upstream: _Upstream,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeRoutedCall(None, guard_tools=["Edit"])
        monkeypatch.setattr(rc, "routed_call", fake)
        response = client.post("/v1/messages", json=_tool_result_body())
        assert response.json() == {"marker": "upstream"}
        assert len(upstream.requests) == 1

        snapshot = client.get("/metrics").json()
        assert snapshot["requests"]["escalated"] == 1
        latest = snapshot["last_requests"][0]
        assert latest["route"] == "premium_guard"
        assert latest["tier"] == "fast"
        assert latest["model"] == "claude-fable-5"
        assert latest["tool_names"] == ["Bash"]
        assert latest["decision_reason"] == "response_premium_guard"


class TestCatchAll:
    def test_other_post_paths_proxied(self, client: TestClient, upstream: _Upstream) -> None:
        response = client.post("/v1/messages/count_tokens", json={"model": "m"})
        assert response.json() == {"marker": "upstream"}
        assert upstream.requests[0].url.path == "/v1/messages/count_tokens"

    def test_get_requests_proxied(self, client: TestClient, upstream: _Upstream) -> None:
        response = client.get("/v1/models")
        assert response.json() == {"marker": "upstream"}
        request = upstream.requests[0]
        assert request.method == "GET"
        assert request.url.path == "/v1/models"

    def test_query_string_preserved(self, client: TestClient, upstream: _Upstream) -> None:
        client.get("/v1/models?limit=5")
        assert upstream.requests[0].url.query == b"limit=5"
