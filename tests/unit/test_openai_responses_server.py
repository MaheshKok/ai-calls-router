"""App-level tests for OpenAI Responses routing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.proxy.server import create_app
from ai_calls_router.routing import engine as rc

CONFIG_YAML = """
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: deepseek/deepseek-v4-pro
    key_env: ACR_TEST_KEY
    max_tokens: 1000
agents:
  codex:
    upstream: https://api.anthropic.com
    premium:
      provider: anthropic
    premium_tools: [apply_patch]
    tools:
      exec_command: fast
      apply_patch: premium
"""

ROUTED_TEXT_BODY: dict[str, object] = {
    "id": "msg_responses",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "routed answer"}],
    "model": "deepseek-v4-pro",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class _Upstream:
    """Mock premium upstream recording proxied requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record request and return a fixed marker."""
        self.requests.append(request)
        return httpx.Response(
            200,
            content=b'{"marker": "upstream"}',
            headers={"content-type": "application/json"},
        )


class _FakeDirectCall:
    """direct_call stand-in recording invocations."""

    def __init__(self, response: dict[str, object] | None) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        *,
        body: dict[str, object],
        tier_cfg: dict[str, object],
        api_key: str,
    ) -> dict[str, object] | None:
        """Record direct call input and return the canned response."""
        self.calls.append({"body": body, "tier_cfg": tier_cfg, "api_key": api_key})
        return self.response


@pytest.fixture
def upstream() -> _Upstream:
    return _Upstream()


@pytest.fixture
def client(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> TestClient:
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


def _responses_tool_result_body(*, stream: bool = True) -> dict[str, object]:
    """Build a Responses request processing one Codex exec result."""
    return {
        "model": "gpt-5-codex",
        "stream": stream,
        "input": [
            {
                "type": "function_call",
                "call_id": "call_exec",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd"}',
            },
            {"type": "reasoning", "encrypted_content": "must-not-route"},
            {"type": "function_call_output", "call_id": "call_exec", "output": "/tmp"},
        ],
    }


def test_responses_streaming_tool_result_routes_to_deepseek_direct(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    response = client.post(
        "/v1/responses",
        json=_responses_tool_result_body(),
        headers={"authorization": "Bearer client-premium-secret"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"event: response.created" in response.content
    assert b"event: response.completed" in response.content
    assert upstream.requests == []
    assert fake.calls[0]["api_key"] == "tier-key"
    assert "client-premium-secret" not in json.dumps(fake.calls[0])
    assert "must-not-route" not in json.dumps(fake.calls[0]["body"])
    assert fake.calls[0]["body"]["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_exec",
        "content": "/tmp",
    }

    latest = client.get("/metrics").json()["last_requests"][0]
    assert latest["path"] == "/v1/responses"
    assert latest["route"] == "direct"


def test_responses_non_streaming_returns_response_json(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    response = client.post("/v1/responses", json=_responses_tool_result_body(stream=False))

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["model"] == "gpt-5-codex"
    assert payload["output"][0]["content"][0]["text"] == "routed answer"
    assert upstream.requests == []


def test_malformed_responses_body_passes_through(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    body = {
        "model": "gpt-5-codex",
        "input": [
            {"type": "reasoning", "encrypted_content": "relay-me"},
            {"type": "function_call", "call_id": "bad"},
        ],
    }
    response = client.post("/v1/responses", json=body)

    assert response.status_code == 200
    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.path == "/v1/responses"
    assert json.loads(upstream.requests[0].content) == body
    assert fake.calls == []
