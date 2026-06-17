"""App-level tests for OpenAI Chat Completions routing.

These tests drive Hermes Chat traffic through the real server decision path and
stub only the native Anthropic provider call. They prove routed, streaming,
premium-guard, and malformed-body behavior without reimplementing the core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
  hermes:
    upstream: https://api.anthropic.com
    premium:
      provider: anthropic
    premium_tools: [patch]
    tools:
      terminal: fast
      patch: premium
"""

ROUTED_TEXT_BODY: dict[str, Any] = {
    "id": "msg_chat",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "routed answer"}],
    "model": "deepseek-v4-pro",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

ROUTED_PATCH_BODY: dict[str, Any] = {
    **ROUTED_TEXT_BODY,
    "content": [{"type": "tool_use", "id": "call_patch", "name": "patch", "input": {}}],
    "stop_reason": "tool_use",
}


class _Upstream:
    """Mock premium upstream recording every proxied request."""

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

    def __init__(self, response: dict[str, Any] | None) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        body: dict[str, Any],
        tier_cfg: dict[str, Any],
        api_key: str,
    ) -> dict[str, Any] | None:
        """Record the direct call and return the canned response."""
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
    metrics_mod._METRICS = None
    config_file = tmp_path / "config.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    with TestClient(app) as test_client:
        yield test_client
    metrics_mod._METRICS = None


def _chat_tool_result_body(*, stream: bool = False) -> dict[str, Any]:
    """Build a Chat request processing one Hermes terminal result."""
    body: dict[str, Any] = {
        "model": "openai/gpt-premium",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_terminal",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_terminal", "content": "file.txt"},
        ],
    }
    if stream:
        body["stream"] = True
    return body


def test_chat_tool_result_routes_to_deepseek_direct(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    response = client.post("/v1/chat/completions", json=_chat_tool_result_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "openai/gpt-premium"
    assert payload["choices"][0]["message"]["content"] == "routed answer"
    assert upstream.requests == []
    assert fake.calls[0]["api_key"] == "tier-key"
    assert fake.calls[0]["body"]["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_terminal",
        "content": "file.txt",
    }

    latest = client.get("/metrics").json()["last_requests"][0]
    assert latest["path"] == "/v1/chat/completions"
    assert latest["route"] == "direct"


def test_chat_streaming_request_returns_chat_sse(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    response = client.post("/v1/chat/completions", json=_chat_tool_result_body(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b"chat.completion.chunk" in response.content
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert upstream.requests == []


def test_chat_response_premium_tool_escalates_to_passthrough(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_PATCH_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    response = client.post("/v1/chat/completions", json=_chat_tool_result_body())

    assert response.json() == {"marker": "upstream"}
    assert len(upstream.requests) == 1
    assert upstream.requests[0].url.path == "/v1/chat/completions"
    assert json.loads(upstream.requests[0].content) == _chat_tool_result_body()

    latest = client.get("/metrics").json()["last_requests"][0]
    assert latest["path"] == "/v1/chat/completions"
    assert latest["decision_reason"] == "response_premium_guard"


def test_malformed_chat_body_passes_through(
    *,
    client: TestClient,
    upstream: _Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDirectCall(ROUTED_TEXT_BODY)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake)
    body = {"model": "openai/gpt-premium", "messages": "not-a-list"}
    response = client.post("/v1/chat/completions", json=body)

    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.path == "/v1/chat/completions"
    assert json.loads(upstream.requests[0].content) == body
    assert fake.calls == []
