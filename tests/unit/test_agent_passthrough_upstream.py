"""Tests for per-agent passthrough upstream selection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from ai_calls_router.proxy import server as server_mod
from tests.acr_testkit import Upstream, make_client

CONFIG_YAML = """
server:
  upstream: https://premium.default.example
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: deepseek/deepseek-v4-pro
    key_env: ACR_TEST_KEY
agents:
  claude_code:
    upstream: https://api.anthropic.com
    premium:
      provider: anthropic
    premium_tools: [Edit]
    tools:
      Bash: fast
      Edit: premium
  codex:
    upstream: https://api.openai.com
    premium:
      provider: anthropic
    premium_tools: [apply_patch]
    tools:
      exec_command: fast
      apply_patch: premium
  hermes:
    upstream: https://hermes.internal.example/
    premium:
      provider: anthropic
    premium_tools: [patch]
    tools:
      terminal: fast
      patch: premium
"""


@pytest.fixture
def client(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> Iterator[TestClient]:
    with make_client(
        config_yaml=CONFIG_YAML,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as test_client:
        yield test_client


def _messages_opener() -> dict[str, Any]:
    """Return a Claude Code opener with no pending tool results."""
    return {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hello"}],
    }


def _responses_opener() -> dict[str, Any]:
    """Return a Responses opener with no pending tool results."""
    return {"model": "gpt-5-codex", "input": "hello", "stream": True}


def _chat_opener() -> dict[str, Any]:
    """Return a Chat Completions opener with no pending tool results."""
    return {"model": "gpt-hermes", "messages": [{"role": "user", "content": "hello"}]}


def test_claude_code_passthrough_targets_anthropic_upstream(
    *, client: TestClient, upstream: Upstream
) -> None:
    response = client.post("/v1/messages", json=_messages_opener())
    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "api.anthropic.com"


def test_codex_passthrough_targets_openai_upstream(
    *, client: TestClient, upstream: Upstream
) -> None:
    response = client.post("/v1/responses", json=_responses_opener())
    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "api.openai.com"


def test_hermes_passthrough_targets_configured_upstream(
    *, client: TestClient, upstream: Upstream
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=_chat_opener(),
        headers={"x-acr-agent": "hermes"},
    )
    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "hermes.internal.example"


def test_passthrough_forwards_client_headers_without_tier_key(
    *, client: TestClient, upstream: Upstream
) -> None:
    body = _responses_opener()
    client.post(
        "/v1/responses",
        json=body,
        headers={"authorization": "Bearer client-secret"},
    )
    request = upstream.requests[0]
    headers = "\n".join(f"{key}: {value}" for key, value in request.headers.items())
    assert request.headers["authorization"] == "Bearer client-secret"
    assert "tier-key" not in headers
    assert b"tier-key" not in request.content
    assert json.loads(request.content) == body


def test_trailing_slash_upstream_is_normalized(*, client: TestClient, upstream: Upstream) -> None:
    client.post("/v1/chat/completions", json=_chat_opener())
    assert upstream.requests[0].url.host == "hermes.internal.example"
    assert upstream.requests[0].url.path == "/v1/chat/completions"


def test_adapter_none_and_proxy_catchall_use_premium_default(
    *,
    client: TestClient,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_mod, "adapter_for_path", lambda _path: None)
    client.post("/v1/messages", json=_messages_opener())
    client.post("/not-routed", json={"ok": True})
    assert [request.url.host for request in upstream.requests] == [
        "premium.default.example",
        "premium.default.example",
    ]
