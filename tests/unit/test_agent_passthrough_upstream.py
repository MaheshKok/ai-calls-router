"""Tests for per-agent passthrough upstream selection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from ai_calls_router.proxy import orchestrator as orchestrator_mod
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
router:
  endpoint_defaults:
    /v1/messages: claude_code
    /v1/chat/completions: hermes
  user_agent_map:
    - contains: claude
      group: claude_code
    - contains: hermes
      group: hermes
  fallback: null
agents:
  claude_code:
    upstream: https://api.anthropic.com
    premium:
      provider: anthropic
    premium_tools: [Edit]
    tools:
      Bash: fast
      Edit: premium
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
    monkeypatch.setattr(server_mod.bootstrap, "ensure_provider_configs", lambda: [])
    with make_client(
        config_yaml=CONFIG_YAML,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as test_client:
        yield test_client


def _messages_opener() -> dict[str, object]:
    """Return a Claude Code opener with no pending tool results."""
    return {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hello"}],
    }


def _chat_opener() -> dict[str, object]:
    """Return a Chat Completions opener with no pending tool result."""
    return {"model": "gpt-hermes", "messages": [{"role": "user", "content": "hello"}]}


def test_claude_code_passthrough_targets_anthropic_upstream(
    *, client: TestClient, upstream: Upstream
) -> None:
    response = client.post("/v1/messages", json=_messages_opener())
    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "api.anthropic.com"


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
    body = _chat_opener()
    client.post(
        "/v1/chat/completions",
        json=body,
        headers={"authorization": "Bearer client-secret"},
    )
    request = upstream.requests[0]
    headers = "\n".join(f"{key}: {value}" for key, value in request.headers.items())
    assert request.headers["authorization"] == "Bearer client-secret"
    assert "tier-key" not in headers
    assert b"tier-key" not in request.content
    assert json.loads(request.content) == body


def test_passthrough_strips_long_context_opt_in(*, client: TestClient, upstream: Upstream) -> None:
    # opus[1m] passthrough turns 429 on a subscription without long-context credits;
    # acr drops the [1m] model suffix and the context-1m beta before relaying.
    body = {**_messages_opener(), "model": "claude-opus-4-8[1m]"}
    client.post(
        "/v1/messages",
        json=body,
        headers={"anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07"},
    )
    request = upstream.requests[0]
    assert request.url.host == "api.anthropic.com"
    assert json.loads(request.content)["model"] == "claude-opus-4-8"
    assert request.headers["anthropic-beta"] == "oauth-2025-04-20"


def test_passthrough_preserves_request_without_long_context(
    *, client: TestClient, upstream: Upstream
) -> None:
    # A turn that does not opt into long context relays byte-identical (cache intact).
    body = _messages_opener()
    client.post("/v1/messages", json=body, headers={"anthropic-beta": "oauth-2025-04-20"})
    request = upstream.requests[0]
    assert json.loads(request.content) == body
    assert request.headers["anthropic-beta"] == "oauth-2025-04-20"


def test_passthrough_adds_prompt_cache_when_enabled(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> None:
    monkeypatch.setattr(server_mod.bootstrap, "ensure_provider_configs", lambda: [])
    with make_client(
        config_yaml=CONFIG_YAML.replace(
            "settings:\n  tier_precedence: [premium, fast]",
            "settings:\n  tier_precedence: [premium, fast]\n  anthropic_prompt_cache: true",
        ),
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as cache_client:
        cache_client.post("/v1/messages", json=_messages_opener())

    sent = json.loads(upstream.requests[0].content)
    assert sent["cache_control"] == {"type": "ephemeral"}


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
    monkeypatch.setattr(orchestrator_mod, "adapter_for_path", lambda _path: None)
    client.post("/v1/messages", json=_messages_opener())
    client.post("/not-routed", json={"ok": True})
    assert [request.url.host for request in upstream.requests] == [
        "premium.default.example",
        "premium.default.example",
    ]
