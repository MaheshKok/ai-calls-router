"""End-to-end smoke tests for the OpenAI compatibility routing surface.

These tests drive each public endpoint through the Starlette app with Phase 7
provider YAML files on disk. They mock only HTTP boundaries: the premium
passthrough upstream and the DeepSeek direct endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import yaml
from starlette.testclient import TestClient

from ai_calls_router._lib import config
from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.proxy.server import create_app
from ai_calls_router.routing import direct as direct_mod
from ai_calls_router.routing.adapters.base import AGENT_GROUP_ENDPOINTS, AGENT_GROUP_WIRES
from ai_calls_router.routing.agent_defaults import (
    AGENT_DEFAULT_PREMIUM_TOOLS,
    AGENT_DEFAULT_TOOLS,
)
from tests.acr_testkit import Upstream

_ORIGINAL_ASYNC_CLIENT = httpx.AsyncClient
_ROUTED_MODEL = "deepseek/deepseek-v4-pro"
_CLIENT_AUTH = "Bearer client-token-for-test"
_ROUTED_TEXT_BODY: dict[str, object] = {
    "id": "msg_smoke",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "routed answer"}],
    "model": "deepseek-v4-pro",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}
_UPSTREAMS = {
    "claude_code": "https://api.anthropic.com",
    "codex": "https://api.openai.com",
    "hermes": "https://hermes.internal.example",
}


class _DirectProvider:
    """Mock DeepSeek endpoint that records native direct-path requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record one DeepSeek request and return an Anthropic response body."""
        self.requests.append(request)
        return httpx.Response(
            200,
            json=_ROUTED_TEXT_BODY,
            headers={"content-type": "application/json"},
        )

    def client(self, *args: object, **kwargs: object) -> httpx.AsyncClient:
        """Return an AsyncClient wired to this mock provider transport."""
        return _ORIGINAL_ASYNC_CLIENT(
            *args,
            transport=httpx.MockTransport(self.handler),
            **kwargs,
        )


@pytest.fixture
def upstream() -> Upstream:
    """Return a premium passthrough recorder."""
    return Upstream()


@pytest.fixture
def client(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> Iterator[TestClient]:
    """Create a proxy app with global router config and provider YAML files."""
    metrics_mod._metrics_singleton = None
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    _write_global_config(tmp_path)
    _write_provider_configs()
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    with TestClient(app) as test_client:
        yield test_client
    metrics_mod._metrics_singleton = None


def _write_global_config(tmp_path: Path) -> None:
    """Write the global config.yaml used by all smoke tests."""
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "server": {"upstream": "https://premium.default.example"},
                "router": {
                    "endpoint_defaults": {
                        "/v1/messages": "claude_code",
                        "/v1/chat/completions": "hermes",
                        "/v1/responses": "codex",
                    },
                    "fallback": None,
                },
                "settings": {"tier_precedence": ["premium", "fast"]},
                "tiers": {
                    "fast": {
                        "model": _ROUTED_MODEL,
                        "key_env": "ACR_TEST_KEY",
                        "max_tokens": 1000,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_provider_configs() -> None:
    """Write per-provider YAML files with distinct premium upstreams."""
    config.provider_config_dir().mkdir(parents=True, exist_ok=True)
    for group, upstream in _UPSTREAMS.items():
        payload = {
            "group": group,
            "upstream": upstream,
            "auth": {"mode": "oauth_passthrough"},
            "wire": AGENT_GROUP_WIRES[group],
            "endpoints": list(AGENT_GROUP_ENDPOINTS[group]),
            "tools": dict(AGENT_DEFAULT_TOOLS[group]),
            "premium_tools": list(AGENT_DEFAULT_PREMIUM_TOOLS[group]),
            "fallback": "passthrough",
        }
        config.provider_config_path(group).write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )


def _claude_tool_result_body() -> dict[str, object]:
    """Return a Claude Code turn processing a Bash tool result."""
    return {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_bash", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_bash", "content": "ok"}],
            },
        ],
    }


def _claude_premium_body() -> dict[str, object]:
    """Return a Claude Code decision turn with no pending tool result."""
    return {
        "model": "claude-fable-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "decide"}],
    }


def _codex_tool_result_body() -> dict[str, object]:
    """Return a Codex Responses turn processing an exec_command result."""
    return {
        "model": "gpt-5-codex",
        "stream": True,
        "input": [
            {
                "type": "function_call",
                "call_id": "call_exec",
                "name": "exec_command",
                "arguments": '{"cmd":"pwd"}',
            },
            {"type": "function_call_output", "call_id": "call_exec", "output": "/tmp"},
        ],
    }


def _codex_premium_body() -> dict[str, object]:
    """Return a Codex decision turn with no pending tool result."""
    return {"model": "gpt-5-codex", "stream": True, "input": "decide"}


def _hermes_tool_result_body() -> dict[str, object]:
    """Return a Hermes Chat turn processing a terminal result."""
    return {
        "model": "gpt-hermes",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_terminal",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_terminal", "content": "file.txt"},
        ],
    }


def _hermes_premium_body() -> dict[str, object]:
    """Return a Hermes Chat decision turn with no pending tool result."""
    return {"model": "gpt-hermes", "messages": [{"role": "user", "content": "decide"}]}


def _post_json_bytes(
    *,
    client: TestClient,
    path: str,
    body: dict[str, object],
) -> httpx.Response:
    """Post byte-stable JSON through the TestClient."""
    raw = json.dumps(body, separators=(",", ":")).encode()
    return client.post(
        path,
        content=raw,
        headers={
            "authorization": _CLIENT_AUTH,
            "content-type": "application/json",
        },
    )


def _assert_routed_direct(
    *,
    client: TestClient,
    response: httpx.Response,
    direct_provider: _DirectProvider,
    path: str,
    tool_name: str,
) -> None:
    """Assert a cheap turn reached DeepSeek direct with the tier key only."""
    assert response.status_code == 200
    assert len(direct_provider.requests) == 1
    request = direct_provider.requests[0]
    assert request.url.host == "api.deepseek.com"
    assert request.headers["authorization"] == "Bearer tier-key"
    assert _CLIENT_AUTH not in request.headers.values()
    assert _CLIENT_AUTH.encode() not in request.content
    assert json.loads(request.content)["model"] == "deepseek-v4-pro"

    latest = client.get("/metrics").json()["last_requests"][0]
    assert latest["path"] == path
    assert latest["route"] == "direct"
    assert latest["model"] == _ROUTED_MODEL
    assert latest["tool_names"] == [tool_name]
    assert latest["shrink_chars_before"] == latest["shrink_chars_after"]


def _assert_premium_passthrough(
    *,
    upstream: Upstream,
    host: str,
    body: dict[str, object],
) -> None:
    """Assert a premium turn stayed byte-identical and used client credentials."""
    request = upstream.requests[-1]
    raw = json.dumps(body, separators=(",", ":")).encode()
    assert request.url.host == host
    assert request.headers["authorization"] == _CLIENT_AUTH
    assert "tier-key" not in "\n".join(f"{key}: {value}" for key, value in request.headers.items())
    assert b"tier-key" not in request.content
    assert request.content == raw


def _exercise_group(
    *,
    client: TestClient,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    tool_name: str,
    cheap_body: dict[str, object],
    premium_body: dict[str, object],
    premium_host: str,
) -> None:
    """Drive one group through routed direct and premium passthrough paths."""
    direct_provider = _DirectProvider()
    monkeypatch.setattr(direct_mod.httpx, "AsyncClient", direct_provider.client)

    routed_response = _post_json_bytes(client=client, path=path, body=cheap_body)
    _assert_routed_direct(
        client=client,
        response=routed_response,
        direct_provider=direct_provider,
        path=path,
        tool_name=tool_name,
    )

    passthrough_response = _post_json_bytes(client=client, path=path, body=premium_body)
    assert passthrough_response.json() == {"marker": "upstream"}
    _assert_premium_passthrough(upstream=upstream, host=premium_host, body=premium_body)


def test_claude_code_messages_smoke(
    *,
    client: TestClient,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_group(
        client=client,
        upstream=upstream,
        monkeypatch=monkeypatch,
        path="/v1/messages",
        tool_name="Bash",
        cheap_body=_claude_tool_result_body(),
        premium_body=_claude_premium_body(),
        premium_host="api.anthropic.com",
    )


def test_codex_responses_smoke(
    *,
    client: TestClient,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_group(
        client=client,
        upstream=upstream,
        monkeypatch=monkeypatch,
        path="/v1/responses",
        tool_name="exec_command",
        cheap_body=_codex_tool_result_body(),
        premium_body=_codex_premium_body(),
        premium_host="api.openai.com",
    )


def test_hermes_chat_completions_smoke(
    *,
    client: TestClient,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_group(
        client=client,
        upstream=upstream,
        monkeypatch=monkeypatch,
        path="/v1/chat/completions",
        tool_name="terminal",
        cheap_body=_hermes_tool_result_body(),
        premium_body=_hermes_premium_body(),
        premium_host="hermes.internal.example",
    )
