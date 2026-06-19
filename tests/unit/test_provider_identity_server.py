"""Server-level tests for provider identity routing.

These tests run the Starlette app with provider YAML files on disk and assert
against the recorded outbound request host. They cover identity resolution,
fail-closed unresolved identity, and per-agent passthrough behavior.
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
from ai_calls_router.proxy import server as server_mod
from ai_calls_router.proxy.server import create_app
from tests.acr_testkit import Upstream

CONFIG_YAML = """
server:
  upstream: https://premium.default.example
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
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: deepseek/deepseek-v4-pro
    key_env: ACR_TEST_KEY
"""

UNRESOLVED_CONFIG_YAML = """
server:
  upstream: https://premium.default.example
router:
  user_agent_map: []
  fallback: null
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: deepseek/deepseek-v4-pro
    key_env: ACR_TEST_KEY
"""


def _provider_payload(group: str, *, upstream: str) -> dict[str, object]:
    """Return a valid provider payload with a distinct upstream."""
    wires = {
        "claude_code": "anthropic_messages",
        "hermes": "openai_chat",
    }
    endpoints = {
        "claude_code": ["/v1/messages"],
        "hermes": ["/v1/chat/completions"],
    }
    tools = {
        "claude_code": {"Bash": "fast", "Edit": "premium"},
        "hermes": {"terminal": "fast", "patch": "premium"},
    }
    premium_tools = {
        "claude_code": ["Edit"],
        "hermes": ["patch"],
    }
    return {
        "group": group,
        "upstream": upstream,
        "auth": {"mode": "oauth_passthrough"},
        "wire": wires[group],
        "endpoints": endpoints[group],
        "tools": tools[group],
        "premium_tools": premium_tools[group],
        "fallback": "passthrough",
    }


class _FakeProviderAssembler:
    """Provider assembler stand-in that raises once for unattributed errors."""

    def __init__(self) -> None:
        self.calls: list[set[str]] = []

    def __call__(
        self,
        base: dict[str, object],
        *,
        provider_files: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        """Record provider groups and fail once while files are present."""
        del base
        self.calls.append(set(provider_files))
        if provider_files:
            raise server_mod.provider_config.ProviderConfigError("boom", group=None)
        return {"agents": {}}


def _write_config_tree(tmp_path: Path, config_yaml: str) -> None:
    """Write global config and all provider files under a temp home."""
    (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
    config.provider_config_dir().mkdir(parents=True, exist_ok=True)
    payloads = {
        "claude_code": _provider_payload(
            "claude_code",
            upstream="https://api.anthropic.com",
        ),
        "hermes": _provider_payload(
            "hermes",
            upstream="https://hermes.internal.example",
        ),
    }
    for group, payload in payloads.items():
        config.provider_config_path(group).write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )


@pytest.fixture
def upstream() -> Upstream:
    """Return a mock passthrough upstream recorder."""
    return Upstream()


@pytest.fixture
def client(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> Iterator[TestClient]:
    """Create an app with provider YAML files."""
    metrics_mod._metrics_singleton = None
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    _write_config_tree(tmp_path, CONFIG_YAML)
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    with TestClient(app) as test_client:
        yield test_client
    metrics_mod._metrics_singleton = None


def _chat_opener() -> dict[str, object]:
    """Return a Chat opener with no pending tool result."""
    return {"model": "gpt-hermes", "messages": [{"role": "user", "content": "hello"}]}


def _premium_chat_turn() -> dict[str, object]:
    """Return a Hermes turn whose pending tool maps to premium."""
    return {
        "model": "gpt-hermes",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_patch",
                        "type": "function",
                        "function": {"name": "patch", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_patch", "content": "applied"},
        ],
    }


def test_header_resolves_hermes_for_chat(
    client: TestClient,
    upstream: Upstream,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=_chat_opener(),
        headers={"x-acr-agent": "hermes"},
    )

    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "hermes.internal.example"


def test_hermes_user_agent_resolves_chat_upstream(
    client: TestClient,
    upstream: Upstream,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=_chat_opener(),
        headers={"user-agent": "Hermes Desktop"},
    )

    assert response.json() == {"marker": "upstream"}
    assert upstream.requests[0].url.host == "hermes.internal.example"


def test_unresolved_identity_returns_400_without_upstream_request(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    _write_config_tree(tmp_path, UNRESOLVED_CONFIG_YAML)
    app = create_app(transport=httpx.MockTransport(upstream.handler))

    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_chat_opener(),
            headers={"authorization": "Bearer client-secret"},
        )

    assert response.status_code == 400
    assert response.json() == {"error": "unresolved agent identity"}
    assert upstream.requests == []


def test_invalid_provider_payload_is_dropped_with_warning(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    _write_config_tree(tmp_path, CONFIG_YAML)
    payload = _provider_payload("hermes", upstream="https://hermes.internal.example")
    payload["tiers"] = {"fast": {"key_env": "MUST_NOT_LEAK"}}
    config.provider_config_path("hermes").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    routes = server_mod._load_assembled_routes()
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    with TestClient(app) as test_client:
        response = test_client.post("/v1/chat/completions", json=_chat_opener())

    assert response.status_code == 200
    assert "provider config assembly failed" in caplog.text
    assert "MUST_NOT_LEAK" not in json.dumps(routes)
    assert upstream.requests[0].url.host == "premium.default.example"


def test_assembled_routes_signature_changes_when_provider_file_is_deleted(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(tmp_path / "config.yaml"))
    _write_config_tree(tmp_path, CONFIG_YAML)

    before = server_mod._assembled_routes_signature()
    config.provider_config_path("hermes").unlink()

    assert server_mod._assembled_routes_signature() != before


def test_assemble_routes_fail_open_drops_unattributed_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProviderAssembler()
    monkeypatch.setattr(server_mod.provider_config, "assemble_routes", fake)

    assert server_mod._assemble_routes_fail_open({}, {"hermes": {}}) == {"agents": {}}
    assert fake.calls == [{"hermes"}, set()]


def test_resolved_premium_turn_passes_through_to_group_upstream(
    client: TestClient,
    upstream: Upstream,
) -> None:
    body = _premium_chat_turn()
    response = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"authorization": "Bearer client-secret"},
    )

    assert response.json() == {"marker": "upstream"}
    request = upstream.requests[0]
    assert request.url.host == "hermes.internal.example"
    assert request.headers["authorization"] == "Bearer client-secret"
    assert "tier-key" not in "\n".join(f"{key}: {value}" for key, value in request.headers.items())
    assert b"tier-key" not in request.content
    assert json.loads(request.content) == body
