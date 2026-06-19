"""App-level tests for OpenAI Responses routing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from ai_calls_router.accounting import metrics as metrics_mod
from ai_calls_router.accounting import shrink_stats
from ai_calls_router.proxy import server as server_mod
from ai_calls_router.proxy.server import create_app
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as rc
from tests.acr_testkit import read_ledger

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

CODEX_DIRECT_CONFIG_YAML = """
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: codex/gpt-5-codex-spark
    provider: codex
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

CODEX_DIRECT_OAUTH_CONFIG_YAML = CODEX_DIRECT_CONFIG_YAML.replace(
    "    max_tokens: 1000\n",
    "    key_env: oauth\n    max_tokens: 1000\n",
)

PRICED_PREMIUM_MODEL = "deepseek/acr-test-premium"

CODEX_DIRECT_PRICED_CONFIG_YAML = """
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
tiers:
  fast:
    model: codex/acr-test-cheap
    provider: codex
    input_cost_per_1m: 1.0
    input_cached_cost_per_1m: 0.5
    output_cost_per_1m: 2.0
  premium_stand_in:
    model: deepseek/acr-test-premium
    input_cost_per_1m: 10.0
    output_cost_per_1m: 20.0
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

DEEPSEEK_RESPONSES_PRICED_CONFIG_YAML = CODEX_DIRECT_PRICED_CONFIG_YAML.replace(
    "    model: codex/acr-test-cheap\n    provider: codex\n",
    "    model: deepseek/acr-test-cheap\n    key_env: ACR_TEST_KEY\n",
)

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
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, object] | None:
        """Record direct call input and return the canned response."""
        del client
        self.calls.append({"body": body, "tier_cfg": tier_cfg, "api_key": api_key})
        return self.response


class _FakeCodexResponsesCall:
    """Codex direct call stand-in recording invocations."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        *,
        body: dict[str, object],
        tier_cfg: dict[str, object],
        credential: str,
        auth_mode: str = "api_key",
        chatgpt_headers: list[tuple[str, str]] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> tuple[dict[str, object], tuple[int, int, int, int], shrink_stats.ShrinkStats]:
        """Record direct Codex call input and return a Responses body."""
        self.calls.append(
            {
                "body": body,
                "tier_cfg": tier_cfg,
                "credential": credential,
                "auth_mode": auth_mode,
                "chatgpt_headers": chatgpt_headers or [],
                "client": client,
            }
        )
        response = {
            "id": "resp_codex_direct",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5-codex-spark",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "codex routed"}],
                }
            ],
            "usage": {
                "input_tokens": 2,
                "output_tokens": 2,
                "total_tokens": 4,
                "input_tokens_details": {"cached_tokens": 1},
            },
        }
        return (
            response,
            (1, 2, 1, 0),
            shrink_stats.ShrinkStats(path="none", chars_before=4, chars_after=4),
        )


class _FailingCodexResponsesCall:
    """Codex direct call stand-in that declines routing."""

    async def __call__(self, **_: object) -> None:
        """Return None so the server falls back to passthrough."""


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


def _client_for_config(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
    config_yaml: str,
) -> TestClient:
    """Build a TestClient for a custom config payload."""
    metrics_mod._metrics_singleton = None
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml, encoding="utf-8")
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    return TestClient(app)


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
        "tools": [
            {"type": "web_search_preview"},
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
        ],
    }


def _responses_tool_result_body_for_model(model: str, *, stream: bool = False) -> dict[str, object]:
    """Build a Responses tool-result body with a chosen premium model."""
    body = _responses_tool_result_body(stream=stream)
    body["model"] = model
    return body


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
    assert fake.calls[0]["body"]["tools"] == [
        {"name": "exec_command", "input_schema": {"type": "object"}}
    ]

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


def test_codex_direct_route_uses_openai_api_key(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> None:
    fake = _FakeCodexResponsesCall()
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("ai_calls_router.proxy.server.codex_direct.responses_call", fake)
    with _client_for_config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=CODEX_DIRECT_CONFIG_YAML,
    ) as test_client:
        response = test_client.post(
            "/v1/responses",
            json=_responses_tool_result_body(stream=False),
            headers={"authorization": "Bearer client-premium-secret"},
        )
        snapshot = metrics_mod.get_metrics().snapshot()

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-5-codex"
    assert response.json()["output"][0]["content"][0]["text"] == "codex routed"
    assert fake.calls[0]["credential"] == "openai-key"
    assert isinstance(fake.calls[0]["client"], httpx.AsyncClient)
    assert upstream.requests == []
    assert snapshot["routed_tokens"] == {
        "input": 1,
        "output": 2,
        "cache_read": 1,
        "cache_creation": 0,
    }
    assert snapshot["last_requests"][0]["input_tokens"] == 1
    assert snapshot["last_requests"][0]["output_tokens"] == 2


def test_routed_codex_call_records_savings_and_tokens(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> None:
    fake = _FakeCodexResponsesCall()
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("ai_calls_router.proxy.server.codex_direct.responses_call", fake)
    with _client_for_config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=CODEX_DIRECT_PRICED_CONFIG_YAML,
    ) as test_client:
        response = test_client.post(
            "/v1/responses",
            json=_responses_tool_result_body_for_model(PRICED_PREMIUM_MODEL),
        )
        snapshot = metrics_mod.get_metrics().snapshot()

    entries = read_ledger(tmp_path)
    assert response.status_code == 200
    assert upstream.requests == []
    assert len(entries) == 1
    entry = entries[0]
    assert entry["premium_model"] == PRICED_PREMIUM_MODEL
    assert entry["routed_model"] == "codex/acr-test-cheap"
    assert entry["input_tokens"] == 2
    assert entry["output_tokens"] == 2
    assert entry["cache_read_input_tokens"] == 1
    assert entry["cache_creation_input_tokens"] == 0
    assert entry["saved_usd"] > 0
    assert snapshot["routed_tokens"] == {
        "input": 1,
        "output": 2,
        "cache_read": 1,
        "cache_creation": 0,
    }
    assert snapshot["last_requests"][0]["path"] == "/v1/responses"
    assert snapshot["last_requests"][0]["input_tokens"] == 1


def test_codex_and_deepseek_record_same_shape(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> None:
    codex_home = tmp_path / "codex"
    deepseek_home = tmp_path / "deepseek"
    codex_home.mkdir()
    deepseek_home.mkdir()
    fake_codex = _FakeCodexResponsesCall()
    fake_deepseek = _FakeDirectCall(
        {
            **ROUTED_TEXT_BODY,
            "model": "acr-test-cheap",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_read_input_tokens": 1,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    monkeypatch.setattr("ai_calls_router.proxy.server.codex_direct.responses_call", fake_codex)
    monkeypatch.setattr(rc.anthropic_direct, "direct_call", fake_deepseek)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")

    with _client_for_config(
        tmp_path=codex_home,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=CODEX_DIRECT_PRICED_CONFIG_YAML,
    ) as test_client:
        assert (
            test_client.post(
                "/v1/responses",
                json=_responses_tool_result_body_for_model(PRICED_PREMIUM_MODEL),
            ).status_code
            == 200
        )
    metrics_mod._metrics_singleton = None
    with _client_for_config(
        tmp_path=deepseek_home,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=DEEPSEEK_RESPONSES_PRICED_CONFIG_YAML,
    ) as test_client:
        assert (
            test_client.post(
                "/v1/responses",
                json=_responses_tool_result_body_for_model(PRICED_PREMIUM_MODEL),
            ).status_code
            == 200
        )

    codex_entry = read_ledger(codex_home)[0]
    deepseek_entry = read_ledger(deepseek_home)[0]
    assert set(codex_entry) == set(deepseek_entry)


def test_codex_direct_route_uses_oauth_sentinel(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> None:
    fake = _FakeCodexResponsesCall()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("ai_calls_router.proxy.server.codex_direct.responses_call", fake)
    with _client_for_config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=CODEX_DIRECT_OAUTH_CONFIG_YAML,
    ) as test_client:
        response = test_client.post(
            "/v1/responses",
            json=_responses_tool_result_body(stream=False),
            headers={
                "authorization": "Bearer chatgpt-oauth",
                "chatgpt-account-id": "acct_123",
            },
        )

    chatgpt_headers = fake.calls[0]["chatgpt_headers"]
    assert isinstance(chatgpt_headers, list)
    headers = {key.lower(): value for key, value in chatgpt_headers}
    assert response.status_code == 200
    assert fake.calls[0]["credential"] == "oauth"
    assert headers["authorization"] == "Bearer chatgpt-oauth"
    assert headers["chatgpt-account-id"] == "acct_123"
    assert upstream.requests == []


def test_codex_upstream_error_falls_back_to_premium(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
) -> None:
    monkeypatch.setattr(
        "ai_calls_router.proxy.server.codex_direct.responses_call",
        _FailingCodexResponsesCall(),
    )
    body = _responses_tool_result_body_for_model(PRICED_PREMIUM_MODEL)
    with _client_for_config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
        config_yaml=CODEX_DIRECT_PRICED_CONFIG_YAML,
    ) as test_client:
        response = test_client.post("/v1/responses", json=body)

    assert response.status_code == 200
    assert response.json() == {"marker": "upstream"}
    assert json.loads(upstream.requests[0].content) == body


@pytest.mark.asyncio
async def test_codex_direct_attempt_reports_yaml_model_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCodexResponsesCall()
    monkeypatch.setattr("ai_calls_router.proxy.server.codex_direct.responses_call", fake)

    async with httpx.AsyncClient() as client:
        attempt = await server_mod._try_codex_direct_route(
            body=_responses_tool_result_body(stream=True),
            tier="codex_code",
            tier_cfg={"model": "codex/gpt-configured-from-yaml", "provider": "codex"},
            credential=routing.TierCredential(value="oauth", auth_mode="oauth"),
            request_headers={},
            streaming=True,
            requested_model="gpt-5.5",
            names=["exec_command"],
            premium_tools=[],
            request_path="/v1/responses",
            user_agent="codex-test",
            agent="codex",
            session="session",
            client=client,
        )

    assert attempt is not None
    assert attempt.model == "codex/gpt-configured-from-yaml"
    assert attempt.input_tokens == 1
    assert attempt.output_tokens == 2
    assert attempt.cache_read_tokens == 1
    assert b'"model": "gpt-5.5"' in bytes(attempt.response.body)


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
