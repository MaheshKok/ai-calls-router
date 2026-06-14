"""End-to-end invariant tests for the ai-calls-router serving pipeline.

Unlike test_server.py (which fakes routed_call wholesale), this suite drives
the real decide -> route -> account -> mask -> serve flow through create_app,
mocking only the two network boundaries: the routed provider (litellm
acompletion, via a FakeLitellm injected through routed_call.load_litellm) and
the premium upstream (httpx.MockTransport). It pins one explicit test to each
of the five carried-over invariants:

1. The served body claims the client-requested model, not the routed model.
2. The client's auth credential never reaches the routed provider; only the
   tier key does.
3. Any routed-call failure falls back to premium passthrough.
4. The savings ledger records the true routed model (and prices it from the
   config-declared tier prices the server registers at serve time).
5. An unpriced routed model still serves the turn but writes no ledger entry
   (no fabricated cost numbers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_calls_router.routing import engine as rc
from tests.acr_testkit import FakeLitellm, Upstream, make_client, make_response, read_ledger

# Non-DeepSeek ids keep the integration flow on the LiteLLM path; DeepSeek tier
# models take the native direct path, exercised separately in the unit suite.
CHEAP_MODEL = "groq/acr-int-cheap"
PREMIUM_MODEL = "groq/acr-int-premium"
UNPRICED_MODEL = "groq/acr-int-unpriced"

# Priced config: the fast tier and a never-routed premium_ref tier both carry
# explicit per-1M prices. The server is responsible for teaching these to
# LiteLLM at serve time; the integration suite deliberately does NOT register
# them itself, so invariant 4 fails unless the server wires registration.
PRICED_CONFIG = f"""
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
  compress_routed: false
  premium_tools: [Edit]
  escalate_on_premium_tools: true
tiers:
  fast:
    model: {CHEAP_MODEL}
    key_env: ACR_TEST_KEY
    max_tokens: 1000
    input_cost_per_1m: 0.01
    output_cost_per_1m: 0.02
  premium_ref:
    model: {PREMIUM_MODEL}
    input_cost_per_1m: 10.0
    output_cost_per_1m: 20.0
tools:
  Bash: fast
  Edit: premium
"""

# Unpriced config: the fast tier carries no price, so the routed side cannot be
# priced even though the premium_ref side can. Invariant 5: the turn is still
# served, but no ledger entry is written.
UNPRICED_CONFIG = f"""
server:
  port: 8747
settings:
  tier_precedence: [premium, fast]
  compress_routed: false
  premium_tools: [Edit]
tiers:
  fast:
    model: {UNPRICED_MODEL}
    key_env: ACR_TEST_KEY
    max_tokens: 1000
  premium_ref:
    model: {PREMIUM_MODEL}
    input_cost_per_1m: 10.0
    output_cost_per_1m: 20.0
tools:
  Bash: fast
  Edit: premium
"""


def _fake_response(text: str = "routed reply") -> Any:
    """Build a routed ModelResponse stand-in with 1M-in / 500k-out usage.

    The integration savings assertions are pinned to these large, round token
    counts, so this wrapper fixes them while delegating shape to the shared
    factory.

    Args:
        text: Assistant message text returned by the fake provider.

    Returns:
        A litellm ModelResponse stand-in carrying 1,000,000 prompt and 500,000
        completion tokens.
    """
    return make_response(text=text, prompt_tokens=1_000_000, completion_tokens=500_000)


def _bash_tool_result_body(stream: bool = False) -> dict[str, Any]:
    """Build a request processing one pending Bash tool result."""
    body: dict[str, Any] = {
        "model": PREMIUM_MODEL,
        "max_tokens": 1000,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "files"}],
            },
        ],
    }
    if stream:
        body["stream"] = True
    return body


class TestInvariant1ModelMasking:
    """Invariant 1: served body claims the client-requested model."""

    def test_served_body_reports_client_model_not_routed_model(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        monkeypatch.setattr(rc, "load_litellm", lambda: FakeLitellm(_fake_response()))
        with make_client(
            config_yaml=PRICED_CONFIG, tmp_path=tmp_path, monkeypatch=monkeypatch, upstream=upstream
        ) as client:
            response = client.post("/v1/messages", json=_bash_tool_result_body())
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == PREMIUM_MODEL
        assert CHEAP_MODEL not in json.dumps(body)
        assert upstream.requests == []


class TestInvariant2KeyIsolation:
    """Invariant 2: only the tier key reaches the routed provider."""

    def test_client_auth_never_forwarded_only_tier_key_sent(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        fake = FakeLitellm(_fake_response())
        monkeypatch.setattr(rc, "load_litellm", lambda: fake)
        with make_client(
            config_yaml=PRICED_CONFIG, tmp_path=tmp_path, monkeypatch=monkeypatch, upstream=upstream
        ) as client:
            client.post(
                "/v1/messages",
                json=_bash_tool_result_body(),
                headers={
                    "authorization": "Bearer client-oauth-secret",
                    "x-api-key": "client-anthropic-key",
                },
            )
        assert len(fake.calls) == 1
        kwargs = fake.calls[0]
        assert kwargs["api_key"] == "tier-key"
        serialized = json.dumps(kwargs, default=str)
        assert "client-oauth-secret" not in serialized
        assert "client-anthropic-key" not in serialized
        assert not any("auth" in key.lower() for key in kwargs)


class TestInvariant3FailureFallsBackToPassthrough:
    """Invariant 3: a routed-call failure never breaks the turn."""

    def test_provider_error_falls_back_to_premium_upstream(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        fake = FakeLitellm(error=RuntimeError("provider down"))
        monkeypatch.setattr(rc, "load_litellm", lambda: fake)
        with make_client(
            config_yaml=PRICED_CONFIG, tmp_path=tmp_path, monkeypatch=monkeypatch, upstream=upstream
        ) as client:
            response = client.post("/v1/messages", json=_bash_tool_result_body())
        assert response.json() == {"marker": "upstream"}
        assert len(upstream.requests) == 1
        assert json.loads(upstream.requests[0].content)["model"] == PREMIUM_MODEL


class TestInvariant4LedgerRecordsTrueModel:
    """Invariant 4: the ledger records the true routed model and prices it."""

    def test_ledger_entry_uses_routed_model_with_config_prices(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        monkeypatch.setattr(rc, "load_litellm", lambda: FakeLitellm(_fake_response()))
        with make_client(
            config_yaml=PRICED_CONFIG, tmp_path=tmp_path, monkeypatch=monkeypatch, upstream=upstream
        ) as client:
            response = client.post("/v1/messages", json=_bash_tool_result_body())
        assert response.json()["model"] == PREMIUM_MODEL
        entries = read_ledger(tmp_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["routed_model"] == CHEAP_MODEL
        assert entry["premium_model"] == PREMIUM_MODEL
        # 1M in @ $0.01/1M + 500k out @ $0.02/1M = $0.02 routed.
        assert entry["routed_usd"] == pytest.approx(0.02, abs=1e-6)
        # 1M in @ $10/1M + 500k out @ $20/1M = $20 premium.
        assert entry["premium_usd"] == pytest.approx(20.0, abs=1e-6)
        assert entry["saved_usd"] == pytest.approx(19.98, abs=1e-6)


class TestInvariant5UnpricedSkipsLedger:
    """Invariant 5: an unpriced routed model writes no ledger entry."""

    def test_unpriced_routed_model_serves_turn_without_ledger_entry(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        monkeypatch.setattr(rc, "load_litellm", lambda: FakeLitellm(_fake_response()))
        with make_client(
            config_yaml=UNPRICED_CONFIG,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            upstream=upstream,
        ) as client:
            response = client.post("/v1/messages", json=_bash_tool_result_body())
        # The turn is still served on the cheap tier, masked to the client model.
        assert response.status_code == 200
        assert response.json()["model"] == PREMIUM_MODEL
        assert upstream.requests == []
        # But with no price for the routed model, no figure is fabricated.
        assert read_ledger(tmp_path) == []


class TestStreamingRoutedTurn:
    """A streaming routed turn is served as a synthesized Messages SSE stream."""

    def test_streaming_request_served_as_masked_sse(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
    ) -> None:
        monkeypatch.setattr(rc, "load_litellm", lambda: FakeLitellm(_fake_response()))
        with make_client(
            config_yaml=PRICED_CONFIG, tmp_path=tmp_path, monkeypatch=monkeypatch, upstream=upstream
        ) as client:
            response = client.post("/v1/messages", json=_bash_tool_result_body(stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: message_start" in response.text
        assert "event: message_stop" in response.text
        assert "routed reply" in response.text
        assert PREMIUM_MODEL in response.text
        assert CHEAP_MODEL not in response.text
        assert upstream.requests == []
