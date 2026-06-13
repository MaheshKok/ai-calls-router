"""Spec-derived tests for ai_calls_router.anthropic_direct.

Contract under test: direct_endpoint is the detection gate -- it returns a base
URL only when a tier model's provider prefix is wired for direct Anthropic
routing (DeepSeek today) and None otherwise, so non-DeepSeek and bare ids keep
the LiteLLM path; native_model_id strips exactly the LiteLLM provider prefix;
direct_call POSTs the prepared body verbatim except for swapping in the native
model id and forcing thinking off, carries only the tier key (never the
client's Anthropic credential), never mutates the caller's body, and returns
None on any non-200 or transport failure so routing falls back to passthrough.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from ai_calls_router.routing import direct as ad

TIER_CFG: dict[str, Any] = {"model": "deepseek/deepseek-v4-pro", "max_tokens": 8192}
RESPONSE_JSON: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "deepseek-v4-pro",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class _Recorder:
    """Records every request a MockTransport receives and answers canned."""

    def __init__(
        self,
        *,
        status: int = 200,
        payload: dict[str, Any] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status
        self._payload = payload if payload is not None else RESPONSE_JSON
        self._raise = raise_exc

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._raise is not None:
            raise self._raise
        return httpx.Response(self._status, json=self._payload)


def _run_direct(
    *, body: dict[str, Any], tier_cfg: dict[str, Any], api_key: str, recorder: _Recorder
) -> dict[str, Any] | None:
    """Drive direct_call with an injected MockTransport client."""

    async def _go() -> dict[str, Any] | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler)) as client:
            return await ad.direct_call(
                body=body, tier_cfg=tier_cfg, api_key=api_key, client=client
            )

    return asyncio.run(_go())


class TestProviderPrefix:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("deepseek/deepseek-v4-pro", "deepseek"),
            ("openrouter/deepseek/chat", "openrouter"),
            ("groq/llama-3.3-70b", "groq"),
        ],
    )
    def test_returns_first_segment(self, model: str, expected: str) -> None:
        assert ad.provider_prefix(model) == expected

    @pytest.mark.parametrize("model", ["bare-id", "", "deepseek"])
    def test_bare_id_has_no_prefix(self, model: str) -> None:
        assert ad.provider_prefix(model) is None

    @pytest.mark.parametrize("model", [None, 42, ["deepseek/x"], {"m": 1}])
    def test_non_string_has_no_prefix(self, model: Any) -> None:
        assert ad.provider_prefix(model) is None


class TestDirectEndpoint:
    def test_deepseek_prefix_maps_to_endpoint(self) -> None:
        assert (
            ad.direct_endpoint("deepseek/deepseek-v4-flash") == "https://api.deepseek.com/anthropic"
        )

    def test_any_deepseek_model_maps(self) -> None:
        assert (
            ad.direct_endpoint("deepseek/whatever-future-model")
            == "https://api.deepseek.com/anthropic"
        )

    def test_non_deepseek_provider_is_not_direct(self) -> None:
        assert ad.direct_endpoint("groq/llama-3.3-70b") is None

    def test_deepseek_only_in_later_segment_is_not_direct(self) -> None:
        # Prefix is "openrouter", not "deepseek": must stay on the LiteLLM path.
        assert ad.direct_endpoint("openrouter/deepseek/chat") is None

    def test_bare_id_is_not_direct(self) -> None:
        assert ad.direct_endpoint("deepseek-v4-pro") is None

    @pytest.mark.parametrize("model", [None, 42, ["x"]])
    def test_non_string_is_not_direct(self, model: Any) -> None:
        assert ad.direct_endpoint(model) is None


class TestNativeModelId:
    def test_strips_provider_prefix(self) -> None:
        assert ad.native_model_id("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"

    def test_strips_only_first_segment(self) -> None:
        assert ad.native_model_id("openrouter/deepseek/chat") == "deepseek/chat"

    def test_bare_id_unchanged(self) -> None:
        assert ad.native_model_id("deepseek-v4-pro") == "deepseek-v4-pro"


class TestDirectCall:
    def test_returns_parsed_body_on_200(self) -> None:
        rec = _Recorder()
        result = _run_direct(body={"messages": []}, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
        assert result == RESPONSE_JSON

    def test_posts_to_anthropic_v1_messages(self) -> None:
        rec = _Recorder()
        _run_direct(body={"messages": []}, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
        assert str(rec.requests[0].url) == "https://api.deepseek.com/anthropic/v1/messages"
        assert rec.requests[0].method == "POST"

    def test_payload_uses_native_model_id(self) -> None:
        rec = _Recorder()
        _run_direct(
            body={"messages": [], "model": "claude-sonnet-4"},
            tier_cfg=TIER_CFG,
            api_key="KEY",
            recorder=rec,
        )
        sent = json.loads(rec.requests[0].content)
        assert sent["model"] == "deepseek-v4-pro"

    def test_payload_disables_thinking_even_when_client_enabled_it(self) -> None:
        rec = _Recorder()
        body = {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 5000}}
        _run_direct(body=body, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
        sent = json.loads(rec.requests[0].content)
        assert sent["thinking"] == {"type": "disabled"}

    def test_authorization_carries_only_tier_key(self) -> None:
        rec = _Recorder()
        _run_direct(
            body={"messages": []}, tier_cfg=TIER_CFG, api_key="sk-tier-secret", recorder=rec
        )
        headers = rec.requests[0].headers
        assert headers["authorization"] == "Bearer sk-tier-secret"
        assert headers["anthropic-version"] == "2023-06-01"
        # Invariant 2: the client's Anthropic credential never reaches a tier.
        assert "x-api-key" not in headers

    def test_does_not_mutate_input_body(self) -> None:
        rec = _Recorder()
        body = {"messages": [{"role": "user", "content": "hi"}], "model": "claude-x"}
        snapshot = json.loads(json.dumps(body))
        _run_direct(body=body, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
        assert body == snapshot

    @pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
    def test_non_200_returns_none(self, status: int) -> None:
        rec = _Recorder(status=status, payload={"error": "nope"})
        assert (
            _run_direct(body={"messages": []}, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
            is None
        )

    def test_transport_error_returns_none(self) -> None:
        rec = _Recorder(raise_exc=httpx.ConnectError("boom"))
        assert (
            _run_direct(body={"messages": []}, tier_cfg=TIER_CFG, api_key="KEY", recorder=rec)
            is None
        )

    def test_unmapped_model_returns_none_without_request(self) -> None:
        rec = _Recorder()
        result = _run_direct(
            body={"messages": []},
            tier_cfg={"model": "groq/llama-3.3-70b"},
            api_key="KEY",
            recorder=rec,
        )
        assert result is None
        assert rec.requests == []

    def test_missing_model_returns_none_without_request(self) -> None:
        rec = _Recorder()
        result = _run_direct(
            body={"messages": []}, tier_cfg={"max_tokens": 8192}, api_key="KEY", recorder=rec
        )
        assert result is None
        assert rec.requests == []

    def test_creates_own_client_when_none_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Covers the transient-client branch (own_client True -> aclose).
        rec = _Recorder()
        real_async_client = httpx.AsyncClient

        def _factory(*_args: Any, **_kwargs: Any) -> httpx.AsyncClient:
            return real_async_client(transport=httpx.MockTransport(rec.handler))

        monkeypatch.setattr(ad.httpx, "AsyncClient", _factory)

        async def _go() -> dict[str, Any] | None:
            return await ad.direct_call(body={"messages": []}, tier_cfg=TIER_CFG, api_key="KEY")

        result = asyncio.run(_go())
        assert result == RESPONSE_JSON
        assert len(rec.requests) == 1
