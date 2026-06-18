"""Tests for direct Codex Responses routed calls.

The Codex subscription path is provider-specific: API-key calls go to the
OpenAI Responses endpoint, while the OAuth sentinel goes to ChatGPT's Codex
backend with ChatGPT headers. These tests exercise auth isolation and routed
body sanitization without relying on a real network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ai_calls_router.routing import codex_direct


def _tier(**extra: object) -> dict[str, object]:
    """Return a Codex tier config."""
    return {"model": "codex/gpt-5-codex-spark", "provider": "codex", **extra}


def _body() -> dict[str, object]:
    """Return a Responses body with one stripped reasoning item."""
    return {
        "model": "gpt-5-codex",
        "stream": True,
        "previous_response_id": "resp_old",
        "max_output_tokens": 9999,
        "input": [
            {"type": "reasoning", "encrypted_content": "secret"},
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        ],
    }


def test_prepare_responses_body_strips_reasoning_and_clamps_without_mutating() -> None:
    body = _body()

    routed = codex_direct.prepare_responses_body(body, _tier(max_tokens=2000))

    assert routed["model"] == "gpt-5-codex-spark"
    assert routed["max_output_tokens"] == 2000
    assert routed["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "done"}
    ]
    assert "stream" not in routed
    assert "previous_response_id" not in routed
    assert body["input"][0]["encrypted_content"] == "secret"


def test_prepare_responses_body_handles_string_input_and_plain_model() -> None:
    routed = codex_direct.prepare_responses_body(
        {"model": "gpt-5-codex", "input": "hello"},
        {"model": "gpt-5-codex", "provider": "codex"},
    )

    assert routed["model"] == "gpt-5-codex"
    assert routed["input"] == "hello"


def test_response_escalates_empty_and_malformed_outputs() -> None:
    assert codex_direct.response_escalates({"output": []}, []) == []
    assert codex_direct.response_escalates({"output": "bad"}, ["apply_patch"]) == []
    assert (
        codex_direct.response_escalates(
            {"output": ["bad", {"type": "message"}, {"type": "function_call", "name": 7}]},
            ["apply_patch"],
        )
        == []
    )


@pytest.mark.asyncio
async def test_responses_call_api_key_uses_openai_endpoint_without_client_auth() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5-codex-spark",
                "output": [],
                "usage": None,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await codex_direct.responses_call(
            body=_body(),
            tier_cfg=_tier(),
            credential="openai-key",
            chatgpt_headers=[("Authorization", "Bearer client-oauth")],
            client=client,
        )

    assert response is not None
    assert str(captured[0].url) == codex_direct.OPENAI_RESPONSES_URL
    assert captured[0].headers["authorization"] == "Bearer openai-key"
    assert "client-oauth" not in json.dumps(dict(captured[0].headers))
    assert "secret" not in captured[0].content.decode()
    assert "stream" not in json.loads(captured[0].content)


@pytest.mark.asyncio
async def test_responses_call_oauth_uses_chatgpt_backend_headers() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_2",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5-codex-spark",
                "output": [],
                "usage": None,
            },
        }
        return httpx.Response(
            200,
            content=f"event: response.completed\ndata: {json.dumps(completed)}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await codex_direct.responses_call(
            body=_body(),
            tier_cfg=_tier(),
            credential="oauth",
            chatgpt_headers=[
                ("Authorization", "Bearer chatgpt-oauth"),
                ("ChatGPT-Account-ID", "acct_123"),
            ],
            client=client,
        )

    assert response is not None
    assert response["id"] == "resp_2"
    assert str(captured[0].url) == codex_direct.CHATGPT_CODEX_RESPONSES_URL
    assert captured[0].headers["authorization"] == "Bearer chatgpt-oauth"
    assert captured[0].headers["chatgpt-account-id"] == "acct_123"
    oauth_payload = json.loads(captured[0].content)
    assert oauth_payload["stream"] is True
    assert "max_output_tokens" not in oauth_payload


@pytest.mark.asyncio
async def test_responses_call_declines_non_codex_and_missing_oauth_headers() -> None:
    assert (
        await codex_direct.responses_call(
            body=_body(),
            tier_cfg={"model": "deepseek/deepseek-v4-flash"},
            credential="openai-key",
        )
        is None
    )
    assert (
        await codex_direct.responses_call(
            body=_body(),
            tier_cfg=_tier(),
            credential="oauth",
        )
        is None
    )


@pytest.mark.asyncio
async def test_responses_call_provider_error_and_transport_error_return_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_status(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": "bad",
                "authorization": "Bearer provider-secret",
                "detail": "x" * 3000,
            },
        )

    def failing_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing_status)) as client:
        assert (
            await codex_direct.responses_call(
                body=_body(),
                tier_cfg=_tier(),
                credential="openai-key",
                client=client,
            )
            is None
        )
    text = caplog.text
    assert "body=" in text
    assert '"error":"bad"' in text or '"error": "bad"' in text
    assert "provider-secret" not in text
    assert "<truncated>" in text
    async with httpx.AsyncClient(transport=httpx.MockTransport(failing_transport)) as client:
        assert (
            await codex_direct.responses_call(
                body=_body(),
                tier_cfg=_tier(),
                credential="openai-key",
                client=client,
            )
            is None
        )


@pytest.mark.asyncio
async def test_responses_call_non_object_json_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["bad"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (
            await codex_direct.responses_call(
                body=_body(),
                tier_cfg=_tier(),
                credential="openai-key",
                client=client,
            )
            is None
        )


def test_response_escalates_on_premium_tool_output() -> None:
    response = {
        "output": [
            {"type": "function_call", "name": "apply_patch"},
            {"type": "function_call", "name": "exec_command"},
        ]
    }

    assert codex_direct.response_escalates(response, ["apply_patch"]) == ["apply_patch"]
