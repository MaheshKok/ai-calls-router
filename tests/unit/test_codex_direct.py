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


def test_prepare_responses_body_keeps_reasoning_drops_state_without_mutating() -> None:
    body = _body()

    routed = codex_direct.prepare_responses_body(body, _tier(max_tokens=2000))

    assert routed["model"] == "gpt-5-codex-spark"
    assert routed["max_output_tokens"] == 2000
    # Reasoning precedes its function_call and is required to pair tool output;
    # the stateless ChatGPT Codex backend rejects previous_response_id, so it
    # is dropped while the input items are replayed in full.
    assert routed["input"] == [
        {"type": "reasoning", "encrypted_content": "secret"},
        {"type": "function_call_output", "call_id": "call_1", "output": "done"},
    ]
    assert "previous_response_id" not in routed
    assert "stream" not in routed
    assert routed["input"] is not body["input"]
    assert body["input"][0]["encrypted_content"] == "secret"
    assert body["previous_response_id"] == "resp_old"


def test_unpaired_function_call_ids_flags_only_unanswered_calls() -> None:
    # Paired call+output is safe; an unanswered call is the 400 trigger.
    assert codex_direct.unpaired_function_call_ids({"input": "hello"}) == []
    assert (
        codex_direct.unpaired_function_call_ids(
            {
                "input": [
                    {"type": "function_call", "call_id": "call_a", "name": "x", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
                ]
            }
        )
        == []
    )
    # Output without its call is a different shape and not this guard's concern.
    assert (
        codex_direct.unpaired_function_call_ids(
            {"input": [{"type": "function_call_output", "call_id": "call_a", "output": "ok"}]}
        )
        == []
    )
    assert codex_direct.unpaired_function_call_ids(
        {
            "input": [
                {"type": "function_call", "call_id": "call_b", "name": "x", "arguments": "{}"},
                {"type": "custom_tool_call", "call_id": "call_a", "name": "y", "input": ""},
            ]
        }
    ) == ["call_a", "call_b"]


@pytest.mark.asyncio
async def test_responses_call_declines_when_input_has_unanswered_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("orphaned function_call must not reach the stateless backend")

    body = {
        "model": "gpt-5-codex",
        "input": [
            {"type": "function_call", "call_id": "call_x", "name": "exec", "arguments": "{}"},
        ],
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (
            await codex_direct.responses_call(
                body=body,
                tier_cfg=_tier(),
                credential="openai-key",
                client=client,
            )
            is None
        )


def test_prepare_responses_body_handles_string_input_and_plain_model() -> None:
    routed = codex_direct.prepare_responses_body(
        {"model": "gpt-5-codex", "input": "hello"},
        {"model": "gpt-5-codex", "provider": "codex"},
    )

    assert routed["model"] == "gpt-5-codex"
    assert routed["input"] == "hello"


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
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 3},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await codex_direct.responses_call(
            body=_body(),
            tier_cfg=_tier(),
            credential="openai-key",
            chatgpt_headers=[("Authorization", "Bearer client-oauth")],
            client=client,
        )

    assert result is not None
    _response, usage, shrink = result
    assert str(captured[0].url) == codex_direct.OPENAI_RESPONSES_URL
    assert captured[0].headers["authorization"] == "Bearer openai-key"
    assert "client-oauth" not in json.dumps(dict(captured[0].headers))
    # Encrypted reasoning is replayed to its issuing provider, not a third party.
    assert "secret" in captured[0].content.decode()
    assert "stream" not in json.loads(captured[0].content)
    assert "previous_response_id" not in json.loads(captured[0].content)
    assert usage == (2, 2, 3, 0)
    assert shrink.chars_before == len("done")
    assert shrink.chars_after == len("done")


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
        result = await codex_direct.responses_call(
            body=_body(),
            tier_cfg=_tier(),
            credential="oauth",
            auth_mode="oauth",
            chatgpt_headers=[
                ("Authorization", "Bearer chatgpt-oauth"),
                ("ChatGPT-Account-ID", "acct_123"),
            ],
            client=client,
        )

    assert result is not None
    response, usage, shrink = result
    assert response["id"] == "resp_2"
    assert usage == (0, 0, 0, 0)
    assert shrink.chars_before == len("done")
    assert shrink.chars_after == len("done")
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
            auth_mode="oauth",
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
