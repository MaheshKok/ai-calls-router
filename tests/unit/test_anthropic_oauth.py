"""Tests for Anthropic subscription-OAuth routed calls.

The Anthropic-OAuth path serves a cheap claude_code turn on api.anthropic.com
with the client's inbound OAuth bearer so it draws the subscription's quota for
the tier model, swapping only the model. These tests exercise tier detection,
model-id normalization, header forwarding, body preparation, and fail-open
behavior without a real network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ai_calls_router.routing import anthropic_oauth


def _tier(**extra: object) -> dict[str, object]:
    """Return a Sonnet Anthropic-OAuth tier config."""
    return {
        "provider": "anthropic",
        "model": "anthropic/claude-sonnet-4-6",
        "max_tokens": 8192,
        **extra,
    }


def _body() -> dict[str, object]:
    """Return an Anthropic tool-result turn with a thinking block to strip."""
    return {
        "model": "claude-opus-4-8",
        "stream": True,
        "max_tokens": 100000,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "deliberating", "signature": "sig"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ],
    }


def _ok_response() -> dict[str, object]:
    """Return a minimal completed Anthropic response with usage."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    }


@pytest.mark.parametrize(
    ("tier_cfg", "expected"),
    [
        ({"provider": "anthropic", "model": "anthropic/claude-sonnet-4-6"}, True),
        ({"model": "anthropic/claude-sonnet-4-6"}, True),
        ({"provider": "anthropic", "model": "claude-sonnet-4-6"}, True),
        ({"provider": "deepseek", "model": "deepseek/deepseek-v4-flash"}, False),
        ({"provider": "codex", "model": "codex/gpt-5-codex-spark"}, False),
        ({"model": "gpt-5"}, False),
        ({}, False),
    ],
)
def test_is_anthropic_oauth_tier(tier_cfg: dict[str, object], expected: bool) -> None:
    assert anthropic_oauth.is_anthropic_oauth_tier(tier_cfg) is expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("anthropic/anthropic/x", "anthropic/x"),
    ],
)
def test_native_model_id_strips_only_one_prefix(model: str, expected: str) -> None:
    assert anthropic_oauth.native_model_id({"model": model}) == expected


@pytest.mark.asyncio
async def test_messages_call_swaps_model_strips_thinking_and_forwards_oauth() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={
                "Authorization": "Bearer client-oauth",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "Content-Type": "text/plain",
            },
            client=client,
        )

    assert result is not None
    response, usage, shrink = result
    assert response["id"] == "msg_1"
    assert usage == (10, 4, 3, 2)
    # No compression on this path: the shrink pass is a no-op.
    assert shrink.path == "none"
    assert shrink.chars_before == shrink.chars_after

    assert str(captured[0].url) == anthropic_oauth.ANTHROPIC_MESSAGES_URL
    sent = json.loads(captured[0].content)
    # Model swapped to the bare native id, never the anthropic/ prefixed form.
    assert sent["model"] == "claude-sonnet-4-6"
    # Routed calls are buffered: the stream flag is dropped.
    assert "stream" not in sent
    # max_tokens clamped down to the tier ceiling.
    assert sent["max_tokens"] == 8192
    # Thinking block stripped; the tool_use survives so the turn still resolves.
    assert sent["messages"][0]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
    ]
    # The subscription bearer is forwarded verbatim so the quota is the user's.
    assert captured[0].headers["authorization"] == "Bearer client-oauth"
    assert captured[0].headers["anthropic-version"] == "2023-06-01"
    assert captured[0].headers["anthropic-beta"] == "oauth-2025-04-20"
    assert captured[0].headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_messages_call_downgrades_xhigh_effort_for_sonnet() -> None:
    # Claude Code forwards output_config.effort='xhigh', which Sonnet rejects
    # (HTTP 400); the routed payload must downgrade it to a supported level.
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    body = _body()
    body["output_config"] = {"effort": "xhigh"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await anthropic_oauth.messages_call(
            body=body, tier_cfg=_tier(), oauth_headers={}, client=client
        )

    sent = json.loads(captured[0].content)
    assert sent["output_config"]["effort"] == "high"
    # The client body is left untouched.
    assert body["output_config"]["effort"] == "xhigh"


@pytest.mark.asyncio
async def test_messages_call_applies_tier_effort_and_logs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A tier "effort" pins the routed reasoning level, overriding the client's
    # xhigh, and the level actually sent upstream is logged for observability.
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    body = _body()
    body["output_config"] = {"effort": "xhigh"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("INFO", logger="acr.anthropic_oauth"):
            await anthropic_oauth.messages_call(
                body=body, tier_cfg=_tier(effort="low"), oauth_headers={}, client=client
            )

    assert json.loads(captured[0].content)["output_config"]["effort"] == "low"
    assert "effort=low" in caplog.text
    assert "model=claude-sonnet-4-6" in caplog.text


@pytest.mark.asyncio
async def test_messages_call_strips_long_context_beta_keeps_others() -> None:
    # opus[1m] forwards context-1m, which the subscription rejects for the routed
    # model (HTTP 429); the routed call must drop it but keep the OAuth beta.
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={
                "Authorization": "Bearer client-oauth",
                "anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07,tool-streaming",
            },
            client=client,
        )

    assert captured[0].headers["anthropic-beta"] == "oauth-2025-04-20,tool-streaming"


@pytest.mark.asyncio
async def test_messages_call_drops_beta_header_when_only_long_context() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={"anthropic-beta": "context-1m-2025-08-07"},
            client=client,
        )

    assert "anthropic-beta" not in captured[0].headers


@pytest.mark.asyncio
async def test_messages_call_does_not_mutate_input_body() -> None:
    body = _body()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await anthropic_oauth.messages_call(
            body=body, tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert body["model"] == "claude-opus-4-8"
    assert body["stream"] is True
    assert body["max_tokens"] == 100000
    assert body["messages"][0]["content"][0]["type"] == "thinking"


@pytest.mark.asyncio
async def test_messages_call_returns_none_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is None


@pytest.mark.asyncio
async def test_messages_call_returns_none_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is None


@pytest.mark.asyncio
async def test_messages_call_declines_non_anthropic_tier_without_calling_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a non-Anthropic tier must not reach api.anthropic.com")

    deepseek_tier = {"provider": "deepseek", "model": "deepseek/deepseek-v4-flash"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=deepseek_tier, oauth_headers={}, client=client
        )

    assert result is None


@pytest.mark.asyncio
async def test_messages_call_returns_none_on_non_json_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"Content-Type": "text/plain"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is None


@pytest.mark.asyncio
async def test_messages_call_returns_none_on_non_object_json_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "shape"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is None


@pytest.mark.asyncio
async def test_messages_call_defaults_usage_to_zero_when_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_3", "content": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is not None
    _response, usage, _shrink = result
    assert usage == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_messages_call_creates_and_closes_its_own_client_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no shared client, the call must build a transient client and close it.
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_ok_response()))
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        anthropic_oauth.httpx,
        "AsyncClient",
        lambda **_kwargs: real_async_client(transport=transport),
    )

    result = await anthropic_oauth.messages_call(body=_body(), tier_cfg=_tier(), oauth_headers={})

    assert result is not None
    _response, usage, _shrink = result
    assert usage == (10, 4, 3, 2)
