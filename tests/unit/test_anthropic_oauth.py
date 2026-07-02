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
                "Cookie": "session=secret",
                "X-Forwarded-For": "203.0.113.7",
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
    # Allowlist contract: client headers outside the OAuth set never reach
    # api.anthropic.com, so unrelated/sensitive headers cannot leak upstream.
    assert "cookie" not in captured[0].headers
    assert "x-forwarded-for" not in captured[0].headers


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


def _passthrough_body(model: str) -> bytes:
    """Return a serialized Anthropic Messages body with the given model id."""
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()


def test_strip_long_context_passthrough_drops_both_markers() -> None:
    body = _passthrough_body("claude-opus-4-8[1m]")
    headers = {"anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07", "x-keep": "1"}

    result = anthropic_oauth.strip_long_context_passthrough(body, headers)

    assert result is not None
    new_body, new_headers = result
    assert json.loads(new_body)["model"] == "claude-opus-4-8"
    assert new_headers["anthropic-beta"] == "oauth-2025-04-20"
    assert new_headers["x-keep"] == "1"


def test_strip_long_context_passthrough_header_only_keeps_body_bytes() -> None:
    body = _passthrough_body("claude-opus-4-8")
    headers = {"anthropic-beta": "context-1m-2025-08-07,tool-streaming"}

    result = anthropic_oauth.strip_long_context_passthrough(body, headers)

    assert result is not None
    new_body, new_headers = result
    assert new_body == body  # model had no suffix, so the body relays byte-identical
    assert new_headers["anthropic-beta"] == "tool-streaming"


def test_strip_long_context_passthrough_drops_empty_beta_header() -> None:
    result = anthropic_oauth.strip_long_context_passthrough(
        _passthrough_body("claude-opus-4-8"), {"anthropic-beta": "context-1m-2025-08-07"}
    )

    assert result is not None
    _new_body, new_headers = result
    assert "anthropic-beta" not in new_headers


def test_strip_long_context_passthrough_model_suffix_without_beta() -> None:
    result = anthropic_oauth.strip_long_context_passthrough(
        _passthrough_body("claude-opus-4-8[1m]"), {}
    )

    assert result is not None
    new_body, new_headers = result
    assert json.loads(new_body)["model"] == "claude-opus-4-8"
    assert new_headers == {}


def test_strip_long_context_passthrough_header_key_is_case_insensitive() -> None:
    result = anthropic_oauth.strip_long_context_passthrough(
        _passthrough_body("claude-opus-4-8"), {"Anthropic-Beta": "context-1m-2025-08-07"}
    )

    assert result is not None
    _new_body, new_headers = result
    assert "Anthropic-Beta" not in new_headers


def test_strip_long_context_passthrough_no_markers_returns_none() -> None:
    result = anthropic_oauth.strip_long_context_passthrough(
        _passthrough_body("claude-opus-4-8"), {"anthropic-beta": "oauth-2025-04-20"}
    )

    assert result is None


def test_strip_long_context_passthrough_suffix_must_be_trailing() -> None:
    # A [1m] that is not the trailing token is not the long-context variant.
    result = anthropic_oauth.strip_long_context_passthrough(
        _passthrough_body("claude-[1m]-experimental"), {}
    )

    assert result is None


def test_strip_long_context_passthrough_non_json_body_with_beta() -> None:
    result = anthropic_oauth.strip_long_context_passthrough(
        b"not-json", {"anthropic-beta": "context-1m-2025-08-07"}
    )

    assert result is not None
    new_body, new_headers = result
    assert new_body == b"not-json"  # unparseable body relays verbatim
    assert "anthropic-beta" not in new_headers


def test_strip_long_context_passthrough_non_json_body_without_beta_returns_none() -> None:
    assert anthropic_oauth.strip_long_context_passthrough(b"not-json", {}) is None


def test_strip_long_context_passthrough_does_not_mutate_input_headers() -> None:
    headers = {"anthropic-beta": "context-1m-2025-08-07,tool-streaming"}
    body = _passthrough_body("claude-opus-4-8[1m]")

    anthropic_oauth.strip_long_context_passthrough(body, headers)

    assert headers == {"anthropic-beta": "context-1m-2025-08-07,tool-streaming"}


@pytest.mark.asyncio
async def test_messages_call_compresses_routed_payload_before_post_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # compress=True must run the routed payload through compress_anthropic and
    # POST the compressed body upstream, returning that pass's shrink stat.
    seen: list[dict[str, object]] = []

    def fake_compress(
        body: dict[str, object], *, enable_text_ml: bool = False
    ) -> tuple[dict[str, object], anthropic_oauth.shrink_stats.ShrinkStats]:
        seen.append({"model": body.get("model"), "enable_text_ml": enable_text_ml})
        compressed = {**body, "_compressed": True}
        return compressed, anthropic_oauth.shrink_stats.ShrinkStats("compress", 120, 40)

    monkeypatch.setattr(anthropic_oauth, "compress_anthropic", fake_compress)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={},
            compress=True,
            client=client,
        )

    assert result is not None
    _response, _usage, shrink = result
    # The returned stat is the compress pass's, not a no-op.
    assert shrink.path == "compress"
    assert shrink.chars_before == 120
    assert shrink.chars_after == 40
    # Compression saw the already-model-swapped routed payload.
    assert seen == [{"model": "claude-sonnet-4-6", "enable_text_ml": False}]
    # The compressed body — not the original — is what reached api.anthropic.com.
    assert json.loads(captured[0].content)["_compressed"] is True


@pytest.mark.asyncio
async def test_messages_call_does_not_compress_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # compress=False (the default) must never invoke the compressor and must
    # report a no-op shrink so the payload is forwarded verbatim.
    def fail_compress(
        body: dict[str, object], *, enable_text_ml: bool = False
    ) -> tuple[dict[str, object], anthropic_oauth.shrink_stats.ShrinkStats]:
        pytest.fail("compress_anthropic must not run when compress is False")

    monkeypatch.setattr(anthropic_oauth, "compress_anthropic", fail_compress)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await anthropic_oauth.messages_call(
            body=_body(), tier_cfg=_tier(), oauth_headers={}, client=client
        )

    assert result is not None
    _response, _usage, shrink = result
    assert shrink.path == "none"
    assert "_compressed" not in json.loads(captured[0].content)


@pytest.mark.asyncio
async def test_messages_call_forwards_text_ml_flag_to_compressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-tier text-ML opt-in must reach compress_anthropic unchanged.
    seen: list[bool] = []

    def fake_compress(
        body: dict[str, object], *, enable_text_ml: bool = False
    ) -> tuple[dict[str, object], anthropic_oauth.shrink_stats.ShrinkStats]:
        seen.append(enable_text_ml)
        return body, anthropic_oauth.shrink_stats.ShrinkStats("none", 0, 0)

    monkeypatch.setattr(anthropic_oauth, "compress_anthropic", fake_compress)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_ok_response()))
    async with httpx.AsyncClient(transport=transport) as client:
        await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={},
            compress=True,
            enable_text_ml=True,
            client=client,
        )

    assert seen == [True]


def test_forward_headers_strips_long_context_beta_by_default() -> None:
    # Default: a routed cheap turn must not opt into 1M context (the subscription
    # 429s it), so context-1m is dropped while the other beta tokens survive.
    headers = anthropic_oauth._forward_headers(
        {"anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07"}
    )

    assert headers["anthropic-beta"] == "oauth-2025-04-20"


def test_forward_headers_keeps_long_context_beta_when_enabled() -> None:
    # keep_long_context=True (an entitled tier) forwards context-1m intact so the
    # routed model serves its full 1M window.
    headers = anthropic_oauth._forward_headers(
        {"anthropic-beta": "oauth-2025-04-20,context-1m-2025-08-07"},
        keep_long_context=True,
    )

    assert headers["anthropic-beta"] == "oauth-2025-04-20,context-1m-2025-08-07"


def test_forward_headers_keep_long_context_preserves_sole_beta_token() -> None:
    # When context-1m is the only beta token, the default drops the header
    # entirely; keeping it must leave the header present and unchanged.
    stripped = anthropic_oauth._forward_headers({"anthropic-beta": "context-1m-2025-08-07"})
    kept = anthropic_oauth._forward_headers(
        {"anthropic-beta": "context-1m-2025-08-07"}, keep_long_context=True
    )

    assert "anthropic-beta" not in stripped
    assert kept["anthropic-beta"] == "context-1m-2025-08-07"


def test_forward_headers_drops_non_allowlisted_headers_regardless_of_flag() -> None:
    # The allowlist is independent of keep_long_context: unrelated client headers
    # never reach upstream even when long context is kept.
    headers = anthropic_oauth._forward_headers(
        {"authorization": "Bearer x", "cookie": "secret", "x-forwarded-for": "1.2.3.4"},
        keep_long_context=True,
    )

    assert headers["authorization"] == "Bearer x"
    assert "cookie" not in headers
    assert "x-forwarded-for" not in headers


@pytest.mark.asyncio
async def test_messages_call_keeps_long_context_when_tier_entitled() -> None:
    # End-to-end wiring: a tier with supports_long_context must forward the
    # context-1m beta to api.anthropic.com; a default tier must strip it.
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_ok_response())

    beta = "oauth-2025-04-20,context-1m-2025-08-07"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(supports_long_context=True),
            oauth_headers={"anthropic-beta": beta},
            client=client,
        )
        await anthropic_oauth.messages_call(
            body=_body(),
            tier_cfg=_tier(),
            oauth_headers={"anthropic-beta": beta},
            client=client,
        )

    assert captured[0].headers["anthropic-beta"] == beta
    assert captured[1].headers["anthropic-beta"] == "oauth-2025-04-20"
