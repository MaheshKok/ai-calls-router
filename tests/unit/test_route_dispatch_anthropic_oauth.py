"""Tests for the Anthropic subscription-OAuth dispatch route.

try_anthropic_oauth_route gates a claude_code cheap turn to api.anthropic.com on
the client's OAuth bearer, runs the premium-escalation guard and routed
accounting, and masks the served model back to the client-requested one. These
tests cover gating, the success path, escalation, and fail-open passthrough
with the upstream call and accounting stubbed.
"""

from __future__ import annotations

import pytest

from ai_calls_router.accounting import shrink_stats
from ai_calls_router.proxy import route_dispatch
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing.adapters import adapter_for_path


def _adapter() -> object:
    """Return the Anthropic Messages client adapter."""
    adapter = adapter_for_path("/v1/messages")
    assert adapter is not None
    return adapter


def _decision(**overrides: object) -> route_dispatch.RouteDecision:
    """Return a routable Anthropic-OAuth decision for a Sonnet cheap tier."""
    base: dict[str, object] = {
        "body": {"model": "claude-opus-4-8"},
        "anthropic_body": {"model": "claude-opus-4-8", "messages": []},
        "requested_model": "claude-opus-4-8",
        "streaming": False,
        "names": ["Bash"],
        "tier": "fast",
        "tier_cfg": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "max_tokens": 8192,
        },
        "credential": routing.TierCredential(value="oauth", auth_mode="oauth"),
        "settings": {},
        "premium_tools": ["Edit"],
    }
    base.update(overrides)
    return route_dispatch.RouteDecision(**base)  # type: ignore[arg-type]


def _shrink() -> shrink_stats.ShrinkStats:
    """Return a no-op shrink record."""
    return shrink_stats.compute_shrink(path="none", before={}, after={})


def _text_response() -> dict[str, object]:
    """Return an Anthropic response with no tool calls."""
    return {
        "id": "msg_1",
        "type": "message",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def _edit_response() -> dict[str, object]:
    """Return an Anthropic response that calls the premium Edit tool."""
    return {
        "id": "msg_2",
        "type": "message",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}],
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }


def _stub_messages_call(
    monkeypatch: pytest.MonkeyPatch, *, result: object, captured: dict[str, object] | None = None
) -> None:
    """Patch anthropic_oauth.messages_call to return a fixed result."""

    async def _fake(**kwargs: object) -> object:
        if captured is not None:
            captured.update(kwargs)
        return result

    monkeypatch.setattr(route_dispatch.anthropic_oauth, "messages_call", _fake)


def _stub_accounting(monkeypatch: pytest.MonkeyPatch, sink: list[object]) -> None:
    """Patch record_route_outcome so accounting does not touch disk."""

    async def _record(outcome: object) -> None:
        sink.append(outcome)

    monkeypatch.setattr(route_dispatch.routed_call, "record_route_outcome", _record)


async def _run(
    decision: route_dispatch.RouteDecision, **over: object
) -> route_dispatch.RouteAttempt | None:
    """Invoke try_anthropic_oauth_route with sane defaults."""
    kwargs: dict[str, object] = {
        "adapter": _adapter(),
        "request_headers": {"authorization": "Bearer client-oauth"},
        "request_path": "/v1/messages",
        "user_agent": "claude-cli",
        "agent": "claude_code",
        "session": "sess",
        "client": None,
    }
    kwargs.update(over)
    return await route_dispatch.try_anthropic_oauth_route(decision, **kwargs)


async def test_returns_none_when_credential_is_not_oauth() -> None:
    decision = _decision(credential=routing.TierCredential(value="k", auth_mode="api_key"))
    assert await _run(decision) is None


async def test_returns_none_when_path_is_not_messages() -> None:
    assert await _run(_decision(), request_path="/v1/responses") is None


async def test_returns_none_for_non_anthropic_oauth_tier() -> None:
    # A Codex OAuth tier must fall through to the Codex dispatch path.
    decision = _decision(tier_cfg={"provider": "codex", "model": "codex/gpt-5-codex-spark"})
    assert await _run(decision) is None


async def test_success_routes_records_and_masks_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    _stub_messages_call(
        monkeypatch, result=(_text_response(), (10, 4, 0, 0), _shrink()), captured=captured
    )
    outcomes: list[object] = []
    _stub_accounting(monkeypatch, outcomes)

    attempt = await _run(_decision())

    assert attempt is not None
    assert attempt.reason == "routed"
    assert attempt.model == "claude-sonnet-4-6"
    assert attempt.premium_model == "claude-opus-4-8"
    assert attempt.response is not None
    assert attempt.input_tokens == 10
    assert attempt.output_tokens == 4
    # The routed turn forwards the OAuth headers and the Anthropic body.
    assert captured["oauth_headers"]["authorization"] == "Bearer client-oauth"
    assert captured["body"] == {"model": "claude-opus-4-8", "messages": []}
    # Accounting is recorded under the true routed model on the OAuth route.
    assert len(outcomes) == 1
    assert outcomes[0].route == "anthropic_oauth"
    assert outcomes[0].routed_model == "claude-sonnet-4-6"
    assert outcomes[0].premium_model == "claude-opus-4-8"


async def test_escalates_to_passthrough_when_response_calls_premium_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_messages_call(monkeypatch, result=(_edit_response(), (5, 1, 0, 0), _shrink()))
    outcomes: list[object] = []
    _stub_accounting(monkeypatch, outcomes)

    attempt = await _run(_decision())

    assert attempt is not None
    assert attempt.response is None
    assert attempt.reason == "response_premium_guard"
    assert attempt.model == "claude-opus-4-8"
    # An escalating turn is replayed on premium, so nothing is billed as routed.
    assert outcomes == []


async def test_fails_open_to_passthrough_when_upstream_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_messages_call(monkeypatch, result=None)
    outcomes: list[object] = []
    _stub_accounting(monkeypatch, outcomes)

    attempt = await _run(_decision())

    assert attempt is not None
    assert attempt.response is None
    assert attempt.reason == "routed_fallback"
    assert attempt.model == "claude-opus-4-8"
    assert outcomes == []
