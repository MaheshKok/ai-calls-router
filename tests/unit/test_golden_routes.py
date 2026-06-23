"""Golden routing decisions for representative supported requests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from ai_calls_router._lib.types import JsonObject
from ai_calls_router.proxy import route_dispatch
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import provider_config
from ai_calls_router.routing.adapters import adapter_for_path


@dataclass(frozen=True)
class GoldenRoute:
    """One routing decision fixture and its expected outcome."""

    name: str
    path: str
    headers: Mapping[str, str]
    body: JsonObject
    expected_group: str
    expected_tools: list[str]
    expected_tier: str
    expected_model: str | None
    expected_auth_mode: str | None


def _routes() -> JsonObject:
    """Return the fixed config used by the golden route table."""
    return {
        "settings": {
            "tier_precedence": ["premium", "structured", "code", "fast", "crud", "no_key"]
        },
        "tiers": {
            "fast": {"model": "deepseek/deepseek-chat", "key_env": "FAST_KEY"},
            "code": {"model": "deepseek/deepseek-coder", "key_env": "CODE_KEY"},
            "crud": {"model": "deepseek/deepseek-crud", "key_env": "CRUD_KEY"},
            "structured": {
                "model": "deepseek/deepseek-structured",
                "key_env": "STRUCTURED_KEY",
            },
            "no_key": {"model": "deepseek/no-key"},
        },
        "router": {
            "endpoint_defaults": {
                "/v1/messages": "claude_code",
                "/v1/chat/completions": "hermes",
            },
            "fallback": None,
        },
        "agents": {
            "claude_code": {
                "tools": {
                    "Bash": "fast",
                    "Read": "code",
                    "TodoWrite": "crud",
                    "NoKey": "no_key",
                    "Edit": "premium",
                },
                "premium_tools": ["Edit"],
                "upstream": "https://api.anthropic.com",
            },
            "hermes": {
                "tools": {
                    "read_file": "code",
                    "write_file": "structured",
                    "todo": "crud",
                    "patch": "premium",
                },
                "premium_tools": ["patch"],
                "upstream": "https://api.openai.com",
            },
        },
    }


def _anthropic_tool_result(tool_name: str) -> JsonObject:
    """Build a Claude-style completed tool-result turn."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ],
    }


def _chat_tool_result(tool_name: str) -> JsonObject:
    """Build an OpenAI Chat Completions completed tool-result turn."""
    return {
        "model": "gpt-5",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }


GOLDEN_ROUTES = [
    GoldenRoute(
        name="claude-fast",
        path="/v1/messages",
        headers={},
        body=_anthropic_tool_result("Bash"),
        expected_group="claude_code",
        expected_tools=["Bash"],
        expected_tier="fast",
        expected_model="deepseek/deepseek-chat",
        expected_auth_mode="api_key",
    ),
    GoldenRoute(
        name="claude-code",
        path="/v1/messages",
        headers={},
        body=_anthropic_tool_result("Read"),
        expected_group="claude_code",
        expected_tools=["Read"],
        expected_tier="code",
        expected_model="deepseek/deepseek-coder",
        expected_auth_mode="api_key",
    ),
    GoldenRoute(
        name="claude-crud",
        path="/v1/messages",
        headers={},
        body=_anthropic_tool_result("TodoWrite"),
        expected_group="claude_code",
        expected_tools=["TodoWrite"],
        expected_tier="crud",
        expected_model="deepseek/deepseek-crud",
        expected_auth_mode="api_key",
    ),
    GoldenRoute(
        name="hermes-structured",
        path="/v1/chat/completions",
        headers={},
        body=_chat_tool_result("write_file"),
        expected_group="hermes",
        expected_tools=["write_file"],
        expected_tier="structured",
        expected_model="deepseek/deepseek-structured",
        expected_auth_mode="api_key",
    ),
    GoldenRoute(
        name="claude-premium-guard",
        path="/v1/messages",
        headers={},
        body=_anthropic_tool_result("Edit"),
        expected_group="claude_code",
        expected_tools=["Edit"],
        expected_tier="premium",
        expected_model=None,
        expected_auth_mode=None,
    ),
    GoldenRoute(
        name="claude-missing-credential-fallback",
        path="/v1/messages",
        headers={},
        body=_anthropic_tool_result("NoKey"),
        expected_group="claude_code",
        expected_tools=["NoKey"],
        expected_tier="no_key",
        expected_model=None,
        expected_auth_mode=None,
    ),
]


@pytest.mark.parametrize("case", GOLDEN_ROUTES, ids=[case.name for case in GOLDEN_ROUTES])
def test_golden_route_decisions(monkeypatch: pytest.MonkeyPatch, case: GoldenRoute) -> None:
    routes = _routes()
    for key in ("FAST_KEY", "CODE_KEY", "CRUD_KEY", "STRUCTURED_KEY"):
        monkeypatch.setenv(key, f"value-for-{key}")
    adapter = adapter_for_path(case.path)
    assert adapter is not None
    group = provider_config.resolve_agent_group(
        path=case.path,
        headers=case.headers,
        routes=routes,
        adapter_default=adapter.default_agent_group,
    )
    assert group == case.expected_group
    names = adapter.extract_pending_tools(case.body)
    tier, tier_cfg, credential, _ = route_dispatch.resolve_tier_config(
        names,
        group=group,
        routes_loader=lambda: routes,
    )

    assert names == case.expected_tools
    assert tier == case.expected_tier
    assert (tier_cfg.get("model") if tier_cfg is not None else None) == case.expected_model
    assert (credential.auth_mode if credential is not None else None) == case.expected_auth_mode


def test_try_route_dispatch_error_falls_back_preserving_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch-stage exception fails open to routing_error, keeping the model.

    The decision stage succeeds, so ``try_route`` must catch a failure raised by
    the native/OAuth dispatch path and return a passthrough attempt that still
    carries the resolved model and tools (so the caller records them).
    """
    decision = route_dispatch.RouteDecision(
        body={"model": "claude-fable-5"},
        anthropic_body={"model": "claude-fable-5"},
        requested_model="claude-fable-5",
        streaming=False,
        names=["Bash"],
        tier="fast",
        tier_cfg={"model": "deepseek/deepseek-chat"},
        credential=routing.TierCredential(value="secret", auth_mode="api_key"),
        settings={},
        premium_tools=[],
    )
    monkeypatch.setattr(route_dispatch, "prepare_route", lambda *_a, **_k: decision)

    async def _boom(*_a: object, **_k: object) -> route_dispatch.RouteAttempt | None:
        raise RuntimeError("dispatch exploded")

    monkeypatch.setattr(route_dispatch, "try_native_or_oauth_route", _boom)

    attempt = asyncio.run(
        route_dispatch.try_route(
            b"{}",
            adapter=object(),  # type: ignore[arg-type]  # unused: dispatch is patched to raise
            group="claude_code",
            request_path="/v1/messages",
            request_headers={},
            routes_loader=dict,
        )
    )

    assert attempt.reason == "routing_error"
    assert attempt.model == "claude-fable-5"
    assert attempt.tool_names == ["Bash"]
    assert attempt.response is None
