"""Contract tests for Phase 7 provider-file routing config.

These tests derive expectations from the provider YAML routing spec: pure
assembly builds the canonical agents dict without mutating the base config,
provider payloads never carry cheap-tier credentials, and identity resolution
uses the documented precedence order.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing import provider_config
from ai_calls_router.routing.adapters.base import KNOWN_GROUPS
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    parse_tier_config,
    validate_routes_payload,
)


def _base_routes(*, router: dict[str, object] | None = None) -> dict[str, object]:
    """Return a minimal global config for provider assembly tests."""
    base: dict[str, object] = {
        "server": {"upstream": "https://premium.default.example/"},
        "settings": {"tier_precedence": ["premium", "fast", "crud"]},
        "tiers": {"fast": {"model": "deepseek/x", "key_env": "CHEAP_KEY"}},
    }
    if router is not None:
        base["router"] = router
    return base


def _router() -> dict[str, object]:
    """Return a strict router block used by Phase 7 configs."""
    return {
        "endpoint_defaults": {
            "/v1/messages": "claude_code",
            "/v1/chat/completions": "hermes",
        },
        "user_agent_map": [
            {"contains": "claude", "group": "claude_code"},
            {"contains": "hermes", "group": "hermes"},
        ],
        "fallback": None,
    }


def _provider_payload(group: str, *, upstream: str | None = None) -> dict[str, object]:
    """Return one valid provider YAML payload."""
    wires = {
        "claude_code": "anthropic_messages",
        "hermes": "openai_chat",
    }
    endpoints = {
        "claude_code": ["/v1/messages"],
        "hermes": ["/v1/chat/completions", "/v1/responses"],
    }
    return {
        "group": group,
        "upstream": upstream or f"https://{group}.example",
        "auth": {"mode": "oauth_passthrough"},
        "wire": wires[group],
        "endpoints": endpoints[group],
        "tools": {"tool": "fast"},
        "premium_tools": [],
        "fallback": "passthrough",
    }


def test_provider_files_merge_to_hand_written_agents_block() -> None:
    base = _base_routes(router=_router())
    provider_files = {group: _provider_payload(group) for group in KNOWN_GROUPS}
    assembled = provider_config.assemble_routes(base, provider_files=provider_files)

    expected_agents = {
        group: {key: copy.deepcopy(value) for key, value in payload.items() if key != "group"}
        for group, payload in provider_files.items()
    }
    expected = copy.deepcopy(base)
    expected["agents"] = expected_agents
    assert assembled == expected


def test_oauth_hermes_tiers_pass_routes_schema_validation() -> None:
    # Regression: ChatGPT-OAuth Hermes tiers (auth.mode: oauth, no key_env) must
    # validate. Rejecting them is the bug that disabled routing at startup with
    # "config schema validation failed (Input should be 'api_key_env')".
    routes = {
        "settings": {"tier_precedence": ["premium", "structured", "code", "fast", "crud"]},
        "agents": {
            "hermes": {
                "upstream": "https://chatgpt.com/backend-api/codex",
                "tools": {"terminal": "fast", "patch": "premium"},
                "premium_tools": ["patch"],
                "tiers": {
                    "fast": {
                        "provider": "codex",
                        "model": "gpt-5.4-mini",
                        "auth": {"mode": "oauth"},
                        "max_tokens": 8192,
                    },
                    "code": {
                        "provider": "codex",
                        "model": "gpt-5.3-codex-spark",
                        "auth": {"mode": "oauth"},
                        "max_tokens": 8192,
                    },
                },
            },
        },
    }
    validate_routes_payload(routes)


def test_api_key_env_tier_without_key_env_still_validates() -> None:
    routes = {
        "agents": {
            "claude_code": {
                "tiers": {
                    "fast": {
                        "provider": "deepseek",
                        "model": "deepseek/deepseek-v4-flash",
                        "auth": {"mode": "api_key_env", "key_env": "DEEPSEEK_API_KEY"},
                    }
                }
            }
        }
    }
    validate_routes_payload(routes)


def test_unknown_auth_mode_is_rejected_by_routes_schema() -> None:
    routes = {
        "agents": {
            "hermes": {
                "tiers": {"fast": {"model": "gpt-5.4-mini", "auth": {"mode": "session_token"}}}
            }
        }
    }
    with pytest.raises(ConfigSchemaError):
        validate_routes_payload(routes)


def test_cheap_key_env_provider_payload_is_rejected() -> None:
    payload = _provider_payload("hermes")
    payload["tiers"] = {"fast": {"key_env": "MUST_NOT_MOVE_HERE"}}

    with pytest.raises(provider_config.ProviderConfigError):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"hermes": payload},
        )


def test_provider_payload_tool_map_must_be_string_to_string() -> None:
    payload = _provider_payload("hermes")
    payload["tools"] = {"terminal": 7}

    with pytest.raises(provider_config.ProviderConfigError):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"hermes": payload},
        )


def test_provider_payload_upstream_must_be_https_with_host() -> None:
    payload = _provider_payload("hermes", upstream="http://127.0.0.1:8747")

    with pytest.raises(provider_config.ProviderConfigError):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"hermes": payload},
        )


def test_missing_provider_file_falls_back_without_dropping_present_groups() -> None:
    assembled = provider_config.assemble_routes(
        _base_routes(router=_router()),
        provider_files={"claude_code": _provider_payload("claude_code")},
    )

    assert assembled["agents"]["hermes"]["tools"]["terminal"] == "fast"
    assert assembled["agents"]["claude_code"]["upstream"] == "https://claude_code.example"
    assert assembled["agents"]["claude_code"]["tools"]["tool"] == "fast"


def test_load_provider_files_skips_missing_and_malformed_files(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config.provider_config_dir().mkdir(parents=True)
    config.provider_config_path("claude_code").write_text(
        yaml.safe_dump(_provider_payload("claude_code")),
        encoding="utf-8",
    )
    config.provider_config_path("hermes").write_text(": bad: [", encoding="utf-8")

    loaded = provider_config.load_provider_files()

    assert set(loaded) == {"claude_code"}
    assert loaded["claude_code"]["group"] == "claude_code"
    assert "failed to load" in caplog.text


def test_load_provider_files_skips_non_mapping_yaml(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config.provider_config_dir().mkdir(parents=True)
    config.provider_config_path("hermes").write_text("- not\n- mapping\n", encoding="utf-8")

    assert provider_config.load_provider_files() == {}
    assert "is not a mapping" in caplog.text


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"group": "claude_code"}, "group mismatch"),
        ({"upstream": ""}, "requires upstream"),
        ({"wire": "garbage"}, "wire mismatch"),
        ({"endpoints": "bad"}, "requires endpoints"),
        ({"endpoints": []}, "endpoints mismatch"),
    ],
)
def test_provider_payload_required_fields_are_validated(
    updates: dict[str, object],
    message: str,
) -> None:
    payload = _provider_payload("hermes")
    payload.update(updates)

    with pytest.raises(provider_config.ProviderConfigError, match=message):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"hermes": payload},
        )


def test_provider_payload_can_omit_tool_maps_and_use_defaults() -> None:
    payload = _provider_payload("hermes")
    del payload["tools"]
    del payload["premium_tools"]

    assembled = provider_config.assemble_routes(
        _base_routes(router=_router()),
        provider_files={"hermes": payload},
    )

    assert assembled["agents"]["hermes"]["tools"]["terminal"] == "fast"
    assert "patch" in assembled["agents"]["hermes"]["premium_tools"]


def test_resolve_agent_group_precedence_header_wins() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/chat/completions",
        headers={"x-acr-agent": "claude_code", "user-agent": "Hermes CLI"},
        routes=routes,
        adapter_default="hermes",
    )

    assert group == "claude_code"


def test_resolve_agent_group_user_agent_wins_over_endpoint_case_insensitively() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/messages",
        headers={"user-agent": "HERMES Desktop"},
        routes=routes,
        adapter_default="claude_code",
    )

    assert group == "hermes"


def test_resolve_agent_group_without_router_uses_adapter_default() -> None:
    group = provider_config.resolve_agent_group(
        path="/v1/messages",
        headers={},
        routes=_base_routes(),
        adapter_default="claude_code",
    )

    assert group == "claude_code"


def test_resolve_agent_group_fallback_null_fails_closed() -> None:
    routes = _base_routes(router={"fallback": None, "user_agent_map": []})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={"user-agent": "unknown"},
        routes=routes,
        adapter_default="claude_code",
    )

    assert group is None


def test_resolve_agent_group_uses_configured_fallback() -> None:
    routes = _base_routes(router={"fallback": "claude_code"})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={},
        routes=routes,
        adapter_default="hermes",
    )

    assert group == "claude_code"


def test_resolve_agent_group_invalid_header_falls_through() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/chat/completions",
        headers={"x-acr-agent": "unknown"},
        routes=routes,
        adapter_default="hermes",
    )

    assert group == "hermes"


def test_resolve_agent_group_skips_malformed_user_agent_rules() -> None:
    routes = _base_routes(
        router={
            "user_agent_map": [
                "bad",
                {"contains": "hermes", "group": "hermes"},
            ],
            "fallback": None,
        }
    )

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={"user-agent": "Hermes CLI"},
        routes=routes,
        adapter_default="claude_code",
    )

    assert group == "hermes"


def test_resolve_agent_group_without_fallback_uses_adapter_default() -> None:
    routes = _base_routes(router={"user_agent_map": []})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={},
        routes=routes,
        adapter_default="hermes",
    )

    assert group == "hermes"


def test_resolve_agent_group_invalid_fallback_uses_adapter_default() -> None:
    routes = _base_routes(router={"fallback": "not-a-group"})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={},
        routes=routes,
        adapter_default="claude_code",
    )

    assert group == "claude_code"


def test_assemble_routes_does_not_mutate_base() -> None:
    base = _base_routes(router=_router())
    before = copy.deepcopy(base)

    provider_config.assemble_routes(
        base,
        provider_files={"hermes": _provider_payload("hermes")},
    )

    assert base == before


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_tier_config_accepts_routed_effort_levels(effort: str) -> None:
    parsed = parse_tier_config({"model": "anthropic/claude-sonnet-5", "effort": effort})

    assert parsed.effort == effort


def test_tier_config_effort_defaults_to_none_when_absent() -> None:
    assert parse_tier_config({"model": "anthropic/claude-sonnet-4-6"}).effort is None


@pytest.mark.parametrize("bad_effort", ["highest", "", "HIGH", 3])
def test_tier_config_rejects_efforts_a_routed_model_cannot_serve(bad_effort: object) -> None:
    # xhigh is now a valid level (Sonnet 5 / Opus accept it); these remain invalid.
    with pytest.raises(ConfigSchemaError):
        parse_tier_config({"model": "anthropic/claude-sonnet-5", "effort": bad_effort})


def test_tier_config_accepts_positive_context_window() -> None:
    parsed = parse_tier_config({"model": "anthropic/claude-sonnet-5", "context_window": 200000})

    assert parsed.context_window == 200000


def test_tier_config_context_window_defaults_to_none_when_absent() -> None:
    # Unset is the only way to disable the guard; None means "no window configured".
    assert parse_tier_config({"model": "anthropic/claude-sonnet-5"}).context_window is None


@pytest.mark.parametrize("bad_window", [0, -1, -200000])
def test_tier_config_rejects_nonpositive_context_window(bad_window: int) -> None:
    # gt=0, matching max_tokens: a non-positive window is a config error, not a
    # silent disable of the overflow guard.
    with pytest.raises(ConfigSchemaError):
        parse_tier_config({"model": "anthropic/claude-sonnet-5", "context_window": bad_window})
