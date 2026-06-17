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
            "/v1/responses": "codex",
        },
        "user_agent_map": [
            {"contains": "claude", "group": "claude_code"},
            {"contains": "hermes", "group": "hermes"},
            {"contains": "codex", "group": "codex"},
        ],
        "fallback": None,
    }


def _provider_payload(group: str, *, upstream: str | None = None) -> dict[str, object]:
    """Return one valid provider YAML payload."""
    wires = {
        "claude_code": "anthropic_messages",
        "codex": "openai_responses",
        "hermes": "openai_chat",
    }
    endpoints = {
        "claude_code": ["/v1/messages"],
        "codex": ["/v1/responses"],
        "hermes": ["/v1/chat/completions"],
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


def test_cheap_key_env_provider_payload_is_rejected() -> None:
    payload = _provider_payload("codex")
    payload["tiers"] = {"fast": {"key_env": "MUST_NOT_MOVE_HERE"}}

    with pytest.raises(provider_config.ProviderConfigError):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"codex": payload},
        )


def test_missing_provider_file_falls_back_without_dropping_present_groups() -> None:
    assembled = provider_config.assemble_routes(
        _base_routes(router=_router()),
        provider_files={"codex": _provider_payload("codex")},
    )

    assert assembled["agents"]["codex"]["upstream"] == "https://codex.example"
    assert assembled["agents"]["hermes"]["tools"]["terminal"] == "fast"
    assert assembled["agents"]["claude_code"]["tools"]["Bash"] == "fast"


def test_load_provider_files_skips_missing_and_malformed_files(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config.provider_config_dir().mkdir(parents=True)
    config.provider_config_path("codex").write_text(
        yaml.safe_dump(_provider_payload("codex")),
        encoding="utf-8",
    )
    config.provider_config_path("hermes").write_text(": bad: [", encoding="utf-8")

    loaded = provider_config.load_provider_files()

    assert set(loaded) == {"codex"}
    assert loaded["codex"]["group"] == "codex"
    assert "failed to load" in caplog.text


def test_load_provider_files_skips_non_mapping_yaml(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config.provider_config_dir().mkdir(parents=True)
    config.provider_config_path("codex").write_text("- not\n- mapping\n", encoding="utf-8")

    assert provider_config.load_provider_files() == {}
    assert "is not a mapping" in caplog.text


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"group": "hermes"}, "group mismatch"),
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
    payload = _provider_payload("codex")
    payload.update(updates)

    with pytest.raises(provider_config.ProviderConfigError, match=message):
        provider_config.assemble_routes(
            _base_routes(router=_router()),
            provider_files={"codex": payload},
        )


def test_provider_payload_can_omit_tool_maps_and_use_defaults() -> None:
    payload = _provider_payload("codex")
    del payload["tools"]
    del payload["premium_tools"]

    assembled = provider_config.assemble_routes(
        _base_routes(router=_router()),
        provider_files={"codex": payload},
    )

    assert assembled["agents"]["codex"]["tools"]["exec_command"] == "fast"
    assert "apply_patch" in assembled["agents"]["codex"]["premium_tools"]


def test_resolve_agent_group_precedence_header_wins() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/responses",
        headers={"x-acr-agent": "hermes", "user-agent": "Codex CLI"},
        routes=routes,
        adapter_default="codex",
    )

    assert group == "hermes"


def test_resolve_agent_group_user_agent_wins_over_endpoint_case_insensitively() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/responses",
        headers={"user-agent": "HERMES Desktop"},
        routes=routes,
        adapter_default="codex",
    )

    assert group == "hermes"


def test_resolve_agent_group_without_router_uses_adapter_default() -> None:
    group = provider_config.resolve_agent_group(
        path="/v1/responses",
        headers={},
        routes=_base_routes(),
        adapter_default="codex",
    )

    assert group == "codex"


def test_resolve_agent_group_fallback_null_fails_closed() -> None:
    routes = _base_routes(router={"fallback": None, "user_agent_map": []})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={"user-agent": "unknown"},
        routes=routes,
        adapter_default="codex",
    )

    assert group is None


def test_resolve_agent_group_uses_configured_fallback() -> None:
    routes = _base_routes(router={"fallback": "claude_code"})

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={},
        routes=routes,
        adapter_default="codex",
    )

    assert group == "claude_code"


def test_resolve_agent_group_invalid_header_falls_through() -> None:
    routes = _base_routes(router=_router())

    group = provider_config.resolve_agent_group(
        path="/v1/responses",
        headers={"x-acr-agent": "unknown"},
        routes=routes,
        adapter_default="codex",
    )

    assert group == "codex"


def test_resolve_agent_group_skips_malformed_user_agent_rules() -> None:
    routes = _base_routes(
        router={
            "user_agent_map": [
                "bad",
                {"contains": "codex", "group": "codex"},
            ],
            "fallback": None,
        }
    )

    group = provider_config.resolve_agent_group(
        path="/v1/unknown",
        headers={"user-agent": "Codex CLI"},
        routes=routes,
        adapter_default="hermes",
    )

    assert group == "codex"


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
        adapter_default="codex",
    )

    assert group == "codex"


def test_assemble_routes_does_not_mutate_base() -> None:
    base = _base_routes(router=_router())
    before = copy.deepcopy(base)

    provider_config.assemble_routes(
        base,
        provider_files={"codex": _provider_payload("codex")},
    )

    assert base == before
