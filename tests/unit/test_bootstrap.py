"""Tests for Phase 7 provider YAML bootstrap.

Bootstrapping is create-only: it materializes missing provider files from
built-in defaults, never overwrites edited files, and keeps generated YAML free
of cheap-tier credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_calls_router._lib import config
from ai_calls_router.ops import bootstrap
from ai_calls_router.routing import provider_config
from ai_calls_router.routing.adapters.base import (
    AGENT_GROUP_ENDPOINTS,
    AGENT_GROUP_WIRES,
)

ACTIVE_GROUPS = {"claude_code", "hermes"}


def _base_routes() -> dict[str, object]:
    """Return a global router config that consumes provider files."""
    return {
        "server": {"upstream": "https://premium.default.example"},
        "router": {
            "endpoint_defaults": {
                "/v1/messages": "claude_code",
                "/v1/chat/completions": "hermes",
            },
            "fallback": None,
        },
        "settings": {"tier_precedence": ["premium", "fast", "crud"]},
        "tiers": {"fast": {"model": "deepseek/x", "key_env": "CHEAP_KEY"}},
    }


def _has_forbidden_key_env(value: object, path: tuple[str, ...] = ()) -> bool:
    """Return whether generated provider YAML carries a non-auth key_env."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if key == "key_env" and child_path[-2:] != ("auth", "key_env"):
                return True
            if _has_forbidden_key_env(child, child_path):
                return True
    if isinstance(value, list):
        return any(_has_forbidden_key_env(item, path) for item in value)
    return False


@pytest.fixture
def acr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point provider bootstrap at an isolated home directory."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.delenv("ACR_CONFIG", raising=False)
    return tmp_path


def test_all_absent_creates_known_provider_configs(acr_home: Path) -> None:
    created = bootstrap.ensure_provider_configs()

    assert {path.name for path in created} == {
        "claude-code.yaml",
        "hermes.yaml",
    }
    loaded = provider_config.load_provider_files()
    assert set(loaded) == ACTIVE_GROUPS
    assembled = provider_config.assemble_routes(_base_routes(), provider_files=loaded)
    assert set(assembled["agents"]) == ACTIVE_GROUPS


def test_existing_edited_file_is_left_byte_identical(acr_home: Path) -> None:
    config.provider_config_dir().mkdir(parents=True)
    existing = config.provider_config_path("hermes")
    original = "group: hermes\nupstream: https://edited.example\nwire: openai_chat\n"
    existing.write_text(original, encoding="utf-8")

    created = bootstrap.ensure_provider_configs()

    assert existing.read_text(encoding="utf-8") == original
    assert {path.name for path in created} == {"claude-code.yaml"}


def test_second_run_is_noop(acr_home: Path) -> None:
    bootstrap.ensure_provider_configs()

    assert bootstrap.ensure_provider_configs() == []


def test_templates_carry_no_cheap_key_env_and_pass_validation(acr_home: Path) -> None:
    bootstrap.ensure_provider_configs()
    loaded = provider_config.load_provider_files()

    for payload in loaded.values():
        assert not _has_forbidden_key_env(payload)

    assembled = provider_config.assemble_routes(_base_routes(), provider_files=loaded)
    assert assembled["agents"]["hermes"]["tiers"]["fast"]["auth"]["mode"] == "api_key_env"


def test_templates_label_reserved_fields(acr_home: Path) -> None:
    bootstrap.ensure_provider_configs()

    body = config.provider_config_path("hermes").read_text(encoding="utf-8")
    assert "Runtime fields: upstream, tools, premium_tools." in body
    assert "Reserved/validated fields:" in body


def test_group_wire_endpoint_tables_match_bootstrap_templates() -> None:
    assert ACTIVE_GROUPS.issubset(set(AGENT_GROUP_WIRES))
    assert ACTIVE_GROUPS.issubset(set(AGENT_GROUP_ENDPOINTS))
    for group in ACTIVE_GROUPS:
        provider_config._validate_provider_payload(group, bootstrap._provider_template(group))
