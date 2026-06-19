"""Tests for routing agent default maps.

The defaults are data used by both the wizard and the legacy-config compat
shim. These tests assert the spec-level invariants instead of mirroring every
entry, so a missing group or invalid premium mapping fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_calls_router.routing.agent_defaults import (
    AGENT_DEFAULT_PREMIUM_TOOLS,
    AGENT_DEFAULT_TOOLS,
)


def _example_config() -> dict[str, object]:
    """Load the shipped example config."""
    path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_all_required_groups_are_present() -> None:
    assert set(AGENT_DEFAULT_TOOLS) == {"claude_code", "hermes", "codex"}
    assert set(AGENT_DEFAULT_PREMIUM_TOOLS) == {"claude_code", "hermes", "codex"}


def test_every_group_has_non_empty_tool_map() -> None:
    assert all(AGENT_DEFAULT_TOOLS[group] for group in AGENT_DEFAULT_TOOLS)


def test_every_referenced_tier_exists_in_example_config() -> None:
    example = _example_config()
    agents = example["agents"]
    known_tiers = {
        tier
        for agent in agents.values()
        if isinstance(agent, dict)
        for tier in (agent.get("tiers") or {})
    } | {"premium"}
    referenced = {tier for tools in AGENT_DEFAULT_TOOLS.values() for tier in tools.values()}
    assert referenced <= known_tiers


def test_premium_tools_map_to_premium_and_exist() -> None:
    for group, premium_tools in AGENT_DEFAULT_PREMIUM_TOOLS.items():
        tool_map = AGENT_DEFAULT_TOOLS[group]
        assert premium_tools
        for tool in premium_tools:
            assert tool in tool_map
            assert tool_map[tool] == "premium"
