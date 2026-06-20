"""Per-agent-group routing topology resolved from config.yaml.

Resolves the shape of each agent group: its tool-to-tier map, premium
response-guard tools, passthrough upstream, and any agent-local tier configs.
Also exposes the legacy-flat-config compatibility shim that synthesizes an
``agents.claude_code`` block from top-level fields. Every lookup falls back to
the group's built-in defaults on missing or malformed config so serving stays
fail-open. This module holds topology only -- no tier selection and no
credential resolution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from ai_calls_router._lib import config
from ai_calls_router.routing.agent_defaults import (
    AGENT_DEFAULT_PREMIUM_TOOLS,
    AGENT_DEFAULT_TOOLS,
)
from ai_calls_router.routing.route_json import json_mapping

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject

logger = logging.getLogger("acr.routing")

_DEFAULT_AGENT_GROUP = "claude_code"


def with_agent_compat(routes: JsonObject) -> JsonObject:
    """Return routes with a legacy flat config exposed as agents.claude_code.

    Args:
        routes: Parsed config.yaml mapping.

    Returns:
        The original mapping when agents: already exists; otherwise a shallow
        copy with agents.claude_code synthesized from legacy top-level fields.
    """
    if "agents" in routes:
        return routes
    settings_cfg = json_mapping(routes.get("settings", None))
    server_cfg = json_mapping(routes.get("server", None))
    premium_cfg = json_mapping(routes.get("premium", {"provider": "anthropic"}))
    tools_cfg = json_mapping(routes.get("tools", None)) or cast(
        "JsonObject", dict(AGENT_DEFAULT_TOOLS[_DEFAULT_AGENT_GROUP])
    )
    premium_tools = settings_cfg.get("premium_tools")
    premium_tool_cfg = (
        premium_tools
        if isinstance(premium_tools, list) and all(isinstance(tool, str) for tool in premium_tools)
        else AGENT_DEFAULT_PREMIUM_TOOLS[_DEFAULT_AGENT_GROUP]
    )
    return cast(
        "JsonObject",
        {
            **routes,
            "agents": {
                _DEFAULT_AGENT_GROUP: {
                    "tools": dict(tools_cfg),
                    "premium_tools": list(premium_tool_cfg),
                    "upstream": server_cfg.get("upstream", config.DEFAULT_UPSTREAM),
                    "premium": dict(premium_cfg),
                }
            },
        },
    )


def _default_agent_group(group: str) -> str:
    """Return a known agent group, defaulting to Claude Code."""
    if group in AGENT_DEFAULT_TOOLS:
        return group
    return _DEFAULT_AGENT_GROUP


def _agent_config(routes: JsonObject, group: str) -> JsonObject:
    """Return the agent config mapping for group, or an empty mapping."""
    compat_routes = with_agent_compat(routes)
    agents = compat_routes.get("agents")
    if not isinstance(agents, dict):
        return {}
    cfg = agents.get(group)
    if isinstance(cfg, dict):
        return cfg
    return {}


def agent_tools(routes: JsonObject, group: str) -> dict[str, str]:
    """Resolve the tool map for one agent group, falling back to defaults.

    Args:
        routes: Parsed config.yaml mapping.
        group: Agent group name.

    Returns:
        A new tool-to-tier mapping. Malformed or missing config falls back to
        the group's built-in defaults, preserving fail-open serving behavior.
    """
    try:
        tools = _agent_config(routes, group).get("tools")
        if isinstance(tools, dict) and all(isinstance(tier, str) for tier in tools.values()):
            return {name: tier for name, tier in tools.items() if isinstance(tier, str)}
    except Exception as exc:
        logger.warning("acr: agent tools lookup failed (%s); using defaults", exc, exc_info=True)
    return dict(AGENT_DEFAULT_TOOLS[_default_agent_group(group)])


def agent_premium_tools(routes: JsonObject, group: str) -> list[str]:
    """Resolve the premium response-guard tools for one agent group.

    Args:
        routes: Parsed config.yaml mapping.
        group: Agent group name.

    Returns:
        A new list of premium tool names, falling back to group defaults on
        missing or malformed config.
    """
    try:
        tools = _agent_config(routes, group).get("premium_tools")
        if isinstance(tools, list) and all(isinstance(tool, str) for tool in tools):
            return [tool for tool in tools if isinstance(tool, str)]
    except Exception as exc:
        logger.warning(
            "acr: agent premium tools lookup failed (%s); using defaults", exc, exc_info=True
        )
    return list(AGENT_DEFAULT_PREMIUM_TOOLS[_default_agent_group(group)])


def agent_upstream(routes: JsonObject, group: str) -> str:
    """Resolve the passthrough upstream for one agent group.

    Args:
        routes: Parsed config.yaml mapping.
        group: Agent group name.

    Returns:
        The agent upstream without a trailing slash, or the premium default
        when the group config is missing, malformed, or cannot be read.
    """
    try:
        agent_cfg = _agent_config(routes, group)
        upstream = agent_cfg.get("upstream")
        if isinstance(upstream, str) and upstream:
            return upstream.rstrip("/")
        if "upstream" in agent_cfg:
            logger.warning(
                "acr: malformed agent upstream for group=%s; using premium default", group
            )
    except Exception as exc:
        logger.warning(
            "acr: agent upstream lookup failed (%s); using premium default",
            exc,
            exc_info=True,
        )
    return config.server_settings(routes).upstream


def agent_tier_config(routes: JsonObject, group: str, tier: str) -> JsonObject | None:
    """Resolve an agent-local tier config, if one is configured."""
    try:
        tiers = _agent_config(routes, group).get("tiers")
        if not isinstance(tiers, dict):
            return None
        tier_cfg = tiers.get(tier)
        if isinstance(tier_cfg, dict):
            return dict(tier_cfg)
    except Exception as exc:
        logger.warning("acr: agent tier lookup failed (%s); using global tiers", exc, exc_info=True)
    return None
