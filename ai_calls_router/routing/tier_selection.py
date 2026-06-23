"""Tier selection: which serving tier a batch of pending tools maps to.

Extracts the tool names whose results an Anthropic Messages request is
processing, canonicalizes them against the configured premium tool list so
alias/casing drift still resolves to the configured literal, then resolves the
batch to a serving tier via the agent group's tool-to-tier map and
``tier_precedence``. Unknown or premium tools force the premium/passthrough
decision. This module owns name canonicalization and tier choice only; it reads
agent topology through :mod:`agent_topology` and holds no credential logic.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, cast

from ai_calls_router.routing.agent_topology import agent_tools
from ai_calls_router.routing.route_json import json_array, json_mapping

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject


def _premium_aliases(settings: JsonObject) -> dict[str, str]:
    """Derive accepted aliases from the configured premium tool list.

    The configured ``premium_tools`` list is the single source of truth: each
    literal yields case-insensitive and separator-insensitive keys mapping back
    to the literal, so the router never carries a second hardcoded premium list.

    Args:
        settings: The settings: section of config.yaml.

    Returns:
        A mapping from normalized alias keys to the configured literal.
    """
    premium_tools = settings.get("premium_tools")
    if not isinstance(premium_tools, list):
        return {}
    aliases: dict[str, str] = {}
    for tool in premium_tools:
        if not isinstance(tool, str):
            continue
        canonical = tool.strip()
        if not canonical:
            continue
        aliases[canonical.lower()] = canonical
        aliases[canonical.replace(" ", "").lower()] = canonical
        aliases[canonical.replace("_", "").lower()] = canonical
        aliases[canonical.replace("-", "").lower()] = canonical
    return aliases


def _canonical_tool_name(name: str, settings: JsonObject) -> str:
    """Resolve a Claude Code tool name to the configured premium literal.

    Args:
        name: Raw tool name from the conversation.
        settings: The settings: section of config.yaml (may be empty).

    Returns:
        The configured literal when the name is a known alias, otherwise the
        whitespace-stripped name unchanged.
    """
    normalized = name.strip()
    return _premium_aliases(settings).get(normalized.lower(), normalized)


def _tool_id_to_name_map(messages: JsonArray, settings: JsonObject) -> dict[str, str]:
    """Build a mapping from tool_use ids to canonicalized tool names.

    Args:
        messages: The messages array, excluding the final user turn.
        settings: The settings: section of config.yaml (drives canonicalization).

    Returns:
        A mapping from tool_use id to the canonical tool name.
    """
    id_to_name: dict[str, str] = {}
    for msg in messages[:-1]:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        msg_content = msg.get("content")
        if not isinstance(msg_content, list):
            continue
        for block in msg_content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id")
                and block.get("name")
            ):
                id_to_name[str(block["id"])] = _canonical_tool_name(str(block["name"]), settings)
    return id_to_name


def pending_tool_names(body: JsonObject, settings: JsonObject | None = None) -> list[str]:
    """Extract the tool names whose results this request is processing.

    In the Anthropic Messages format, pending tool results are tool_result
    blocks in the LAST message (role=user). Each block carries a tool_use_id
    that matches a tool_use block in an earlier assistant message, which holds
    the tool NAME. Names are canonicalized against the configured premium tool
    list so alias/casing drift still resolves to the configured literal.

    Args:
        body: Parsed /v1/messages request body.
        settings: The settings: section of config.yaml; when omitted, no
            canonicalization is applied and raw names are returned.

    Returns:
        Ordered, deduplicated tool names; [] when the request is a turn
        opener / plain reply (no pending results) or on shape surprises.
        ["<unknown>"] when any result id cannot be resolved to a name, which
        forces the whole batch onto the premium path.
    """
    settings = settings or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    message_items = cast("JsonArray", messages)

    last = message_items[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return []
    content = last.get("content")
    if not isinstance(content, list):
        return []

    result_ids = [
        block.get("tool_use_id")
        for block in cast("JsonArray", content)
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    ]
    if not result_ids:
        return []

    id_to_name = _tool_id_to_name_map(message_items, settings)
    names: list[str] = []
    for rid in result_ids:
        name = id_to_name.get(str(rid))
        # An unresolvable id means we cannot prove the result is cheap to
        # interpret -- treat the whole batch as premium via the unknown path.
        if name is None:
            return ["<unknown>"]
        if name not in names:
            names.append(name)
    return names


def lookup_tool(
    name: str, tools_map: dict[str, str], settings: JsonObject | None = None
) -> str | None:
    """Resolve a tool name to a tier, exact match first then trailing-* glob.

    Args:
        name: Tool name from the conversation.
        tools_map: The `tools:` section of config.yaml.
        settings: The settings: section of config.yaml; drives alias
            canonicalization before matching when provided.

    Returns:
        Tier name, or None when unmapped (callers treat None as premium).
    """
    canonical_name = _canonical_tool_name(name, settings or {})
    if canonical_name in tools_map:
        return tools_map[canonical_name]
    for pattern, tier in tools_map.items():
        if pattern.endswith("*") and fnmatch.fnmatch(canonical_name, pattern):
            return tier
    return None


def tier_for_tools(names: list[str], routes: JsonObject, *, group: str) -> str:
    """Pick the serving tier for a batch of pending tools.

    Unknown tools resolve to premium; mixed batches resolve to the highest
    tier in settings.tier_precedence (premium always outranks).

    Args:
        names: Pending tool names (output of pending_tool_names).
        routes: Parsed config.yaml.
        group: Agent group to use for tool-to-tier lookup.

    Returns:
        Tier name ("premium" means passthrough).
    """
    tools_map = agent_tools(routes, group)
    settings = json_mapping(routes.get("settings", None))
    precedence_value = settings.get(
        "tier_precedence", ["premium", "structured", "code", "fast", "crud"]
    )
    precedence = [
        str(tier) for tier in json_array(precedence_value) if isinstance(tier, str) and tier
    ] or ["premium", "structured", "code", "fast", "crud"]

    tiers: set[str] = set()
    for name in names:
        tier = lookup_tool(name, tools_map, settings)
        if tier is None or tier == "premium":
            return "premium"
        tiers.add(tier)
    if not tiers:
        return "premium"

    for tier in precedence:
        if tier in tiers:
            return tier
    # A tier configured in tools: but absent from precedence -- be safe.
    return "premium"
