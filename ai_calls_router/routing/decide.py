"""Routing decisions for ai-calls-router.

Loads config.yaml (mtime-cached per path), extracts which tools' results an
Anthropic Messages API request is processing, and resolves the serving tier.
Every failure path returns the premium/passthrough decision -- routing must
never break traffic that would otherwise succeed.

Tool names that arrive from Claude Code are canonicalized against the
configured premium tool list before matching, so alias/casing drift (e.g.
``edit`` -> ``Edit``) still resolves to the configured literal. Ported from the
proven headroom-tool-router engine; the only behavioral changes are a per-path
cache (instead of a single global slot) and tilde expansion for the env_file
setting.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing.agent_defaults import AGENT_DEFAULT_PREMIUM_TOOLS, AGENT_DEFAULT_TOOLS
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
    validate_routes_payload,
)

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

logger = logging.getLogger("acr.routing")

_cache_lock = threading.Lock()
_cache: dict[Path, tuple[float, JsonObject]] = {}
_DEFAULT_AGENT_GROUP = "claude_code"


def _json_mapping(value: JsonValue) -> JsonObject:
    """Return a JSON object when value is mapping-shaped, else an empty mapping."""
    if isinstance(value, dict):
        return value
    return {}


def _json_array(value: JsonValue) -> JsonArray:
    """Return a JSON array when value is list-shaped, else an empty list."""
    if isinstance(value, list):
        return value
    return []


def load_routes(path: Path | None = None) -> JsonObject:
    """Load config.yaml with a per-path mtime cache.

    Args:
        path: Config file location (default: config.config_path()).

    Returns:
        Parsed config mapping, or {} on any error (callers treat an empty
        mapping as "route nothing" -- fail open).
    """
    if path is None:
        path = config.config_path()
    try:
        mtime = path.stat().st_mtime
        with _cache_lock:
            cached = _cache.get(path)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        with path.open(encoding="utf-8") as fh:
            parsed = cast("JsonValue", yaml.safe_load(fh) or {})
        if not isinstance(parsed, dict):
            return {}
        try:
            validate_routes_payload(parsed)
        except ConfigSchemaError as exc:
            logger.warning("acr: config schema validation failed (%s); routing disabled", exc)
            return {}
        with _cache_lock:
            _cache[path] = (mtime, parsed)
        return parsed
    except Exception as exc:
        logger.warning("acr: config load failed (%s); routing disabled", exc, exc_info=True)
        return {}


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
    settings_cfg = _json_mapping(routes.get("settings", None))
    server_cfg = _json_mapping(routes.get("server", None))
    premium_cfg = _json_mapping(routes.get("premium", {"provider": "anthropic"}))
    tools_cfg = _json_mapping(routes.get("tools", None)) or cast(
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
    settings = _json_mapping(routes.get("settings", None))
    precedence_value = settings.get(
        "tier_precedence", ["premium", "structured", "code", "fast", "crud"]
    )
    precedence = [
        str(tier) for tier in _json_array(precedence_value) if isinstance(tier, str) and tier
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


def _search_env_file(env_path: Path, key_env: str) -> str | None:
    """Search a .env file for a variable, returning its value or None."""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        if name.strip() in (key_env, f"export {key_env}"):
            return raw.strip().strip("'\"") or None
    return None


def resolve_api_key(tier_cfg: JsonObject, settings: JsonObject) -> str | None:
    """Resolve the provider API key for a tier.

    Looks up the env var named by key_env in the process environment first,
    then in the env_file configured under settings (simple KEY=VALUE lines).
    Key values are never logged.

    Args:
        tier_cfg: Tier entry from config.yaml.
        settings: The settings: section of config.yaml.

    Returns:
        The key, or None when unavailable (callers must fail open).
    """
    key_env = tier_cfg.get("key_env")
    if not isinstance(key_env, str) or not key_env:
        return None
    value = os.environ.get(key_env)
    if value:
        return value

    env_file = settings.get("env_file")
    if not isinstance(env_file, str) or not env_file:
        return None
    try:
        env_path = Path(env_file).expanduser()
        return _search_env_file(env_path, key_env)
    except Exception as exc:
        logger.warning("acr: env_file read failed (%s)", exc, exc_info=True)
    return None


def resolve_tier_credential(tier_cfg: JsonObject, settings: JsonObject) -> str | None:
    """Resolve a tier credential, with Codex tiers defaulting to OPENAI_API_KEY.

    Args:
        tier_cfg: Tier entry from config.yaml.
        settings: The settings: section of config.yaml.

    Returns:
        The resolved credential, ``"oauth"`` for Codex OAuth sentinel tiers, or
        None when unavailable.
    """
    try:
        parsed = parse_tier_config(tier_cfg)
    except ConfigSchemaError as exc:
        logger.warning("acr: tier schema validation failed (%s); passing through", exc)
        return None
    if parsed.key_env == "oauth" and is_codex_tier(tier_cfg):
        return "oauth"
    credential = resolve_api_key(tier_cfg, settings)
    if credential:
        return credential
    if is_codex_tier(tier_cfg):
        return os.environ.get("OPENAI_API_KEY")
    return None
