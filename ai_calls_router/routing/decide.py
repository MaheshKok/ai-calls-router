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
from typing import Any

import yaml

from ai_calls_router._lib import config

logger = logging.getLogger("acr.routing")

_cache_lock = threading.Lock()
_cache: dict[Path, tuple[float, dict[str, Any]]] = {}


def load_routes(path: Path | None = None) -> dict[str, Any]:
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
            parsed = yaml.safe_load(fh) or {}
        if not isinstance(parsed, dict):
            return {}
        with _cache_lock:
            _cache[path] = (mtime, parsed)
        return parsed
    except Exception as exc:
        logger.warning("acr: config load failed (%s); routing disabled", exc, exc_info=True)
        return {}


def _premium_aliases(settings: dict[str, Any]) -> dict[str, str]:
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


def _canonical_tool_name(name: str, settings: dict[str, Any]) -> str:
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


def _tool_id_to_name_map(messages: list[Any], settings: dict[str, Any]) -> dict[str, str]:
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


def pending_tool_names(body: dict[str, Any], settings: dict[str, Any] | None = None) -> list[str]:
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

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return []
    content = last.get("content")
    if not isinstance(content, list):
        return []

    result_ids = [
        block.get("tool_use_id")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id")
    ]
    if not result_ids:
        return []

    id_to_name = _tool_id_to_name_map(messages, settings)
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
    name: str, tools_map: dict[str, str], settings: dict[str, Any] | None = None
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


def tier_for_tools(names: list[str], routes: dict[str, Any]) -> str:
    """Pick the serving tier for a batch of pending tools.

    Unknown tools resolve to premium; mixed batches resolve to the highest
    tier in settings.tier_precedence (premium always outranks).

    Args:
        names: Pending tool names (output of pending_tool_names).
        routes: Parsed config.yaml.

    Returns:
        Tier name ("premium" means passthrough).
    """
    tools_map = routes.get("tools") or {}
    settings = routes.get("settings") or {}
    precedence = settings.get("tier_precedence", ["premium", "structured", "code", "fast", "crud"])

    tiers = set()
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


def resolve_api_key(tier_cfg: dict[str, Any], settings: dict[str, Any]) -> str | None:
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
    if not key_env:
        return None
    value = os.environ.get(key_env)
    if value:
        return value

    env_file = settings.get("env_file")
    if not env_file:
        return None
    try:
        env_path = Path(env_file).expanduser()
        return _search_env_file(env_path, key_env)
    except Exception as exc:
        logger.warning("acr: env_file read failed (%s)", exc, exc_info=True)
    return None
