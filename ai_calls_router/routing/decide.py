"""Routing decision facade for ai-calls-router.

Owns config.yaml loading (mtime-cached per path, schema-validated, fail-open)
and re-exports the routing decision surface from the focused modules it was
split into:

* :mod:`agent_topology` -- per-agent tool maps, premium tools, upstream, tiers.
* :mod:`tier_selection` -- pending tool extraction, canonicalization, tier choice.
* :mod:`credential_resolution` -- tier API key / OAuth credential resolution.

Callers continue to import this module as ``decide`` (commonly aliased
``routing``) and reach every decision helper through it, so the split is
internal. Every failure path returns the premium/passthrough decision --
routing must never break traffic that would otherwise succeed.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, cast

import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing.agent_topology import (
    agent_premium_tools,
    agent_tier_config,
    agent_tools,
    agent_upstream,
    with_agent_compat,
)
from ai_calls_router.routing.config_schema import ConfigSchemaError, validate_routes_payload
from ai_calls_router.routing.credential_resolution import (
    TierCredential,
    resolve_api_key,
    resolve_tier_credential,
)
from ai_calls_router.routing.tier_selection import (
    lookup_tool,
    pending_tool_names,
    tier_for_tools,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ai_calls_router._lib.types import JsonObject, JsonValue

logger = logging.getLogger("acr.routing")

_cache_lock = threading.Lock()
_cache: dict[Path, tuple[float, JsonObject]] = {}

# Re-exported decision surface; listed so static checkers keep the imports.
__all__ = [
    "TierCredential",
    "agent_premium_tools",
    "agent_tier_config",
    "agent_tools",
    "agent_upstream",
    "load_routes",
    "lookup_tool",
    "pending_tool_names",
    "resolve_api_key",
    "resolve_tier_credential",
    "tier_for_tools",
    "with_agent_compat",
]


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
