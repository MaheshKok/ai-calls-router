"""Provider-file assembly and agent identity resolution.

This module is the boundary between Phase 7's on-disk provider YAML files and
the canonical routes dict consumed by the existing decision core. Reads are
kept separate from pure assembly so serving can skip malformed provider files
without changing `decide.load_routes`.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing.adapters.base import (
    AGENT_GROUP_ENDPOINTS,
    AGENT_GROUP_WIRES,
    KNOWN_GROUPS,
)
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    validate_provider_payload,
)
from ai_calls_router.routing.decide import (
    agent_premium_tools,
    agent_tools,
    agent_upstream,
    with_agent_compat,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

logger = logging.getLogger("acr.provider_config")


class ProviderConfigError(ValueError):
    """Raised when a provider file violates the Phase 7 config contract."""

    def __init__(self, message: str, *, group: str | None = None) -> None:
        """Initialize the validation error.

        Args:
            message: Human-readable validation failure.
            group: Provider group whose payload failed validation, when known.
        """
        super().__init__(message)
        self.group = group


def load_provider_files() -> dict[str, JsonObject]:
    """Load present per-provider YAML files.

    Missing files are absent from the returned mapping. Malformed files are
    skipped with a warning so serving can keep using the global config.
    """
    loaded: dict[str, JsonObject] = {}
    for group in sorted(KNOWN_GROUPS):
        path = config.provider_config_path(group)
        if not path.exists():
            continue
        try:
            parsed = cast("JsonValue", yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except Exception as exc:
            logger.warning("acr: provider config %s failed to load (%s); skipping", path, exc)
            continue
        if not isinstance(parsed, dict):
            logger.warning("acr: provider config %s is not a mapping; skipping", path)
            continue
        loaded[group] = parsed
    return loaded


def router_map(routes: JsonObject) -> JsonObject | None:
    """Return the configured router block, if present."""
    router = routes.get("router")
    return router if isinstance(router, dict) else None


def _validate_provider_payload(group: str, payload: JsonObject) -> None:
    """Validate one provider file payload."""
    upstream = payload.get("upstream")
    if not isinstance(upstream, str) or not upstream:
        raise ProviderConfigError("provider config requires upstream", group=group)
    parsed_upstream = urlsplit(upstream)
    if parsed_upstream.scheme != "https" or not parsed_upstream.hostname:
        raise ProviderConfigError("provider config upstream must be an https URL", group=group)
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not all(isinstance(item, str) for item in endpoints):
        raise ProviderConfigError("provider config requires endpoints", group=group)
    try:
        parsed = validate_provider_payload(group, payload)
    except ConfigSchemaError as exc:
        raise ProviderConfigError(str(exc), group=group) from exc
    if parsed.group != group or parsed.group not in KNOWN_GROUPS:
        raise ProviderConfigError("provider config group mismatch", group=group)
    if parsed.wire != AGENT_GROUP_WIRES[group]:
        raise ProviderConfigError("provider config wire mismatch", group=group)
    if not set(AGENT_GROUP_ENDPOINTS[group]).issubset(parsed.endpoints):
        raise ProviderConfigError("provider config endpoints mismatch", group=group)


def _fallback_agent_config(routes: JsonObject, group: str) -> JsonObject:
    """Build an agent config from canonical defaults for a missing provider file."""
    return cast(
        "JsonObject",
        {
            "tools": agent_tools(routes, group),
            "premium_tools": agent_premium_tools(routes, group),
            "upstream": agent_upstream(routes, group),
            "premium": {"provider": "anthropic"},
        },
    )


def _provider_agent_config(routes: JsonObject, group: str, payload: JsonObject) -> JsonObject:
    """Build canonical agents[group] config from one provider payload."""
    _validate_provider_payload(group, payload)
    agent_cfg = {key: copy.deepcopy(value) for key, value in payload.items() if key != "group"}
    if "tools" not in agent_cfg:
        agent_cfg["tools"] = cast("JsonValue", agent_tools(routes, group))
    if "premium_tools" not in agent_cfg:
        agent_cfg["premium_tools"] = cast("JsonValue", agent_premium_tools(routes, group))
    return agent_cfg


def assemble_routes(base: JsonObject, *, provider_files: dict[str, JsonObject]) -> JsonObject:
    """Merge provider-file payloads into a new canonical routes dict.

    Args:
        base: Parsed global config.
        provider_files: Provider YAML payloads keyed by group.

    Returns:
        Canonical routes dict. With no ``router:`` block, provider files are
        ignored so legacy flat and single-file ``agents:`` configs keep their
        existing behavior.

    Raises:
        ProviderConfigError: If a provider payload violates an invariant.
    """
    compat = with_agent_compat(base)
    if not provider_files or router_map(compat) is None:
        return compat

    result = copy.deepcopy(compat)
    existing_agents = result.get("agents")
    agents = dict(existing_agents) if isinstance(existing_agents, dict) else {}
    for group in sorted(KNOWN_GROUPS):
        payload = provider_files.get(group)
        if payload is None:
            agents.setdefault(group, _fallback_agent_config(compat, group))
            continue
        agents[group] = _provider_agent_config(compat, group, payload)
    result["agents"] = agents
    return result


def _known(candidate: JsonValue) -> str | None:
    """Return a valid group name, or None."""
    return candidate if isinstance(candidate, str) and candidate in KNOWN_GROUPS else None


def _header_value(headers: Mapping[str, str], name: str) -> str:
    """Return a request header value case-insensitively."""
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def _user_agent_match(router: JsonObject, user_agent: str) -> str | None:
    """Resolve group from router.user_agent_map."""
    mappings = router.get("user_agent_map")
    if not isinstance(mappings, list):
        return None
    normalized_ua = user_agent.lower()
    for item in cast("JsonArray", mappings):
        if not isinstance(item, dict):
            continue
        contains = item.get("contains")
        group = _known(item.get("group"))
        if isinstance(contains, str) and group and contains.lower() in normalized_ua:
            return group
    return None


def _endpoint_default(router: JsonObject, path: str) -> str | None:
    """Resolve group from router.endpoint_defaults."""
    endpoint_defaults = router.get("endpoint_defaults")
    if isinstance(endpoint_defaults, dict):
        return _known(endpoint_defaults.get(path))
    return None


def _fallback_group(router: JsonObject, adapter_default: str) -> str | None:
    """Resolve the configured fallback group or adapter default."""
    if "fallback" not in router:
        return _known(adapter_default)
    fallback = router.get("fallback")
    if fallback is None:
        return None
    return _known(fallback) or _known(adapter_default)


def resolve_agent_group(
    *,
    path: str,
    headers: Mapping[str, str],
    routes: JsonObject,
    adapter_default: str,
) -> str | None:
    """Resolve an agent group by Phase 7 identity precedence."""
    header_group = _known(_header_value(headers, "x-acr-agent"))
    if header_group is not None:
        return header_group

    router = router_map(routes)
    if router is None:
        return _known(adapter_default)

    return (
        _user_agent_match(router, _header_value(headers, "user-agent"))
        or _endpoint_default(router, path)
        or _fallback_group(router, adapter_default)
    )
