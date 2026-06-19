"""Bootstrap provider-specific YAML config files.

Phase 7 keeps global cheap-tier config in config.yaml and materializes one
provider file per agent family. The helper is create-only so edited provider
files are never overwritten.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing.adapters.base import (
    AGENT_GROUP_ENDPOINTS,
    AGENT_GROUP_WIRES,
)
from ai_calls_router.routing.agent_defaults import (
    AGENT_DEFAULT_PREMIUM_TOOLS,
    AGENT_DEFAULT_TOOLS,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ai_calls_router._lib.types import JsonObject

_UPSTREAM_BY_GROUP: dict[str, str] = {
    "claude_code": config.DEFAULT_UPSTREAM,
    "hermes": "https://api.openai.com",
}
_BOOTSTRAP_GROUPS: tuple[str, ...] = ("claude_code", "hermes")
_TIERS_BY_GROUP: dict[str, JsonObject] = {
    "claude_code": {
        "fast": {
            "provider": "deepseek",
            "model": "deepseek/deepseek-v4-flash",
            "auth": {"mode": "api_key_env", "key_env": "DEEPSEEK_API_KEY"},
            "max_tokens": 8192,
            "input_cost_per_1m": 0.14,
            "input_cached_cost_per_1m": 0.0028,
            "output_cost_per_1m": 0.28,
        },
        "code": {
            "provider": "deepseek",
            "model": "deepseek/deepseek-v4-pro",
            "auth": {"mode": "api_key_env", "key_env": "DEEPSEEK_API_KEY"},
            "max_tokens": 8192,
            "input_cost_per_1m": 0.435,
            "input_cached_cost_per_1m": 0.003625,
            "output_cost_per_1m": 0.87,
        },
        "crud": {
            "provider": "deepseek",
            "model": "deepseek/deepseek-v4-flash",
            "auth": {"mode": "api_key_env", "key_env": "DEEPSEEK_API_KEY"},
            "max_tokens": 4096,
            "input_cost_per_1m": 0.14,
            "input_cached_cost_per_1m": 0.0028,
            "output_cost_per_1m": 0.28,
        },
        "structured": {
            "provider": "deepseek",
            "model": "deepseek/deepseek-v4-flash",
            "auth": {"mode": "api_key_env", "key_env": "DEEPSEEK_API_KEY"},
            "max_tokens": 8192,
            "input_cost_per_1m": 0.14,
            "input_cached_cost_per_1m": 0.0028,
            "output_cost_per_1m": 0.28,
        },
    },
    "hermes": {
        "fast": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "auth": {"mode": "api_key_env", "key_env": "OPENAI_API_KEY"},
            "max_tokens": 8192,
        },
        "code": {
            "provider": "openai",
            "model": "gpt-4.1",
            "auth": {"mode": "api_key_env", "key_env": "OPENAI_API_KEY"},
            "max_tokens": 8192,
        },
        "crud": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "auth": {"mode": "api_key_env", "key_env": "OPENAI_API_KEY"},
            "max_tokens": 4096,
        },
        "structured": {
            "provider": "openai",
            "model": "gpt-4.1",
            "auth": {"mode": "api_key_env", "key_env": "OPENAI_API_KEY"},
            "max_tokens": 8192,
        },
    },
}

_TEMPLATE_HEADER = """# Runtime fields: upstream, tools, premium_tools.
# Reserved/validated fields: auth, wire, endpoints, model_defaults,
# tool_choice, reasoning, fallback.
# Adapter selection still comes from config.yaml router rules and request path.
"""


def _provider_template(group: str) -> JsonObject:
    """Build one provider YAML template from built-in defaults."""
    return cast(
        "JsonObject",
        {
            "group": group,
            "upstream": _UPSTREAM_BY_GROUP[group],
            "auth": {"mode": "oauth_passthrough"},
            "wire": AGENT_GROUP_WIRES[group],
            "endpoints": list(AGENT_GROUP_ENDPOINTS[group]),
            "model_defaults": {},
            "tool_choice": "passthrough",
            "reasoning": "strip",
            "tools": dict(AGENT_DEFAULT_TOOLS[group]),
            "premium_tools": list(AGENT_DEFAULT_PREMIUM_TOOLS[group]),
            "tiers": copy.deepcopy(_TIERS_BY_GROUP[group]),
            "fallback": "passthrough",
        },
    )


def ensure_provider_configs(*, ask: Callable[[str], str] | None = None) -> list[Path]:
    """Create missing per-provider YAML files.

    Args:
        ask: Reserved for future interactive bootstraps; ignored today.

    Returns:
        Paths created during this call, or an empty list on no-op.
    """
    del ask
    created: list[Path] = []
    config.provider_config_dir().mkdir(parents=True, exist_ok=True)
    for group in sorted(_BOOTSTRAP_GROUPS):
        path = config.provider_config_path(group)
        if path.exists():
            continue
        path.write_text(
            f"{_TEMPLATE_HEADER}{yaml.safe_dump(_provider_template(group), sort_keys=False)}",
            encoding="utf-8",
        )
        created.append(path)
    return created
