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
    "hermes": "https://chatgpt.com/backend-api/codex",
}
_BOOTSTRAP_GROUPS: tuple[str, ...] = ("claude_code", "hermes")
# Cheap claude_code turns route to Sonnet 4.6 on the subscription OAuth bearer
# (auth.mode: oauth) so they draw the subscription's separate Sonnet quota
# instead of a pay-per-token key; the premium (Opus) passthrough is untouched and
# a Sonnet-quota 429 fails open to it. effort pins each tier's routed reasoning
# level (Sonnet rejects the Opus-only "xhigh" that Claude Code sends). hermes
# cheap turns route to Codex on the ChatGPT subscription OAuth bearer.
_TIERS_BY_GROUP: dict[str, JsonObject] = {
    "claude_code": {
        "fast": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "auth": {"mode": "oauth"},
            "effort": "low",
            "input_cost_per_1m": 3.0,
            "input_cached_cost_per_1m": 0.30,
            "output_cost_per_1m": 15.0,
        },
        "code": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "auth": {"mode": "oauth"},
            "effort": "high",
            "input_cost_per_1m": 3.0,
            "input_cached_cost_per_1m": 0.30,
            "output_cost_per_1m": 15.0,
        },
        "crud": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "auth": {"mode": "oauth"},
            "max_tokens": 4096,
            "effort": "low",
            "input_cost_per_1m": 3.0,
            "input_cached_cost_per_1m": 0.30,
            "output_cost_per_1m": 15.0,
        },
        "structured": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4-6",
            "auth": {"mode": "oauth"},
            "effort": "high",
            "input_cost_per_1m": 3.0,
            "input_cached_cost_per_1m": 0.30,
            "output_cost_per_1m": 15.0,
        },
    },
    "hermes": {
        "fast": {
            "provider": "codex",
            "model": "gpt-5.4-mini",
            "auth": {"mode": "oauth"},
            "max_tokens": 8192,
        },
        "code": {
            "provider": "codex",
            "model": "gpt-5.3-codex-spark",
            "auth": {"mode": "oauth"},
            "max_tokens": 8192,
        },
        "crud": {
            "provider": "codex",
            "model": "gpt-5.3-codex-spark",
            "auth": {"mode": "oauth"},
            "max_tokens": 4096,
        },
        "structured": {
            "provider": "codex",
            "model": "gpt-5.3-codex-spark",
            "auth": {"mode": "oauth"},
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
