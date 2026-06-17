"""Bootstrap provider-specific YAML config files.

Phase 7 keeps global cheap-tier config in config.yaml and materializes one
provider file per agent family. The helper is create-only so edited provider
files are never overwritten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from ai_calls_router._lib import config
from ai_calls_router.routing.adapters.base import (
    AGENT_GROUP_ENDPOINTS,
    AGENT_GROUP_WIRES,
    KNOWN_GROUPS,
)
from ai_calls_router.routing.agent_defaults import (
    AGENT_DEFAULT_PREMIUM_TOOLS,
    AGENT_DEFAULT_TOOLS,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_UPSTREAM_BY_GROUP: dict[str, str] = {
    "claude_code": config.DEFAULT_UPSTREAM,
    "codex": "https://api.openai.com",
    "hermes": config.DEFAULT_UPSTREAM,
}

_TEMPLATE_HEADER = """# Runtime fields: upstream, tools, premium_tools.
# Reserved/validated fields: auth, wire, endpoints, model_defaults,
# tool_choice, reasoning, fallback.
# Adapter selection still comes from config.yaml router rules and request path.
"""


def _provider_template(group: str) -> dict[str, Any]:
    """Build one provider YAML template from built-in defaults."""
    return {
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
        "fallback": "passthrough",
    }


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
    for group in sorted(KNOWN_GROUPS):
        path = config.provider_config_path(group)
        if path.exists():
            continue
        path.write_text(
            f"{_TEMPLATE_HEADER}{yaml.safe_dump(_provider_template(group), sort_keys=False)}",
            encoding="utf-8",
        )
        created.append(path)
    return created
