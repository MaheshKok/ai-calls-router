"""Resolve client adapters for routed request paths.

Phase 1 wires only the Anthropic Messages endpoint so existing Claude Code traffic stays
on the identity path. The agent-group resolver intentionally implements the small Phase 1
header precedence and leaves richer policy for later phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_calls_router.routing.adapters.anthropic_messages import AnthropicMessagesAdapter
from ai_calls_router.routing.adapters.base import KNOWN_GROUPS, ClientAdapter
from ai_calls_router.routing.adapters.openai_chat import OpenAIChatAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "KNOWN_GROUPS",
    "AnthropicMessagesAdapter",
    "ClientAdapter",
    "OpenAIChatAdapter",
    "adapter_for_path",
    "resolve_agent_group",
]


def adapter_for_path(path: str) -> ClientAdapter | None:
    """Return the Phase 1 adapter for a request path.

    Args:
        path: Incoming request URL path.

    Returns:
        The Anthropic Messages identity adapter for `/v1/messages`, otherwise `None`.
    """
    if path == "/v1/messages":
        return AnthropicMessagesAdapter()
    if path == "/v1/chat/completions":
        return OpenAIChatAdapter()
    return None


def resolve_agent_group(default_group: str, headers: Mapping[str, str]) -> str:
    """Resolve the Phase 1 agent group from request headers.

    Args:
        default_group: Adapter default group to use when no trusted override exists.
        headers: Incoming request headers.

    Returns:
        A known `x-acr-agent` value when present, otherwise the adapter default.
    """
    candidate = headers.get("x-acr-agent")
    if candidate in KNOWN_GROUPS:
        return candidate
    return default_group
