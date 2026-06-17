"""Resolve client adapters for routed request paths.

Each client wire format has a small adapter that converts edge requests into
the router's Anthropic Messages canonical format. Agent identity is resolved
at the server boundary from the assembled provider routing config.
"""

from __future__ import annotations

from ai_calls_router.routing.adapters.anthropic_messages import AnthropicMessagesAdapter
from ai_calls_router.routing.adapters.base import KNOWN_GROUPS, ClientAdapter
from ai_calls_router.routing.adapters.openai_chat import OpenAIChatAdapter
from ai_calls_router.routing.adapters.openai_responses import OpenAIResponsesAdapter

__all__ = [
    "KNOWN_GROUPS",
    "AnthropicMessagesAdapter",
    "ClientAdapter",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "adapter_for_path",
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
    if path == "/v1/responses":
        return OpenAIResponsesAdapter()
    return None
