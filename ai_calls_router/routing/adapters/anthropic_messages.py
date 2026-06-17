"""Implement the identity adapter for Anthropic Messages requests.

The current Claude Code path already speaks the internal canonical format, so this adapter
never copies or mutates request and response bodies. It still exposes the same conversion
hooks as future adapters so the server can route through one seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import synthesis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonObject


class AnthropicMessagesAdapter:
    """Bridge Anthropic Messages clients to the existing routing engine unchanged."""

    default_agent_group = "claude_code"

    def extract_pending_tools(self, body: JsonObject) -> list[str]:
        """Return pending tool names using the existing Anthropic request parser.

        Args:
            body: Anthropic Messages request body.

        Returns:
            Tool names extracted by the existing routing decision module.
        """
        return routing.pending_tool_names(body)

    def to_anthropic_request(self, body: JsonObject) -> JsonObject:
        """Return the Anthropic request body by identity.

        Args:
            body: Anthropic Messages request body.

        Returns:
            The same object because Anthropic Messages is the canonical format.
        """
        return body

    def to_client_response(self, anthropic_response: JsonObject) -> JsonObject:
        """Return the Anthropic response body by identity.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Returns:
            The same object because the client already expects Anthropic Messages.
        """
        return anthropic_response

    def to_client_sse(self, anthropic_response: JsonObject) -> Iterator[bytes]:
        """Yield the existing Anthropic SSE synthesis as a single byte chunk.

        Args:
            anthropic_response: Routed Anthropic Messages response body.

        Yields:
            The exact bytes produced by the current SSE synthesizer.
        """
        yield synthesis.synthesize_sse(anthropic_response)
