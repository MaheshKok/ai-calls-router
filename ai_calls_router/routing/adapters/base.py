"""Define the client-adapter protocol used by routed requests.

Adapters isolate client-facing wire formats from the internal Anthropic Messages representation.
Phase 1 installs the protocol and shared identity vocabulary for supported
client formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_calls_router._lib.types import JsonObject

KNOWN_GROUPS: frozenset[str] = frozenset({"claude_code", "hermes"})

AGENT_GROUP_WIRES: dict[str, str] = {
    "claude_code": "anthropic_messages",
    "hermes": "openai_chat",
}

AGENT_GROUP_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "claude_code": ("/v1/messages",),
    "hermes": ("/v1/chat/completions", "/v1/responses"),
}


class ClientAdapter(Protocol):
    """Describe a client format bridge for the routing engine.

    Implementations translate incoming client bodies into the Anthropic Messages shape
    consumed by the existing router, then translate routed responses back to the client.
    """

    default_agent_group: str

    def extract_pending_tools(self, body: JsonObject) -> list[str]: ...

    def to_anthropic_request(self, body: JsonObject) -> JsonObject: ...

    def to_client_response(self, anthropic_response: JsonObject) -> JsonObject: ...

    def to_client_sse(self, anthropic_response: JsonObject) -> Iterator[bytes]: ...
