"""Define the client-adapter protocol used by routed requests.

Adapters isolate client-facing wire formats from the internal Anthropic Messages representation.
Phase 1 only installs the protocol and shared identity vocabulary; later phases add
non-Anthropic client formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

KNOWN_GROUPS: frozenset[str] = frozenset({"claude_code", "codex", "hermes"})

AGENT_GROUP_WIRES: dict[str, str] = {
    "claude_code": "anthropic_messages",
    "codex": "openai_responses",
    "hermes": "openai_chat",
}

AGENT_GROUP_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "claude_code": ("/v1/messages",),
    "codex": ("/v1/responses",),
    "hermes": ("/v1/chat/completions",),
}


class ClientAdapter(Protocol):
    """Describe a client format bridge for the routing engine.

    Implementations translate incoming client bodies into the Anthropic Messages shape
    consumed by the existing router, then translate routed responses back to the client.
    """

    default_agent_group: str

    def extract_pending_tools(self, body: dict[str, Any]) -> list[str]: ...

    def to_anthropic_request(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def to_client_response(self, anthropic_response: dict[str, Any]) -> dict[str, Any]: ...

    def to_client_sse(self, anthropic_response: dict[str, Any]) -> Iterator[bytes]: ...
