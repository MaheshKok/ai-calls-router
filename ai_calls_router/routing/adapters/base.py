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
