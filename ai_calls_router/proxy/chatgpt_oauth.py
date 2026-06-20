"""ChatGPT OAuth header helpers for Codex-backend routing.

Hermes clients authenticate to the ChatGPT backend (``chatgpt.com/backend-api``)
with a ChatGPT OAuth bearer and a ``chatgpt-account-id``. Both the premium
passthrough path and the OAuth-tier routed serving path need to detect that auth
and build the forwardable upstream headers. These helpers are pure (no transport
state) so they can be shared without dragging in the HTTP or WebSocket layers.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ai_calls_router._lib.types import JsonValue

_CHATGPT_ACCOUNT_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"
# Hop-by-hop headers plus body-framing headers. The framing headers
# (content-length, content-encoding, transfer-encoding) describe the original
# client body; every consumer of these headers re-serializes the body (the
# Codex direct path posts json=payload, passthrough rebuilds via build_request),
# so a forwarded content-length over-declares the new body and the upstream send
# fails with "Too little data for declared Content-Length". httpx recomputes the
# correct content-length from the actual payload once the stale one is gone.
_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "upgrade",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }
)


def codex_chatgpt_headers(headers: Mapping[str, str]) -> list[tuple[str, str]] | None:
    """Build upstream headers and detect ChatGPT OAuth.

    Args:
        headers: Client request headers.

    Returns:
        Forwardable headers, or ``None`` when the request is not ChatGPT-auth.
    """
    account_id = headers.get("chatgpt-account-id") or _account_id_from_auth(headers)
    if not account_id:
        return None
    result = _forwardable_headers(headers)
    if not _has_header(result, "chatgpt-account-id"):
        result.append(("ChatGPT-Account-ID", account_id))
    if not _has_header(result, "originator"):
        result.append(("originator", "codex_cli_rs"))
    return result


def _forwardable_headers(headers: Mapping[str, str]) -> list[tuple[str, str]]:
    """Return request headers minus hop-by-hop headers."""
    return [(key, value) for key, value in headers.items() if key.lower() not in _HOP_HEADERS]


def _has_header(headers: Iterable[tuple[str, str]], name: str) -> bool:
    """Return whether a header name is already present (case-insensitive)."""
    expected = name.lower()
    return any(key.lower() == expected for key, _ in headers)


def _account_id_from_auth(headers: Mapping[str, str]) -> str | None:
    """Return the ChatGPT account id encoded in the bearer JWT, if any."""
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = _decode_jwt_payload(parts[1])
    if not isinstance(payload, dict):
        return None
    account_id = payload.get(_CHATGPT_ACCOUNT_CLAIM)
    return account_id if isinstance(account_id, str) and account_id else None


def _decode_jwt_payload(segment: str) -> JsonValue:
    """Decode a base64url JWT payload segment into JSON, or ``None`` on failure."""
    padded = segment + "=" * (-len(segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return cast("JsonValue", json.loads(decoded))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
