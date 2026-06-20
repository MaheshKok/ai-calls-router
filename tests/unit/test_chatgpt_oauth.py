"""Adversarial tests for the ChatGPT OAuth header helper.

These tests exercise ``codex_chatgpt_headers`` from its contract: it must detect
ChatGPT auth (either an explicit ``chatgpt-account-id`` header or the account id
encoded in the bearer JWT claim), strip hop-by-hop headers, and never duplicate
account-id/originator headers. The JWT claim name is OpenAI's external contract,
so it is pinned as a literal rather than imported from the implementation.
"""

from __future__ import annotations

import base64
import json

from ai_calls_router.proxy.chatgpt_oauth import codex_chatgpt_headers

# OpenAI's ChatGPT account-id JWT claim (external contract, pinned deliberately).
_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"


def _jwt_with_payload(payload: dict[str, object]) -> str:
    """Build a three-segment token whose payload base64url-decodes to ``payload``."""
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return f"header.{body.rstrip('=')}.signature"


def _lower_map(headers: list[tuple[str, str]]) -> dict[str, str]:
    """Return a lowercased single-value view of header pairs."""
    return {key.lower(): value for key, value in headers}


def test_returns_none_when_no_account_id_anywhere() -> None:
    assert codex_chatgpt_headers({"authorization": "Bearer plain-api-key"}) is None


def test_returns_none_when_headers_empty() -> None:
    assert codex_chatgpt_headers({}) is None


def test_returns_none_for_non_bearer_authorization() -> None:
    token = _jwt_with_payload({_CLAIM: "acct_jwt"})
    assert codex_chatgpt_headers({"authorization": f"Basic {token}"}) is None


def test_account_id_from_explicit_header_is_forwarded_with_originator() -> None:
    result = codex_chatgpt_headers(
        {"chatgpt-account-id": "acct_header", "authorization": "Bearer opaque"}
    )
    assert result is not None
    headers = _lower_map(result)
    assert headers["chatgpt-account-id"] == "acct_header"
    assert headers["originator"] == "codex_cli_rs"


def test_account_id_recovered_from_jwt_claim_when_header_absent() -> None:
    token = _jwt_with_payload({"sub": "user-1", _CLAIM: "acct_jwt"})
    result = codex_chatgpt_headers({"authorization": f"Bearer {token}"})
    assert result is not None
    assert _lower_map(result)["chatgpt-account-id"] == "acct_jwt"


def test_explicit_header_wins_over_jwt_claim() -> None:
    token = _jwt_with_payload({_CLAIM: "acct_jwt"})
    result = codex_chatgpt_headers(
        {"chatgpt-account-id": "acct_header", "authorization": f"Bearer {token}"}
    )
    assert result is not None
    account_values = [value for key, value in result if key.lower() == "chatgpt-account-id"]
    assert account_values == ["acct_header"]


def test_single_segment_token_yields_none() -> None:
    assert codex_chatgpt_headers({"authorization": "Bearer onlyonesegment"}) is None


def test_malformed_base64_payload_yields_none() -> None:
    assert codex_chatgpt_headers({"authorization": "Bearer header.!!!not-base64!!!.sig"}) is None


def test_non_json_payload_yields_none() -> None:
    body = base64.urlsafe_b64encode(b"not-json").decode("ascii").rstrip("=")
    assert codex_chatgpt_headers({"authorization": f"Bearer header.{body}.sig"}) is None


def test_jwt_without_account_claim_yields_none() -> None:
    token = _jwt_with_payload({"sub": "user-1"})
    assert codex_chatgpt_headers({"authorization": f"Bearer {token}"}) is None


def test_empty_claim_string_treated_as_missing() -> None:
    token = _jwt_with_payload({_CLAIM: ""})
    assert codex_chatgpt_headers({"authorization": f"Bearer {token}"}) is None


def test_non_string_claim_value_treated_as_missing() -> None:
    token = _jwt_with_payload({_CLAIM: 12345})
    assert codex_chatgpt_headers({"authorization": f"Bearer {token}"}) is None


def test_hop_by_hop_headers_are_stripped() -> None:
    result = codex_chatgpt_headers(
        {
            "chatgpt-account-id": "acct_header",
            "host": "chatgpt.com",
            "connection": "keep-alive",
            "upgrade": "websocket",
            "sec-websocket-key": "abc",
            "sec-websocket-version": "13",
            "authorization": "Bearer opaque",
            "content-type": "application/json",
        }
    )
    assert result is not None
    keys = {key.lower() for key, _ in result}
    assert keys.isdisjoint(
        {"host", "connection", "upgrade", "sec-websocket-key", "sec-websocket-version"}
    )
    assert {"authorization", "content-type"} <= keys


def test_body_framing_headers_are_stripped() -> None:
    # The body is re-serialized downstream (Codex direct posts json=payload),
    # so a forwarded content-length over-declares the new body and the upstream
    # send fails with "Too little data for declared Content-Length". The builder
    # must drop content-length/content-encoding/transfer-encoding.
    result = codex_chatgpt_headers(
        {
            "chatgpt-account-id": "acct_header",
            "authorization": "Bearer opaque",
            "content-length": "99999",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "content-type": "application/json",
        }
    )
    assert result is not None
    keys = {key.lower() for key, _ in result}
    assert keys.isdisjoint({"content-length", "content-encoding", "transfer-encoding"})
    assert {"authorization", "content-type"} <= keys


def test_existing_account_id_and_originator_are_not_duplicated() -> None:
    result = codex_chatgpt_headers(
        {
            "chatgpt-account-id": "acct_header",
            "originator": "custom_originator",
            "authorization": "Bearer opaque",
        }
    )
    assert result is not None
    account_values = [value for key, value in result if key.lower() == "chatgpt-account-id"]
    originator_values = [value for key, value in result if key.lower() == "originator"]
    assert account_values == ["acct_header"]
    assert originator_values == ["custom_originator"]
