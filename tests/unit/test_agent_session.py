"""Tests for agent identity detection and session tracking."""

from __future__ import annotations

import time

from ai_calls_router.accounting import metrics


def test_identify_agent_claude_code_cli() -> None:
    """Agent detection from the stdout-REPL UA reported by claude-code."""
    assert (
        metrics.identify_agent("claude-code/v1.2.3 (node; v20.11.0; linux; cmd)")
        == "claude-code-cli"
    )


def test_identify_agent_claude_desktop() -> None:
    """Agent detection from the desktop app User-Agent."""
    assert metrics.identify_agent("Claude-Desktop/0.45.0 (Darwin; x86_64)") == "claude-desktop"


def test_identify_agent_api_anthropic() -> None:
    """Generic anthropic-SDK User-Agents (Python, JS, openai-compat)."""
    assert metrics.identify_agent("anthropic-python/0.20.0 python/3.11") == "api"


def test_identify_agent_api_node() -> None:
    assert metrics.identify_agent("anthropic-node/1.12.0") == "api"


def test_identify_agent_unknown() -> None:
    """objectthing else surfaces as the agent string truncated."""
    ua = "curl/8.4.0"
    assert metrics.identify_agent(ua) == ua


def test_identify_agent_empty() -> None:
    assert metrics.identify_agent("") == "unknown"
    assert metrics.identify_agent(None) == "unknown"  # type: ignore[arg-type]


# ── session fingerprint ────────────────────────────────────────────────


def test_session_fingerprint_from_system_message() -> None:
    fp1 = metrics.session_fingerprint(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
    )
    fp2 = metrics.session_fingerprint(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
    )
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 16  # 8 hex bytes


def test_session_fingerprint_noopener() -> None:
    """Second+ turns never produce a new session id."""
    fp = metrics.session_fingerprint(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
    )
    assert fp is None


def test_session_fingerprint_malformed() -> None:
    """Non-list / missing messages produce None."""
    assert metrics.session_fingerprint({"msg": 1}) is None  # type: ignore[arg-type]
    assert metrics.session_fingerprint(None) is None  # type: ignore[arg-type]


def test_session_fingerprint_deterministic() -> None:
    msgs = [{"role": "user", "content": "paint"}]
    a = metrics.session_fingerprint(msgs)
    time.sleep(0.05)
    b = metrics.session_fingerprint(msgs)
    assert a == b


def test_session_fingerprint_not_faked_by_deliberate_encoded_stamp() -> None:
    frozen = metrics.session_fingerprint(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    )
    fake = metrics.session_fingerprint(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"hello|session:{frozen}"},
                ],
            }
        ]
    )
    assert frozen != fake, "fingerprint must not be spoofable by content"
