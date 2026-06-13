"""End-to-end scenario tests that drive error/defensive branches in the proxy.

These tests use the real server application (via ``make_client``) to POST
malformed or edge-case request bodies and verify that the proxy *falls back to
passthrough* instead of crashing — exercising the defensive isinstance/continue
guards in routing, compression, and the server itself.
"""

from __future__ import annotations

from acr_testkit import FakeLitellm, Upstream, make_client, make_response

# ---------------------------------------------------------------------------
# Configs used by the tests below
# ---------------------------------------------------------------------------

FULL_CONFIG = """
server:
  port: 8710

premium:
  provider: anthropic

tiers:
  fast:
    # Non-DeepSeek: keeps this turn on the FakeLitellm path; a DeepSeek tier
    # model would divert to the native direct path and bypass the fake.
    model: groq/acr-e2e-fast
    key_env: ACR_TEST_KEY
    input_cost_per_1m: 1.0
    output_cost_per_1m: 2.0

tools:
  Bash: fast
"""

# A config where the "fast" tier value is a bare string instead of a dict.
MALFORMED_TIER_CONFIG = """
server:
  port: 8711

premium:
  provider: anthropic

tiers:
  fast: "not-a-dict"

tools:
  Bash: fast
"""


# ---------------------------------------------------------------------------
# Helper: build an Anthropic body with a Bash tool_result in the last message
# ---------------------------------------------------------------------------


def _body(*, system: str = "You are a bot.", tool_result_text: str = "ok") -> dict:
    """Return a minimal /v1/messages body with a Bash tool_result."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01Abc123",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01Abc123",
                        "content": tool_result_text,
                    }
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# server.py line 84: JSON array body → passthrough
# ---------------------------------------------------------------------------


class TestJsonArrayBodyPassthrough:
    """A POST body that is a JSON array (not an object) must fall back to passthrough."""

    def test_array_body_falls_back_to_passthrough(self, tmp_path, monkeypatch):
        """A non-dict body makes _try_route return None, so the array reaches
        the upstream untouched (server.py:84)."""
        upstream = Upstream()
        with make_client(FULL_CONFIG, tmp_path, monkeypatch, upstream) as client:
            response = client.post("/v1/messages", json=[1, 2, 3])
            assert response.status_code == 200
            assert upstream.requests
            # The upstream should get the raw JSON array forwarded
            import json

            raw = upstream.requests[0].content
            parsed = json.loads(raw)
            assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# server.py line 95: malformed tier cfg (string value) → passthrough
# ---------------------------------------------------------------------------


class TestMalformedTierConfigPassthrough:
    """When the tier entry in the config is not a dict, routing falls back."""

    def test_string_tier_cfg_falls_back_to_passthrough(self, tmp_path, monkeypatch):
        """tier_cfg is a bare string → _try_route returns None → upstream serves."""
        upstream = Upstream()
        with make_client(MALFORMED_TIER_CONFIG, tmp_path, monkeypatch, upstream) as client:
            body = _body()
            response = client.post("/v1/messages", json=body)
            assert response.status_code == 200
            assert upstream.requests


# ---------------------------------------------------------------------------
# routing.py line 101: leading user message in history
# routing.py line 104: assistant with string content
# ---------------------------------------------------------------------------


class TestPendingToolNamesDefensiveSkips:
    """Messages with non-assistant roles or string content are skipped gracefully."""

    def test_leading_user_message_and_string_content_assistant_pass_through(
        self, tmp_path, monkeypatch
    ):
        """A multi-message body where earlier messages are non-assistant or
        have string content should not crash pending_tool_names; the last
        message still has a tool_result → routing proceeds correctly.

        This exercises the ``continue`` on routing.py:101 (non-dict / non-
        assistant) and :104 (assistant content that is a string, not a list).
        """
        upstream = Upstream()
        # Inject a FakeLitellm that returns a simple routed response so the
        # request actually routes (not passthrough).
        fake = FakeLitellm(response=make_response(text="routed"))
        import ai_calls_router.routing.engine as rc

        monkeypatch.setattr(rc, "load_litellm", lambda: fake)

        with make_client(FULL_CONFIG, tmp_path, monkeypatch, upstream) as client:
            body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    # Leading user message (non-assistant → routing.py:101
                    # "continue" triggers)
                    {"role": "user", "content": "hello"},
                    # Assistant with string content (not a list →
                    # routing.py:104 "continue" triggers)
                    {
                        "role": "assistant",
                        "content": "I will now use a tool.",
                    },
                    # Real assistant tool_use message (produces the id map)
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_01Def456",
                                "name": "Bash",
                                "input": {"command": "ls"},
                            }
                        ],
                    },
                    # User message with the tool_result (this is the *last*
                    # message that triggers routing)
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_01Def456",
                                "content": "file1.txt",
                            }
                        ],
                    },
                ],
            }
            response = client.post("/v1/messages", json=body)
            assert response.status_code == 200
            # FakeLitellm was used → request was routed, not passed through
            assert len(fake.calls) >= 1
