"""Spec-derived tests for client-adapter routing seams.

The Phase 1 Anthropic adapter is intentionally identity-preserving so Claude Code traffic
remains byte-stable. These tests pin the path resolver and the one-chunk SSE wrapper.
"""

from __future__ import annotations

import pytest

from ai_calls_router._lib.types import JsonObject
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import synthesis
from ai_calls_router.routing.adapters import adapter_for_path
from ai_calls_router.routing.adapters.anthropic_messages import AnthropicMessagesAdapter


def _body_with_tool_results(*pairs: tuple[str, str]) -> JsonObject:
    """Build a minimal Anthropic request with pending tool results.

    Args:
        pairs: Tool-use identifiers paired with their tool names.

    Returns:
        A request body containing matching assistant tool_use and user tool_result blocks.
    """
    assistant_blocks = [
        {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}}
        for tool_id, tool_name in pairs
    ]
    user_blocks = [{"type": "tool_result", "tool_use_id": tool_id} for tool_id, _ in pairs]
    return {
        "model": "claude-test",
        "messages": [
            {"role": "assistant", "content": assistant_blocks},
            {"role": "user", "content": user_blocks},
        ],
    }


class TestAdapterForPath:
    """Verify Phase 1 path-to-adapter resolution."""

    def test_messages_path_returns_anthropic_adapter(self) -> None:
        """Return the identity adapter for the Anthropic Messages endpoint."""
        assert isinstance(adapter_for_path("/v1/messages"), AnthropicMessagesAdapter)

    def test_other_paths_return_none(self) -> None:
        """Leave non-Messages endpoints outside the Phase 1 seam."""
        assert adapter_for_path("/unknown") is None


class TestAnthropicMessagesAdapter:
    """Verify the identity behavior of the Anthropic Messages adapter."""

    def test_request_and_response_conversions_preserve_identity(self) -> None:
        """Return input dictionaries by identity without copying or mutating them."""
        adapter = AnthropicMessagesAdapter()
        request_body = _body_with_tool_results(("t1", "Bash"))
        response_body = {"content": [{"type": "text", "text": "ok"}]}

        assert adapter.to_anthropic_request(request_body) is request_body
        assert adapter.to_client_response(response_body) is response_body
        assert request_body == _body_with_tool_results(("t1", "Bash"))
        assert response_body == {"content": [{"type": "text", "text": "ok"}]}

    def test_system_role_message_is_normalized_for_routed_path(self) -> None:
        """Move Claude-style system messages without mutating the original body."""
        adapter = AnthropicMessagesAdapter()
        request_body: JsonObject = {
            "model": "claude-test",
            "system": "base",
            "messages": [
                {"role": "system", "content": "runtime"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1"}],
                },
            ],
        }

        routed_body = adapter.to_anthropic_request(request_body)

        assert routed_body is not request_body
        assert routed_body["system"] == "base\n\nruntime"
        routed_messages = routed_body["messages"]
        assert routed_messages == [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
        ]
        original_messages = request_body["messages"]
        assert isinstance(original_messages, list)
        assert original_messages[0] == {"role": "system", "content": "runtime"}
        assert request_body["system"] == "base"
        assert adapter.extract_pending_tools(request_body) == ["Bash"]

    def test_malformed_request_raises_for_fail_open_server_path(self) -> None:
        """Reject malformed Anthropic envelopes before routed decisions."""
        adapter = AnthropicMessagesAdapter()

        with pytest.raises(ValueError, match="Field required"):
            adapter.to_anthropic_request({"messages": []})

    def test_extract_pending_tools_matches_existing_routing_logic(self) -> None:
        """Delegate pending-tool extraction to the existing routing function."""
        adapter = AnthropicMessagesAdapter()
        empty_body: JsonObject = {}
        tool_body = _body_with_tool_results(("t1", "Read"), ("t2", "Bash"))

        assert adapter.extract_pending_tools(empty_body) == routing.pending_tool_names(empty_body)
        assert adapter.extract_pending_tools(empty_body) == []
        assert adapter.extract_pending_tools(tool_body) == routing.pending_tool_names(tool_body)
        assert adapter.extract_pending_tools(tool_body) == ["Read", "Bash"]

    def test_client_sse_matches_existing_synthesizer_bytes(self) -> None:
        """Yield exactly the pre-refactor Anthropic SSE bytes in one chunk."""
        adapter = AnthropicMessagesAdapter()
        response_body = {
            "id": "msg_routed",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "routed answer"}],
            "model": "claude-fable-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        expected = synthesis.synthesize_sse(response_body)
        chunks = list(adapter.to_client_sse(response_body))

        assert chunks == [expected]
        assert b"".join(chunks) == expected
