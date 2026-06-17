"""Spec-derived tests for client-adapter routing seams.

The Phase 1 Anthropic adapter is intentionally identity-preserving so Claude Code traffic
remains byte-stable. These tests pin the path resolver and the one-chunk SSE wrapper.
"""

from __future__ import annotations

from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import synthesis
from ai_calls_router.routing.adapters import adapter_for_path
from ai_calls_router.routing.adapters.anthropic_messages import AnthropicMessagesAdapter
from ai_calls_router.routing.adapters.openai_responses import OpenAIResponsesAdapter


def _body_with_tool_results(*pairs: tuple[str, str]) -> dict[str, object]:
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
        "messages": [
            {"role": "assistant", "content": assistant_blocks},
            {"role": "user", "content": user_blocks},
        ]
    }


class TestAdapterForPath:
    """Verify Phase 1 path-to-adapter resolution."""

    def test_messages_path_returns_anthropic_adapter(self) -> None:
        """Return the identity adapter for the Anthropic Messages endpoint."""
        assert isinstance(adapter_for_path("/v1/messages"), AnthropicMessagesAdapter)

    def test_other_paths_return_none(self) -> None:
        """Leave non-Messages endpoints outside the Phase 1 seam."""
        assert adapter_for_path("/unknown") is None

    def test_responses_path_returns_openai_responses_adapter(self) -> None:
        """Return the Responses adapter for the OpenAI Responses endpoint."""
        assert isinstance(adapter_for_path("/v1/responses"), OpenAIResponsesAdapter)


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

    def test_extract_pending_tools_matches_existing_routing_logic(self) -> None:
        """Delegate pending-tool extraction to the existing routing function."""
        adapter = AnthropicMessagesAdapter()
        empty_body: dict[str, object] = {}
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
