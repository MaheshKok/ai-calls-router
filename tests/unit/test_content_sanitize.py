"""Tests for blank-text-block stripping in content_sanitize.

The Anthropic Messages API rejects any text content block whose text is empty or
whitespace-only. These tests pin the two public helpers: clean_response_content
(applied to routed responses so the proxy never emits a blank block) and
clean_request_messages (applied to outbound bodies so an already-poisoned
history still serves). Cases are derived from that contract -- boundary text
values, blocks alongside tool_use, message-emptying guards, and input
immutability -- not from the implementation.
"""

from __future__ import annotations

from ai_calls_router._lib.types import JsonObject
from ai_calls_router.routing import content_sanitize


def _blank() -> JsonObject:
    """Return a blank (empty-text) content block."""
    return {"type": "text", "text": ""}


def _tool_use() -> JsonObject:
    """Return a tool_use content block."""
    return {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}


class TestCleanResponseContent:
    def test_strips_blank_block_preceding_tool_use(self) -> None:
        response: JsonObject = {
            "id": "msg_1",
            "content": [_blank(), _tool_use()],
        }
        cleaned = content_sanitize.clean_response_content(response)
        assert cleaned["content"] == [_tool_use()]

    def test_strips_blank_block_keeping_real_text(self) -> None:
        response: JsonObject = {"content": [_blank(), {"type": "text", "text": "hi"}]}
        cleaned = content_sanitize.clean_response_content(response)
        assert cleaned["content"] == [{"type": "text", "text": "hi"}]

    def test_strips_whitespace_only_text_block(self) -> None:
        response: JsonObject = {"content": [{"type": "text", "text": " \n\t "}, _tool_use()]}
        cleaned = content_sanitize.clean_response_content(response)
        assert cleaned["content"] == [_tool_use()]

    def test_keeps_zero_string_text_block(self) -> None:
        # "0" is non-blank after strip; only empty/whitespace text is dropped.
        response: JsonObject = {"content": [{"type": "text", "text": "0"}]}
        cleaned = content_sanitize.clean_response_content(response)
        assert cleaned["content"] == [{"type": "text", "text": "0"}]

    def test_returns_same_object_when_no_blank_block(self) -> None:
        response: JsonObject = {"content": [{"type": "text", "text": "hi"}, _tool_use()]}
        assert content_sanitize.clean_response_content(response) is response

    def test_returns_same_object_when_content_not_a_list(self) -> None:
        response: JsonObject = {"content": "plain"}
        assert content_sanitize.clean_response_content(response) is response

    def test_returns_same_object_when_content_missing(self) -> None:
        response: JsonObject = {"id": "msg_1"}
        assert content_sanitize.clean_response_content(response) is response

    def test_all_blank_content_collapses_to_empty_list(self) -> None:
        response: JsonObject = {"content": [_blank(), {"type": "text", "text": "  "}]}
        cleaned = content_sanitize.clean_response_content(response)
        assert cleaned["content"] == []

    def test_does_not_mutate_input(self) -> None:
        content = [_blank(), _tool_use()]
        response: JsonObject = {"content": content}
        content_sanitize.clean_response_content(response)
        assert content == [_blank(), _tool_use()]
        assert response["content"] is content


class TestCleanRequestMessages:
    def test_strips_blank_block_alongside_tool_use(self) -> None:
        body: JsonObject = {
            "model": "m",
            "messages": [{"role": "assistant", "content": [_blank(), _tool_use()]}],
        }
        cleaned = content_sanitize.clean_request_messages(body)
        assert cleaned is not None
        assert cleaned["messages"][0]["content"] == [_tool_use()]

    def test_returns_none_for_clean_body(self) -> None:
        # Clean bodies must be reported unchanged so the caller forwards the
        # original bytes byte-identically and keeps the upstream prompt cache.
        body: JsonObject = {
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
        }
        assert content_sanitize.clean_request_messages(body) is None

    def test_leaves_message_intact_when_stripping_would_empty_it(self) -> None:
        # A message whose only block is blank is left as-is: an empty content
        # array is itself invalid, so the blank block is left for upstream.
        body: JsonObject = {"messages": [{"role": "assistant", "content": [_blank()]}]}
        assert content_sanitize.clean_request_messages(body) is None

    def test_cleans_only_the_poisoned_message(self) -> None:
        body: JsonObject = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [_blank(), _tool_use()]},
            ],
        }
        cleaned = content_sanitize.clean_request_messages(body)
        assert cleaned is not None
        assert cleaned["messages"][0]["content"] == [{"type": "text", "text": "q"}]
        assert cleaned["messages"][1]["content"] == [_tool_use()]

    def test_emptyable_message_kept_while_sibling_is_cleaned(self) -> None:
        body: JsonObject = {
            "messages": [
                {"role": "assistant", "content": [_blank()]},
                {"role": "assistant", "content": [_blank(), _tool_use()]},
            ],
        }
        cleaned = content_sanitize.clean_request_messages(body)
        assert cleaned is not None
        assert cleaned["messages"][0]["content"] == [_blank()]
        assert cleaned["messages"][1]["content"] == [_tool_use()]

    def test_returns_none_when_messages_missing(self) -> None:
        assert content_sanitize.clean_request_messages({"model": "m"}) is None

    def test_returns_none_when_messages_not_a_list(self) -> None:
        assert content_sanitize.clean_request_messages({"messages": "nope"}) is None

    def test_ignores_string_content_messages(self) -> None:
        body: JsonObject = {"messages": [{"role": "user", "content": "hello"}]}
        assert content_sanitize.clean_request_messages(body) is None

    def test_ignores_non_dict_message_entries(self) -> None:
        body: JsonObject = {"messages": ["weird", {"role": "user", "content": "hi"}]}
        assert content_sanitize.clean_request_messages(body) is None

    def test_does_not_mutate_input(self) -> None:
        content = [_blank(), _tool_use()]
        body: JsonObject = {"messages": [{"role": "assistant", "content": content}]}
        content_sanitize.clean_request_messages(body)
        assert content == [_blank(), _tool_use()]
