"""Spec-derived tests for the LiteLLM-path headroom compression wrapper.

Contract under test (from compress_litellm_messages' docstring and signature):
the wrapper runs headroom over already-converted OpenAI messages, measures the
string-content character delta, and reports it as a ShrinkStats. It is
best-effort: an absent headroom (``_load_compressor`` returns None) or a raising
``compress`` must leave the messages untouched and report a no-op ("none") stat,
never break the caller. A successful run returns the compressor's own message
list with path "compress". The tests inject a fake compressor through the
``_load_compressor`` seam so no real headroom dependency is required.
"""

from __future__ import annotations

import logging

import pytest

from ai_calls_router.routing import compression


def _openai_messages(tool_content: str) -> list[dict[str, object]]:
    """Build a minimal OpenAI-shaped conversation with one tool result.

    Args:
        tool_content: String content of the ``role="tool"`` message.

    Returns:
        A user/assistant/tool message list in the post-conversion wire shape.
    """
    return [
        {"role": "user", "content": "investigate"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": tool_content},
    ]


def _shrink_tool_content(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a new message list with role=tool content halved.

    Mirrors what a real compressor does to the converted messages: it shrinks
    string content in place of a message, producing a new list (the input is
    never mutated, matching the wrapper's no-mutation contract).
    """
    out: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            out.append({**message, "content": content[: len(content) // 2]})
        else:
            out.append(message)
    return out


class _FakeCompressor:
    """Records calls and returns a CompressResult-shaped object."""

    def __init__(self, transform: object, transforms_applied: object = None) -> None:
        """Store the transform applied to the messages on each call.

        Args:
            transform: Callable mapping the input messages to compressed output.
            transforms_applied: Optional value for the result's
                ``transforms_applied`` attribute; when None the attribute is
                omitted entirely so the defensive getattr path is exercised.
        """
        self.calls: list[dict[str, object]] = []
        self._transform = transform
        self._transforms_applied = transforms_applied

    def __call__(self, messages: list[dict[str, object]], **kwargs: object) -> object:
        """Record kwargs and return an object exposing ``.messages``."""
        self.calls.append({"messages": messages, **kwargs})
        from types import SimpleNamespace

        result = SimpleNamespace(messages=self._transform(messages))
        if self._transforms_applied is not None:
            result.transforms_applied = self._transforms_applied
        return result


@pytest.fixture(autouse=True)
def _clear_compressor_cache() -> object:
    """Reset the lru_cache around _load_compressor between tests."""
    compression._load_compressor.cache_clear()
    yield
    compression._load_compressor.cache_clear()


class TestCompressLitellmMessages:
    def test_compressible_tool_output_is_shrunk_and_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A non-excluded tool with a long string result: the compressor's output
        # is returned verbatim and the char delta is reported as path "compress".
        fake = _FakeCompressor(_shrink_tool_content)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)
        messages = _openai_messages("x" * 400)

        out, stats = compression.compress_litellm_messages(messages, model="gpt-5.5")

        assert out[-1]["content"] == "x" * 200
        assert stats.path == "compress"
        assert stats.chars_before == len("investigate") + 400
        assert stats.chars_after == len("investigate") + 200
        assert stats.chars_saved == 200

    def test_model_and_limit_forwarded_to_compressor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The wrapper must hand headroom the routed model and the token budget so
        # headroom selects the right tokenizer/limit; optimize is always on.
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(
            _openai_messages("log"), model="gpt-5.5", model_limit=123_456
        )

        call = fake.calls[0]
        assert call["model"] == "gpt-5.5"
        assert call["model_limit"] == 123_456
        assert call["optimize"] is True

    def test_default_model_limit_is_used_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(_openai_messages("log"), model="gpt-5.5")

        assert fake.calls[0]["model_limit"] == compression.DEFAULT_MODEL_LIMIT

    def test_uncompressible_output_reports_zero_saved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A compressor that returns its input unchanged (the realistic outcome
        # for an excluded coding-agent tool whose output headroom leaves verbatim)
        # must yield chars_before == chars_after and zero savings, not an error.
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)
        messages = _openai_messages("y" * 300)

        out, stats = compression.compress_litellm_messages(messages, model="gpt-5.5")

        assert out[-1]["content"] == "y" * 300
        assert stats.chars_before == stats.chars_after
        assert stats.chars_saved == 0

    def test_absent_headroom_passes_messages_through_as_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _load_compressor returns None when headroom-ai is not installed: the
        # input list is returned by identity and the stat is "none", so the
        # caller still serves the request.
        monkeypatch.setattr(compression, "_load_compressor", lambda: None)
        messages = _openai_messages("z" * 250)

        out, stats = compression.compress_litellm_messages(messages, model="gpt-5.5")

        assert out is messages
        assert stats.path == "none"
        assert stats.chars_before == stats.chars_after
        assert stats.chars_saved == 0

    def test_raising_compressor_falls_back_to_verbatim_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Compression is best-effort: a compress() that raises must not propagate.
        # The wrapper returns the input unchanged, reports "none", and logs the
        # failure with context (never silently swallowed).
        def _boom(messages: list[dict[str, object]], **_: object) -> object:
            raise RuntimeError("tokenizer exploded")

        monkeypatch.setattr(compression, "_load_compressor", lambda: _boom)
        messages = _openai_messages("w" * 200)

        with caplog.at_level(logging.ERROR, logger="acr.compression"):
            out, stats = compression.compress_litellm_messages(messages, model="gpt-5.5")

        assert out is messages
        assert stats.path == "none"
        assert stats.chars_saved == 0
        assert any("compression failed" in record.message for record in caplog.records)

    def test_empty_messages_do_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        out, stats = compression.compress_litellm_messages([], model="gpt-5.5")

        assert out == []
        assert stats.chars_before == 0
        assert stats.chars_after == 0

    def test_non_string_and_none_content_contribute_zero_chars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The assistant message has content=None and a list tool_calls; only the
        # string user content is counted, so a no-op compressor reports exactly
        # that length with no crash on the non-string fields.
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "system", "content": ["not", "a", "string"]},
        ]

        _, stats = compression.compress_litellm_messages(messages, model="gpt-5.5")

        assert stats.chars_before == len("hello")
        assert stats.chars_after == len("hello")

    def test_same_input_yields_identical_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Determinism: the wrapper adds no nondeterminism of its own, so two runs
        # over equal inputs produce equal compressed output and equal stats.
        monkeypatch.setattr(
            compression, "_load_compressor", lambda: _FakeCompressor(_shrink_tool_content)
        )
        first_out, first_stats = compression.compress_litellm_messages(
            _openai_messages("d" * 400), model="gpt-5.5"
        )
        second_out, second_stats = compression.compress_litellm_messages(
            _openai_messages("d" * 400), model="gpt-5.5"
        )

        assert first_out == second_out
        assert (first_stats.chars_before, first_stats.chars_after) == (
            second_stats.chars_before,
            second_stats.chars_after,
        )


class TestTextMlToggle:
    """The enable_text_ml flag gates headroom's lossy ML plain-text compressor.

    Contract: text-ML is off by default, so the wrapper passes
    ``kompress_model="disabled"`` to headroom; opting a tier in omits that kwarg
    so headroom's default (ML on) applies. Asserted through the fake compressor's
    recorded kwargs so no real ML stack is needed.
    """

    def test_default_disables_text_ml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Omitting the flag must disable Kompress so merely installing the ML
        # extra never changes behaviour for an un-opted-in tier.
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(_openai_messages("log"), model="gpt-5.5")

        assert fake.calls[0]["kompress_model"] == "disabled"

    def test_explicit_false_disables_text_ml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(
            _openai_messages("log"), model="gpt-5.5", enable_text_ml=False
        )

        assert fake.calls[0]["kompress_model"] == "disabled"

    def test_enabled_omits_kompress_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Opting in must NOT pass kompress_model, leaving headroom's default
        # (ML on) in force. The key must be absent, not merely non-"disabled".
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(
            _openai_messages("log"), model="gpt-5.5", enable_text_ml=True
        )

        assert "kompress_model" not in fake.calls[0]

    def test_toggle_does_not_disturb_model_and_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The flag must not perturb the other forwarded kwargs.
        fake = _FakeCompressor(lambda m: m)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        compression.compress_litellm_messages(
            _openai_messages("log"), model="gpt-5.5", model_limit=999, enable_text_ml=True
        )

        call = fake.calls[0]
        assert call["model"] == "gpt-5.5"
        assert call["model_limit"] == 999
        assert call["optimize"] is True


class TestSummarizeContentTypes:
    """summarize_content_types reduces headroom's router:* markers to a label tuple.

    Contract (from the docstring): strip the ``router:`` prefix; for a nested
    ``tool_result``/``text_block`` head the label is the SECOND segment, otherwise
    the FIRST; dedupe preserving first-seen order. Expected tuples are derived by
    hand from that rule, never by echoing the implementation.
    """

    def test_single_strategy_marker_maps_to_its_label(self) -> None:
        assert compression.summarize_content_types(["router:smart_crusher:0.34"]) == (
            "smart_crusher",
        )

    def test_duplicate_markers_are_deduped(self) -> None:
        assert compression.summarize_content_types(
            ["router:excluded:tool", "router:excluded:tool"]
        ) == ("excluded",)

    def test_nested_heads_take_the_second_segment(self) -> None:
        # tool_result/text_block are containers; the real strategy is the child.
        assert compression.summarize_content_types(
            ["router:tool_result:smart_crusher", "router:text_block:kompress"]
        ) == ("smart_crusher", "kompress")

    def test_first_seen_order_is_preserved(self) -> None:
        assert compression.summarize_content_types(
            ["router:protected:user_message", "router:smart_crusher:0.34"]
        ) == ("protected", "smart_crusher")

    def test_bare_marker_without_strategy_maps_to_its_head(self) -> None:
        assert compression.summarize_content_types(["router:noop"]) == ("noop",)

    def test_empty_marker_list_yields_empty_tuple(self) -> None:
        assert compression.summarize_content_types([]) == ()

    def test_nested_head_without_child_falls_back_to_head(self) -> None:
        # A malformed nested marker missing its strategy segment must not IndexError;
        # it degrades to the head label rather than crashing the serve path.
        assert compression.summarize_content_types(["router:tool_result"]) == ("tool_result",)

    def test_mixed_nested_and_flat_dedupe_across_forms(self) -> None:
        # tool_result:smart_crusher and a bare smart_crusher collapse to one label.
        assert compression.summarize_content_types(
            ["router:tool_result:smart_crusher", "router:smart_crusher:0.5"]
        ) == ("smart_crusher",)


class TestContentTypesPropagation:
    """compress_litellm_messages surfaces headroom's markers on the returned stat.

    The wrapper reads ``result.transforms_applied`` (router:* markers) and reduces
    them via summarize_content_types onto ShrinkStats.content_types. Fail-open and
    header-absent paths must leave content_types empty.
    """

    def test_success_carries_reduced_markers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeCompressor(
            _shrink_tool_content,
            transforms_applied=[
                "router:protected:user_message",
                "router:smart_crusher:0.34",
                "not-a-router-marker",
            ],
        )
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        _, stats = compression.compress_litellm_messages(
            _openai_messages("x" * 400), model="gpt-5.5"
        )

        assert stats.content_types == ("protected", "smart_crusher")

    def test_result_without_transforms_applied_yields_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake omits transforms_applied entirely (transforms_applied=None): the
        # defensive getattr must degrade to no markers, not raise.
        fake = _FakeCompressor(_shrink_tool_content)
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        _, stats = compression.compress_litellm_messages(
            _openai_messages("x" * 400), model="gpt-5.5"
        )

        assert stats.content_types == ()

    def test_absent_headroom_yields_empty_content_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(compression, "_load_compressor", lambda: None)

        _, stats = compression.compress_litellm_messages(
            _openai_messages("z" * 250), model="gpt-5.5"
        )

        assert stats.content_types == ()

    def test_raising_compressor_yields_empty_content_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(messages: list[dict[str, object]], **_: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(compression, "_load_compressor", lambda: _boom)

        _, stats = compression.compress_litellm_messages(
            _openai_messages("w" * 200), model="gpt-5.5"
        )

        assert stats.content_types == ()

    def test_non_list_transforms_applied_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A build that exposes transforms_applied as a non-list (e.g. a string) must
        # be treated as no markers, not iterated character-by-character.
        fake = _FakeCompressor(_shrink_tool_content, transforms_applied="router:smart_crusher:0.3")
        monkeypatch.setattr(compression, "_load_compressor", lambda: fake)

        _, stats = compression.compress_litellm_messages(
            _openai_messages("x" * 400), model="gpt-5.5"
        )

        assert stats.content_types == ()


class TestLoadCompressor:
    def test_returns_none_and_warns_once_when_headroom_absent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Force the optional import to fail and confirm the seam degrades to None
        # with a single WARNING, so the caller no-ops rather than erroring.
        import builtins

        real_import = builtins.__import__

        def _no_headroom(name: str, *args: object, **kwargs: object) -> object:
            if name == "headroom":
                raise ImportError("no headroom")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_headroom)
        compression._load_compressor.cache_clear()

        with caplog.at_level(logging.WARNING, logger="acr.compression"):
            result = compression._load_compressor()

        assert result is None
        assert sum("not installed" in r.message for r in caplog.records) == 1
