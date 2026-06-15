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
from typing import Any

import pytest

from ai_calls_router.routing import compression


def _openai_messages(tool_content: str) -> list[dict[str, Any]]:
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


def _shrink_tool_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new message list with role=tool content halved.

    Mirrors what a real compressor does to the converted messages: it shrinks
    string content in place of a message, producing a new list (the input is
    never mutated, matching the wrapper's no-mutation contract).
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            out.append({**message, "content": content[: len(content) // 2]})
        else:
            out.append(message)
    return out


class _FakeCompressor:
    """Records calls and returns a CompressResult-shaped object."""

    def __init__(self, transform: Any) -> None:
        """Store the transform applied to the messages on each call.

        Args:
            transform: Callable mapping the input messages to compressed output.
        """
        self.calls: list[dict[str, Any]] = []
        self._transform = transform

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Record kwargs and return an object exposing ``.messages``."""
        self.calls.append({"messages": messages, **kwargs})
        from types import SimpleNamespace

        return SimpleNamespace(messages=self._transform(messages))


@pytest.fixture(autouse=True)
def _clear_compressor_cache() -> Any:
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
        def _boom(messages: list[dict[str, Any]], **_: Any) -> Any:
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


class TestLoadCompressor:
    def test_returns_none_and_warns_once_when_headroom_absent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Force the optional import to fail and confirm the seam degrades to None
        # with a single WARNING, so the caller no-ops rather than erroring.
        import builtins

        real_import = builtins.__import__

        def _no_headroom(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "headroom":
                raise ImportError("no headroom")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_headroom)
        compression._load_compressor.cache_clear()

        with caplog.at_level(logging.WARNING, logger="acr.compression"):
            result = compression._load_compressor()

        assert result is None
        assert sum("not installed" in r.message for r in caplog.records) == 1
