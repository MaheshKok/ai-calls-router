"""Spec-derived tests for ai_calls_router.compression.

Contract under test: compress_body returns a compressed copy of an Anthropic
request body -- the last keep_recent_messages messages stay byte-identical,
older tool_result text is truncated to max_tool_result_chars, and the input
body is never mutated (the passthrough fallback depends on it). Optional rtk
filtering applies only to old tool results from tools with a known-safe rtk
filter mapping, is fail-open on every subprocess problem, and never runs
when use_rtk is "never". Compression is an optimization, never a gate: any
malformed structure passes through unchanged rather than raising.
"""

from __future__ import annotations

import copy
import subprocess
from typing import Any

import pytest

from ai_calls_router.routing import compression


def _tool_round(tool_use_id: str, tool_name: str, result_content: Any) -> list[dict[str, Any]]:
    """Build an assistant tool_use + user tool_result message pair."""
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }
            ],
        },
    ]


def _padding(count: int) -> list[dict[str, Any]]:
    """Build trivial filler messages to push earlier ones out of the window."""
    messages: list[dict[str, Any]] = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"filler {i}"})
    return messages


def _settings(**compression_overrides: Any) -> dict[str, Any]:
    """Build a settings dict with rtk disabled and overridable knobs."""
    cfg: dict[str, Any] = {"use_rtk": "never"}
    cfg.update(compression_overrides)
    return {"compress_routed": True, "compression": cfg}


class TestSkipPaths:
    def test_disabled_returns_same_object(self) -> None:
        body = {"messages": [{"role": "user", "content": "x" * 10_000}]}
        result = compression.compress_body(body, {"compress_routed": False})
        assert result is body

    def test_missing_messages_returns_same_object(self) -> None:
        body: dict[str, Any] = {"model": "m"}
        assert compression.compress_body(body, _settings()) is body

    def test_empty_messages_returns_same_object(self) -> None:
        body: dict[str, Any] = {"messages": []}
        assert compression.compress_body(body, _settings()) is body

    def test_non_list_messages_returns_same_object(self) -> None:
        body: dict[str, Any] = {"messages": "not a list"}
        assert compression.compress_body(body, _settings()) is body

    def test_all_messages_recent_returns_same_object(self) -> None:
        body = {"messages": _tool_round("t1", "Bash", "z" * 10_000)}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        assert compression.compress_body(body, settings) is body


class TestTruncation:
    def test_old_long_tool_result_string_truncated(self) -> None:
        long_text = "a" * 10_000
        messages = _tool_round("t1", "Bash", long_text) + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        truncated = result["messages"][1]["content"][0]["content"]
        assert truncated.startswith("a" * 100)
        assert "truncated" in truncated
        assert len(truncated) < 200

    def test_truncation_marker_reports_removed_chars(self) -> None:
        messages = _tool_round("t1", "Bash", "b" * 150) + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        assert "50 chars" in result["messages"][1]["content"][0]["content"]

    def test_old_short_tool_result_untouched(self) -> None:
        messages = _tool_round("t1", "Bash", "short result") + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        assert result["messages"][1]["content"][0]["content"] == "short result"

    def test_recent_long_tool_result_untouched(self) -> None:
        long_text = "c" * 10_000
        messages = _padding(6) + _tool_round("t1", "Bash", long_text)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        assert result["messages"][-1]["content"][0]["content"] == long_text

    def test_list_content_budget_shared_across_text_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "d" * 80},
            {"type": "text", "text": "e" * 80},
        ]
        messages = _tool_round("t1", "Bash", blocks) + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        out_blocks = result["messages"][1]["content"][0]["content"]
        assert out_blocks[0]["text"] == "d" * 80
        assert out_blocks[1]["text"].startswith("e" * 20)
        assert "truncated" in out_blocks[1]["text"]

    def test_non_text_blocks_in_list_content_preserved(self) -> None:
        blocks = [
            {"type": "image", "source": {"type": "base64", "data": "xyz"}},
            {"type": "text", "text": "f" * 200},
        ]
        messages = _tool_round("t1", "Bash", blocks) + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        out_blocks = result["messages"][1]["content"][0]["content"]
        assert out_blocks[0] == blocks[0]

    def test_old_plain_text_messages_untouched(self) -> None:
        messages = [{"role": "user", "content": "g" * 10_000}] + _padding(6)
        body = {"messages": messages}
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        result = compression.compress_body(body, settings)
        assert result["messages"][0]["content"] == "g" * 10_000

    def test_input_body_never_mutated(self) -> None:
        messages = _tool_round("t1", "Bash", "h" * 10_000) + _padding(6)
        body = {"messages": messages, "model": "m"}
        snapshot = copy.deepcopy(body)
        compression.compress_body(
            body, _settings(keep_recent_messages=6, max_tool_result_chars=100)
        )
        assert body == snapshot

    def test_default_settings_compress_old_results(self) -> None:
        # Defaults: keep 6 recent, 4000-char budget, compression enabled.
        long_text = "i" * 10_000
        messages = _tool_round("t1", "Bash", long_text) + _padding(6)
        body = {"messages": messages}
        result = compression.compress_body(body, {})
        truncated = result["messages"][1]["content"][0]["content"]
        assert truncated.startswith("i" * 4000)
        assert len(truncated) < 5000

    @pytest.mark.parametrize(
        "content",
        [None, 42, [None, 42], [{"type": "text"}]],
        ids=["none", "int", "non-dict-blocks", "text-block-no-text"],
    )
    def test_malformed_tool_result_content_never_raises(self, content: Any) -> None:
        messages = _tool_round("t1", "Bash", content) + _padding(6)
        body = {"messages": messages}
        compression.compress_body(
            body, _settings(keep_recent_messages=6, max_tool_result_chars=100)
        )

    def test_non_dict_messages_never_raise(self) -> None:
        body: dict[str, Any] = {"messages": ["raw", None] + _padding(6)}
        compression.compress_body(
            body, _settings(keep_recent_messages=6, max_tool_result_chars=100)
        )

    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"compression": "nope"},
            {"compression": {"keep_recent_messages": "six"}},
            {"compression": {"max_tool_result_chars": None}},
        ],
        ids=["non-dict-compression", "str-keep-recent", "none-budget"],
    )
    def test_malformed_settings_fall_back_to_defaults(self, bad_settings: dict[str, Any]) -> None:
        bad_settings["compress_routed"] = True
        messages = _tool_round("t1", "Bash", "j" * 10_000) + _padding(6)
        body = {"messages": messages}
        result = compression.compress_body(body, bad_settings)
        truncated = result["messages"][1]["content"][0]["content"]
        assert len(truncated) < 5000


class TestRunRtk:
    def test_returns_filtered_stdout_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="compact\n")

        monkeypatch.setattr(compression.subprocess, "run", fake_run)
        assert compression.run_rtk("raw text", "grep") == "compact\n"

    def test_nonzero_exit_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="")

        monkeypatch.setattr(compression.subprocess, "run", fake_run)
        assert compression.run_rtk("raw text", "grep") is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="rtk", timeout=5)

        monkeypatch.setattr(compression.subprocess, "run", fake_run)
        assert compression.run_rtk("raw text", "grep") is None

    def test_missing_binary_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("rtk")

        monkeypatch.setattr(compression.subprocess, "run", fake_run)
        assert compression.run_rtk("raw text", "grep") is None

    def test_empty_stdout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="  \n")

        monkeypatch.setattr(compression.subprocess, "run", fake_run)
        assert compression.run_rtk("raw text", "grep") is None


class TestRtkIntegration:
    def _grep_body(self, text_len: int = 10_000) -> dict[str, Any]:
        """Build a body whose old tool_result came from the Grep tool."""
        messages = _tool_round("t1", "Grep", "k" * text_len) + _padding(6)
        return {"messages": messages}

    def test_never_mode_does_not_invoke_rtk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("rtk must not run when use_rtk is never")

        monkeypatch.setattr(compression, "run_rtk", boom)
        settings = _settings(keep_recent_messages=6, max_tool_result_chars=100)
        compression.compress_body(self._grep_body(), settings)

    def test_auto_mode_without_rtk_on_path_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(compression.shutil, "which", lambda _: None)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("rtk must not run when not on PATH")

        monkeypatch.setattr(compression, "run_rtk", boom)
        settings = {
            "compress_routed": True,
            "compression": {
                "use_rtk": "auto",
                "keep_recent_messages": 6,
                "max_tool_result_chars": 100,
            },
        }
        compression.compress_body(self._grep_body(), settings)

    def test_auto_mode_filters_mapped_tool_through_rtk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(compression.shutil, "which", lambda _: "/usr/local/bin/rtk")
        monkeypatch.setattr(compression, "run_rtk", lambda text, f: f"rtk[{f}]")
        settings = {
            "compress_routed": True,
            "compression": {
                "use_rtk": "auto",
                "keep_recent_messages": 6,
                "max_tool_result_chars": 100,
            },
        }
        result = compression.compress_body(self._grep_body(), settings)
        assert result["messages"][1]["content"][0]["content"] == "rtk[grep]"

    def test_unmapped_tool_not_sent_to_rtk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(compression.shutil, "which", lambda _: "/usr/local/bin/rtk")

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("unmapped tools must not reach rtk")

        monkeypatch.setattr(compression, "run_rtk", boom)
        messages = _tool_round("t1", "Bash", "m" * 10_000) + _padding(6)
        settings = {
            "compress_routed": True,
            "compression": {
                "use_rtk": "auto",
                "keep_recent_messages": 6,
                "max_tool_result_chars": 100,
            },
        }
        result = compression.compress_body({"messages": messages}, settings)
        assert "truncated" in result["messages"][1]["content"][0]["content"]

    def test_rtk_failure_falls_back_to_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(compression.shutil, "which", lambda _: "/usr/local/bin/rtk")
        monkeypatch.setattr(compression, "run_rtk", lambda text, f: None)
        settings = {
            "compress_routed": True,
            "compression": {
                "use_rtk": "auto",
                "keep_recent_messages": 6,
                "max_tool_result_chars": 100,
            },
        }
        result = compression.compress_body(self._grep_body(), settings)
        truncated = result["messages"][1]["content"][0]["content"]
        assert truncated.startswith("k" * 100)
        assert "truncated" in truncated

    def test_oversized_rtk_output_still_budget_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(compression.shutil, "which", lambda _: "/usr/local/bin/rtk")
        monkeypatch.setattr(compression, "run_rtk", lambda text, f: "n" * 500)
        settings = {
            "compress_routed": True,
            "compression": {
                "use_rtk": "auto",
                "keep_recent_messages": 6,
                "max_tool_result_chars": 100,
            },
        }
        result = compression.compress_body(self._grep_body(), settings)
        truncated = result["messages"][1]["content"][0]["content"]
        assert truncated.startswith("n" * 100)
        assert "truncated" in truncated


class TestCompressBodyFailOpen:
    """compress_body is an optimization, never a gate: any internal failure
    returns the original body object unchanged so the turn still routes."""

    def test_internal_error_returns_original_body_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a helper raises after the guards pass, the original body is
        returned (identity-equal), not a copy and not an exception
        (compression.py:245-247).
        """
        body = {"messages": [{"role": "user", "content": f"msg {i}"} for i in range(10)]}
        settings = {
            "compress_routed": True,
            "compression": {
                "keep_recent_messages": 2,
                "max_tool_result_chars": 100,
            },
        }

        def _boom(_messages: Any) -> dict[str, str]:
            raise RuntimeError("boom")

        monkeypatch.setattr(compression, "_tool_names_by_id", _boom)
        result = compression.compress_body(body, settings)
        assert result is body


class TestMixedContentToolNameResolution:
    """An assistant turn often interleaves a text block with its tool_use
    block; tool-name resolution must skip the text block and still map the
    tool_use id to its name so the matching tool_result is compressed."""

    def test_text_block_is_skipped_and_tool_use_still_resolves(self) -> None:
        """A text block in the assistant content list is skipped
        (compression.py:93) while the sibling tool_use block populates the id
        map, so the older tool_result is truncated to the char budget.
        """
        long_text = "x" * 500
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me run that"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": long_text,
                        }
                    ],
                },
                *_padding(8),
            ]
        }
        settings = {
            "compress_routed": True,
            "compression": {
                "keep_recent_messages": 2,
                "max_tool_result_chars": 100,
                "use_rtk": "never",
            },
        }
        result = compression.compress_body(body, settings)
        truncated = result["messages"][1]["content"][0]["content"]
        assert truncated.startswith("x" * 100)
        assert "truncated" in truncated
        # Immutability: the original body must be left untouched.
        assert body["messages"][1]["content"][0]["content"] == long_text
