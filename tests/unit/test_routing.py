"""Spec-derived tests for ai_calls_router.routing.

Contract under test (ported from the proven headroom-tool-router engine):
load_routes fail-opens to {} and hot-reloads on mtime change; pending tool
names come only from trailing tool_result blocks in the last user message;
every ambiguity (unknown tool, unresolvable id, missing tier) resolves to
premium; API keys come from the process env then env_file, never elsewhere.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from ai_calls_router.routing import decide as routing


def _body_with_tool_results(*pairs: tuple[str, str]) -> dict[str, Any]:
    """Build a request body whose last user message carries tool results.

    Args:
        pairs: (tool_use_id, tool_name) tuples; the assistant message gets a
            tool_use block per pair and the user message a tool_result block.

    Returns:
        A minimal Anthropic Messages request body.
    """
    assistant_blocks = [
        {"type": "tool_use", "id": tid, "name": name, "input": {}} for tid, name in pairs
    ]
    user_blocks = [{"type": "tool_result", "tool_use_id": tid} for tid, _ in pairs]
    return {
        "messages": [
            {"role": "assistant", "content": assistant_blocks},
            {"role": "user", "content": user_blocks},
        ]
    }


class TestLoadRoutes:
    def test_parses_valid_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("tools:\n  Bash: fast\n", encoding="utf-8")
        assert routing.load_routes(cfg) == {"tools": {"Bash": "fast"}}

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert routing.load_routes(tmp_path / "nope.yaml") == {}

    def test_non_mapping_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- a\n- b\n", encoding="utf-8")
        assert routing.load_routes(cfg) == {}

    def test_invalid_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("tools: [unclosed\n", encoding="utf-8")
        assert routing.load_routes(cfg) == {}

    def test_hot_reload_on_mtime_change(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("tools:\n  Bash: fast\n", encoding="utf-8")
        assert routing.load_routes(cfg)["tools"]["Bash"] == "fast"
        cfg.write_text("tools:\n  Bash: code\n", encoding="utf-8")
        # Force a distinct mtime even on coarse-grained filesystems.
        future = time.time() + 5
        os.utime(cfg, (future, future))
        assert routing.load_routes(cfg)["tools"]["Bash"] == "code"

    def test_two_paths_do_not_share_cache(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("tools:\n  Bash: fast\n", encoding="utf-8")
        b.write_text("tools:\n  Bash: crud\n", encoding="utf-8")
        # Identical mtimes must not cross-contaminate the cache.
        now = time.time()
        os.utime(a, (now, now))
        os.utime(b, (now, now))
        assert routing.load_routes(a)["tools"]["Bash"] == "fast"
        assert routing.load_routes(b)["tools"]["Bash"] == "crud"
        assert routing.load_routes(a)["tools"]["Bash"] == "fast"


class TestPendingToolNames:
    def test_turn_opener_returns_empty(self) -> None:
        body = {"messages": [{"role": "user", "content": "hello"}]}
        assert routing.pending_tool_names(body) == []

    def test_no_messages_returns_empty(self) -> None:
        assert routing.pending_tool_names({}) == []
        assert routing.pending_tool_names({"messages": []}) == []
        assert routing.pending_tool_names({"messages": "bad"}) == []

    def test_last_message_not_user_returns_empty(self) -> None:
        body = {"messages": [{"role": "assistant", "content": []}]}
        assert routing.pending_tool_names(body) == []

    def test_resolves_single_tool_name(self) -> None:
        body = _body_with_tool_results(("t1", "Bash"))
        assert routing.pending_tool_names(body) == ["Bash"]

    def test_deduplicates_preserving_order(self) -> None:
        body = _body_with_tool_results(("t1", "Read"), ("t2", "Bash"), ("t3", "Read"))
        assert routing.pending_tool_names(body) == ["Read", "Bash"]

    def test_unresolvable_id_returns_unknown_sentinel(self) -> None:
        body = {
            "messages": [
                {"role": "assistant", "content": []},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "ghost"}],
                },
            ]
        }
        assert routing.pending_tool_names(body) == ["<unknown>"]

    def test_partial_resolution_still_returns_unknown(self) -> None:
        body = _body_with_tool_results(("t1", "Bash"))
        body["messages"][-1]["content"].append({"type": "tool_result", "tool_use_id": "ghost"})
        assert routing.pending_tool_names(body) == ["<unknown>"]

    def test_tool_result_without_id_is_ignored(self) -> None:
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "tool_result"}]},
            ]
        }
        assert routing.pending_tool_names(body) == []


class TestLookupTool:
    def test_exact_match(self) -> None:
        assert routing.lookup_tool("Bash", {"Bash": "fast"}) == "fast"

    def test_trailing_glob_match(self) -> None:
        assert routing.lookup_tool("mcp__github__get_issue", {"mcp__*": "fast"}) == "fast"

    def test_exact_match_wins_over_glob(self) -> None:
        tools = {"mcp__*": "fast", "mcp__special": "code"}
        assert routing.lookup_tool("mcp__special", tools) == "code"

    def test_unmapped_returns_none(self) -> None:
        assert routing.lookup_tool("Edit", {"Bash": "fast"}) is None

    def test_non_trailing_wildcard_pattern_is_exact_only(self) -> None:
        assert routing.lookup_tool("aXb", {"a*b": "fast"}) is None


class TestTierForTools:
    ROUTES: dict[str, Any] = {
        "settings": {"tier_precedence": ["premium", "structured", "code", "fast", "crud"]},
        "tools": {"Bash": "fast", "Read": "code", "TodoWrite": "crud", "Edit": "premium"},
    }

    def test_empty_batch_is_premium(self) -> None:
        assert routing.tier_for_tools([], self.ROUTES) == "premium"

    def test_unknown_tool_is_premium(self) -> None:
        assert routing.tier_for_tools(["Mystery"], self.ROUTES) == "premium"

    def test_explicit_premium_mapping_is_premium(self) -> None:
        assert routing.tier_for_tools(["Edit"], self.ROUTES) == "premium"

    def test_single_mapped_tool_returns_its_tier(self) -> None:
        assert routing.tier_for_tools(["Bash"], self.ROUTES) == "fast"

    def test_mixed_batch_resolves_by_precedence(self) -> None:
        assert routing.tier_for_tools(["Bash", "Read"], self.ROUTES) == "code"
        assert routing.tier_for_tools(["TodoWrite", "Bash"], self.ROUTES) == "fast"

    def test_premium_in_batch_overrides_everything(self) -> None:
        assert routing.tier_for_tools(["Bash", "Edit"], self.ROUTES) == "premium"

    def test_tier_absent_from_precedence_is_premium(self) -> None:
        routes = {
            "settings": {"tier_precedence": ["premium", "fast"]},
            "tools": {"X": "exotic"},
        }
        assert routing.tier_for_tools(["X"], routes) == "premium"

    def test_default_precedence_when_settings_missing(self) -> None:
        routes = {"tools": {"Bash": "fast"}}
        assert routing.tier_for_tools(["Bash"], routes) == "fast"


class TestResolveApiKey:
    def test_process_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ACR_TEST_KEY", "from-env")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {})
        assert key == "from-env"

    def test_env_file_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nACR_TEST_KEY=from-file\n", encoding="utf-8")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {"env_file": str(env_file)})
        assert key == "from-file"

    def test_env_file_export_prefix_and_quotes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text('export ACR_TEST_KEY="quoted-value"\n', encoding="utf-8")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {"env_file": str(env_file)})
        assert key == "quoted-value"

    def test_missing_key_env_returns_none(self) -> None:
        assert routing.resolve_api_key({}, {}) is None

    def test_unavailable_everywhere_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=x\n", encoding="utf-8")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {"env_file": str(env_file)})
        assert key is None

    def test_missing_env_file_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        key = routing.resolve_api_key(
            {"key_env": "ACR_TEST_KEY"}, {"env_file": "/nonexistent/.env"}
        )
        assert key is None

    def test_env_file_tilde_is_expanded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".env").write_text("ACR_TEST_KEY=home-file\n", encoding="utf-8")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {"env_file": "~/.env"})
        assert key == "home-file"

    def test_empty_value_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ACR_TEST_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("ACR_TEST_KEY=\n", encoding="utf-8")
        key = routing.resolve_api_key({"key_env": "ACR_TEST_KEY"}, {"env_file": str(env_file)})
        assert key is None
