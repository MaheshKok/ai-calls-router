"""Cross-wire routing parity tests.

The router accepts Anthropic Messages and OpenAI Chat Completions wires, but
should make the same tier decision for the same logical tool-result turn.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_calls_router.routing import decide
from ai_calls_router.routing.adapters import adapter_for_path

ROUTES: dict[str, object] = {
    "settings": {"tier_precedence": ["premium", "fast"]},
    "tiers": {"fast": {"model": "deepseek/acr-test-cheap"}},
    "agents": {
        "claude_code": {"tools": {"exec_command": "fast", "apply_patch": "premium"}},
        "hermes": {"tools": {"exec_command": "fast", "apply_patch": "premium"}},
    },
}


def _anthropic_body(tool_name: str) -> dict[str, object]:
    return {
        "model": "claude-test",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": tool_name, "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}],
            },
        ],
    }


def _chat_body(tool_name: str) -> dict[str, object]:
    return {
        "model": "gpt-test",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    }


@pytest.mark.parametrize(
    ("tool_name", "expected_tier"),
    [("exec_command", "fast"), ("apply_patch", "premium")],
)
def test_tool_detection_parity_across_wires(tool_name: str, expected_tier: str) -> None:
    cases = [
        ("/v1/messages", "claude_code", _anthropic_body(tool_name)),
        ("/v1/chat/completions", "hermes", _chat_body(tool_name)),
    ]

    decisions: list[tuple[list[str], str]] = []
    for path, group, body in cases:
        adapter = adapter_for_path(path)
        assert adapter is not None
        names = adapter.extract_pending_tools(body)
        decisions.append((names, decide.tier_for_tools(names, ROUTES, group=group)))

    assert decisions == [([tool_name], expected_tier)] * 2


def test_direct_module_contains_no_decision_logic() -> None:
    tree = ast.parse(Path("ai_calls_router/routing/direct.py").read_text(encoding="utf-8"))
    imports_decide = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ai_calls_router.routing"
        and any(alias.name == "decide" for alias in node.names)
        for node in ast.walk(tree)
    )
    suspicious_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and ("escalat" in node.name or "select" in node.name or "resolve_tier" in node.name)
    }

    assert not imports_decide
    assert suspicious_functions == set()
