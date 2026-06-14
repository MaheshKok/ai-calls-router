"""Enforce keyword-only signatures for wide call surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "tests")
RECEIVER_NAMES = {"self", "cls"}


def _python_files() -> list[Path]:
    return sorted(path for root in SCAN_ROOTS for path in root.rglob("*.py"))


def _qualname(stack: list[str], name: str) -> str:
    return ".".join([*stack, name])


def test_functions_with_more_than_two_user_parameters_are_keyword_only() -> None:
    """Wide signatures must force named arguments after any method receiver."""
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._check(node)

        def _check(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            positional = [*node.args.posonlyargs, *node.args.args]
            has_receiver = bool(self.stack and positional and positional[0].arg in RECEIVER_NAMES)
            user_positionals = positional[1:] if has_receiver else positional
            if len(user_positionals) > 2:
                relative = self.path.relative_to(PROJECT_ROOT)
                violations.append(
                    f"{relative}:{node.lineno}: {_qualname(self.stack, node.name)} "
                    f"allows positional parameters {[arg.arg for arg in user_positionals]!r}"
                )
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    for path in _python_files():
        Visitor(path).visit(ast.parse(path.read_text(), filename=str(path)))

    assert violations == []
