#!/usr/bin/env python3
"""Check cognitive complexity of all functions in the codebase.

Threshold: 15 per function (matches PyCharm IDE flag level).
Reports violations matching the clang/gcc warning format for CI integration.
"""

import ast
import sys
from pathlib import Path

from cognitive_complexity.api import get_cognitive_complexity

THRESHOLD = 15
THRESHOLD_WARN = 20  # Warning level (non-blocking)


def check_file(filepath: Path) -> list[tuple[int, str, int]]:
    """Return list of (lineno, func_name, complexity) violations."""
    violations = []
    try:
        with Path.open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            score = get_cognitive_complexity(node)
            if score > THRESHOLD:
                violations.append((node.lineno, node.name, score))
    return violations


def main() -> int:
    root = Path("ai_calls_router")
    all_violations = []

    for pyfile in sorted(root.rglob("*.py")):
        violations = check_file(pyfile)
        for lineno, name, score in violations:
            all_violations.append((pyfile, lineno, name, score))
            level = "warning" if score <= THRESHOLD_WARN else "error"
            print(
                f"{pyfile}:{lineno}: {level}: {name} has cognitive complexity {score} (threshold {THRESHOLD})"  # noqa: E501
            )

    if all_violations:
        print(f"\nFound {len(all_violations)} violations (max allowed: {THRESHOLD})")
        return 1
    print("All functions within cognitive complexity threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
