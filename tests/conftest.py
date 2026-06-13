"""Shared pytest configuration for the ai-calls-router suite.

Registers the unit/integration/e2e markers, auto-applies the marker that
matches each test's subdirectory so the suite can be sliced by layer without
hand-decorating every test, and exposes the fixtures that the layered suites
share: the fixtures directory and a fresh mock premium upstream per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the shared harness importable from any test subdirectory under
# --import-mode=importlib, independent of plugin load ordering.
sys.path.insert(0, str(Path(__file__).parent))

from acr_testkit import Upstream  # noqa: E402

_LAYER_MARKERS = ("unit", "integration", "e2e")


def pytest_configure(config: pytest.Config) -> None:
    """Register the layer markers so strict-marker runs do not warn.

    Args:
        config: The active pytest configuration object.
    """
    for marker in _LAYER_MARKERS:
        config.addinivalue_line("markers", f"{marker}: {marker}-level test.")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag each test with the marker matching its tests/<layer>/ directory.

    Args:
        config: The active pytest configuration object.
        items: Collected test items, mutated in place to add markers.
    """
    tests_root = Path(__file__).parent
    for item in items:
        try:
            relative = Path(str(item.fspath)).relative_to(tests_root)
        except ValueError:
            continue
        layer = relative.parts[0] if len(relative.parts) > 1 else ""
        if layer in _LAYER_MARKERS:
            item.add_marker(getattr(pytest.mark, layer))


@pytest.fixture()
def fixtures_dir() -> Path:
    """Return the absolute path to the shared tests/fixtures directory.

    Returns:
        Path to tests/fixtures, resolved independently of the cwd.
    """
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def upstream() -> Upstream:
    """Provide a fresh mock premium upstream for app-level tests.

    Returns:
        An Upstream recorder with an empty request log.
    """
    return Upstream()
