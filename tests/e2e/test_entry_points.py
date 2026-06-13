"""End-to-end coverage for the ``python -m ai_calls_router`` entry point.

The daemon spawns the proxy as a detached ``python -m ai_calls_router serve``
process, so the module entry point must delegate to the acr CLI and propagate
its exit code unchanged. These tests drive the real module both through a
subprocess (true process boundary) and in-process via runpy, exercising the
``if __name__ == "__main__"`` guard.
"""

from __future__ import annotations

import runpy
import subprocess
import sys

import pytest

from ai_calls_router import __version__


class TestModuleEntryPoint:
    """``python -m ai_calls_router`` dispatches to the CLI and exits with its code."""

    def test_run_module_as_main_dispatches_to_cli_and_propagates_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Running the module with ``run_name="__main__"`` executes the guard
        in-process: it calls ``cli.main()`` and exits with its return code.

        This is the in-process counterpart to the subprocess tests below; it
        drives the ``if __name__ == "__main__": sys.exit(main())`` block so the
        guard's wiring is exercised directly.
        """
        monkeypatch.setattr(sys, "argv", ["ai_calls_router", "version"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("ai_calls_router.__main__", run_name="__main__")
        assert exc_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_version_subcommand_prints_version_and_exits_zero(self) -> None:
        """``python -m ai_calls_router version`` prints the version and exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_calls_router", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_no_subcommand_exits_with_usage_error(self) -> None:
        """Invoking the module with no subcommand exits non-zero (argparse usage)."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_calls_router"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
