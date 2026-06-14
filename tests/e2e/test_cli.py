"""Spec-derived tests for ai_calls_router.cli.

Contract under test: the acr CLI dispatches argparse subcommands to the
daemon, wizard, ledger, and server modules through module references (so
each layer stays independently testable), reports daemon state through exit
codes (status: 0 running / 1 stopped), launches claude with
ANTHROPIC_BASE_URL pointing at the proxy, renders the savings report, runs
the server in the foreground for serve, and surfaces errors as exit code 1
with a message on stderr instead of tracebacks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from ai_calls_router import cli
from ai_calls_router._lib import config
from ai_calls_router.ops import daemon, wizard

CONFIG_YAML = """
server:
  host: 127.0.0.1
  port: 9321
"""


@pytest.fixture
def acr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the acr home directory and config at a temp dir."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    return tmp_path


class TestParser:
    @pytest.mark.parametrize(
        "command",
        ["init", "start", "stop", "status", "code", "savings", "serve", "version"],
    )
    def test_known_commands_parse(self, command: str) -> None:
        args = cli.build_parser().parse_args([command])
        assert args.command == command

    def test_no_command_exits_with_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args([])
        assert excinfo.value.code == 2

    def test_unknown_command_exits_with_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.build_parser().parse_args(["explode"])
        assert excinfo.value.code == 2

    def test_code_collects_claude_arguments(self) -> None:
        args = cli.build_parser().parse_args(["code", "-p", "hello world"])
        assert args.claude_args == ["-p", "hello world"]


class TestStatus:
    def test_running_daemon_reports_pid_and_url(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(daemon, "status", lambda: 4242)
        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert "4242" in out
        assert "http://127.0.0.1:9321" in out

    def test_stopped_daemon_exits_nonzero(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(daemon, "status", lambda: None)
        assert cli.main(["status"]) == 1
        assert "not running" in capsys.readouterr().out


class TestStartStop:
    def test_start_reports_listen_url(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(daemon, "start", lambda: 4242)
        assert cli.main(["start"]) == 0
        assert "http://127.0.0.1:9321" in capsys.readouterr().out

    def test_start_failure_reports_error_without_traceback(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        def _fail() -> int:
            raise daemon.DaemonError("did not become healthy")

        monkeypatch.setattr(daemon, "start", _fail)
        assert cli.main(["start"]) == 1
        assert "did not become healthy" in capsys.readouterr().err

    def test_stop_running_daemon(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(daemon, "stop", lambda: True)
        assert cli.main(["stop"]) == 0
        assert "stopped" in capsys.readouterr().out

    def test_stop_when_not_running_is_not_an_error(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        monkeypatch.setattr(daemon, "stop", lambda: False)
        assert cli.main(["stop"]) == 0
        assert "not running" in capsys.readouterr().out


class TestInit:
    def test_init_runs_wizard(self, acr_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []

        def _wizard(**kwargs: Any) -> Path:
            calls.append(True)
            return config.config_path()

        monkeypatch.setattr(wizard, "run_wizard", _wizard)
        assert cli.main(["init"]) == 0
        assert calls == [True]


class TestCode:
    def test_code_launches_claude_with_proxy_env(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(daemon, "start", lambda: 4242)
        runs: list[tuple[list[str], dict[str, str]]] = []

        def _run(cmd: list[str], env: dict[str, str], **kwargs: Any) -> Any:
            runs.append((cmd, env))
            return subprocess.CompletedProcess(cmd, 7)

        monkeypatch.setattr(subprocess, "run", _run)
        assert cli.main(["code", "-p", "hi"]) == 7
        cmd, env = runs[0]
        assert cmd == ["claude", "-p", "hi"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9321"
        assert env.get("PATH") == os.environ.get("PATH")

    def test_code_aborts_when_daemon_cannot_start(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        def _fail() -> int:
            raise daemon.DaemonError("boom")

        monkeypatch.setattr(daemon, "start", _fail)
        runs: list[Any] = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: runs.append(a))
        assert cli.main(["code"]) == 1
        assert runs == []
        assert "boom" in capsys.readouterr().err


class TestSavings:
    def test_empty_ledger_reports_no_calls(self, acr_home: Path, capsys: Any) -> None:
        assert cli.main(["savings"]) == 0
        assert "No routed calls" in capsys.readouterr().out

    def test_ledger_entries_aggregated(self, acr_home: Path, capsys: Any) -> None:
        entry = {
            "routed_model": "deepseek/test",
            "input_tokens": 100,
            "output_tokens": 50,
            "routed_usd": 0.01,
            "premium_usd": 0.50,
            "saved_usd": 0.49,
        }
        ledger_file = Path(os.environ["ACR_SAVINGS_LEDGER"])
        ledger_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        assert cli.main(["savings"]) == 0
        out = capsys.readouterr().out
        assert "Routing savings" in out
        assert "deepseek/test" in out


class TestServe:
    def test_serve_runs_uvicorn_on_configured_address(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs: list[dict[str, Any]] = []

        def _run(app: Any, **kwargs: Any) -> None:
            runs.append({"app": app, **kwargs})

        monkeypatch.setattr(uvicorn, "run", _run)
        assert cli.main(["serve"]) == 0
        assert runs[0]["host"] == "127.0.0.1"
        assert runs[0]["port"] == 9321

    def test_serve_rejects_unsupported_premium_provider(
        self, *, acr_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        config_file = Path(os.environ["ACR_CONFIG"])
        config_file.write_text(CONFIG_YAML + "premium:\n  provider: openai\n", encoding="utf-8")
        runs: list[Any] = []
        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: runs.append(a))
        assert cli.main(["serve"]) == 1
        assert runs == []
        assert "openai" in capsys.readouterr().err


class TestVersion:
    def test_version_prints_package_version(self, capsys: Any) -> None:
        assert cli.main(["version"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("acr ")
        assert out.strip() != "acr"
