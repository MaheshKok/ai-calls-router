"""Spec-derived tests for ai_calls_router.daemon.

Contract under test: the daemon lifecycle helpers manage a detached proxy
process through a pidfile under the acr home directory -- read_pid tolerates
missing or corrupt pidfiles, is_running probes with signal 0, status cleans
up stale pidfiles, start is idempotent and spawns a detached "python -m
ai_calls_router serve" process that must answer /health before start
returns, and stop terminates with SIGTERM escalating to SIGKILL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_calls_router import config, daemon

CONFIG_YAML = """
server:
  host: 127.0.0.1
  port: 9321
"""


@pytest.fixture()
def acr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the acr home directory and config at a temp dir."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    return tmp_path


class _FakePopen:
    """subprocess.Popen stand-in recording the spawn call."""

    def __init__(self, cmd: list[str], **kwargs: Any) -> None:
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 54321
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


class _KillRecorder:
    """os.kill stand-in tracking signals and simulated process liveness."""

    def __init__(self, alive: bool = True, dies_on: int | None = 15) -> None:
        self.alive = alive
        self.dies_on = dies_on
        self.signals: list[tuple[int, int]] = []

    def __call__(self, pid: int, sig: int) -> None:
        if sig == 0:
            if not self.alive:
                raise ProcessLookupError(pid)
            return
        self.signals.append((pid, sig))
        if sig == self.dies_on:
            self.alive = False


class TestReadPid:
    def test_missing_pidfile_returns_none(self, acr_home: Path) -> None:
        assert daemon.read_pid() is None

    def test_corrupt_pidfile_returns_none(self, acr_home: Path) -> None:
        config.pid_path().write_text("not-a-pid\n", encoding="utf-8")
        assert daemon.read_pid() is None

    def test_valid_pidfile_returns_int(self, acr_home: Path) -> None:
        config.pid_path().write_text("12345\n", encoding="utf-8")
        assert daemon.read_pid() == 12345


class TestIsRunning:
    def test_live_process_is_running(self) -> None:
        assert daemon.is_running(os.getpid()) is True

    def test_dead_process_is_not_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "kill", _KillRecorder(alive=False))
        assert daemon.is_running(99999) is False

    def test_permission_error_counts_as_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _denied(pid: int, sig: int) -> None:
            raise PermissionError(pid)

        monkeypatch.setattr(os, "kill", _denied)
        assert daemon.is_running(1) is True


class TestStatus:
    def test_no_pidfile_means_not_running(self, acr_home: Path) -> None:
        assert daemon.status() is None

    def test_live_pid_reported(self, acr_home: Path) -> None:
        config.pid_path().write_text(str(os.getpid()), encoding="utf-8")
        assert daemon.status() == os.getpid()

    def test_stale_pidfile_cleaned_up(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config.pid_path().write_text("99999", encoding="utf-8")
        monkeypatch.setattr(os, "kill", _KillRecorder(alive=False))
        assert daemon.status() is None
        assert not config.pid_path().exists()


class TestStart:
    def test_start_is_idempotent_when_running(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config.pid_path().write_text(str(os.getpid()), encoding="utf-8")
        spawned: list[_FakePopen] = []

        def _spawn(cmd: list[str], **kwargs: Any) -> _FakePopen:
            popen = _FakePopen(cmd, **kwargs)
            spawned.append(popen)
            return popen

        monkeypatch.setattr(subprocess, "Popen", _spawn)
        assert daemon.start() == os.getpid()
        assert spawned == []

    def test_start_spawns_detached_serve_process(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[_FakePopen] = []

        def _spawn(cmd: list[str], **kwargs: Any) -> _FakePopen:
            popen = _FakePopen(cmd, **kwargs)
            spawned.append(popen)
            return popen

        monkeypatch.setattr(subprocess, "Popen", _spawn)
        monkeypatch.setattr(daemon, "_wait_healthy", lambda url: True)

        pid = daemon.start()

        assert pid == 54321
        popen = spawned[0]
        assert popen.cmd[0] == sys.executable
        assert popen.cmd[1:4] == ["-m", "ai_calls_router", "serve"]
        assert popen.kwargs["start_new_session"] is True
        assert popen.kwargs["stdout"].name == str(config.log_path())
        assert popen.kwargs["stderr"] == subprocess.STDOUT
        assert daemon.read_pid() == 54321

    def test_start_polls_health_on_configured_port(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        polled: list[str] = []

        def _healthy(url: str) -> bool:
            polled.append(url)
            return True

        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(daemon, "_wait_healthy", _healthy)
        daemon.start()
        assert polled == ["http://127.0.0.1:9321/health"]

    def test_start_failure_terminates_child_and_cleans_pidfile(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[_FakePopen] = []

        def _spawn(cmd: list[str], **kwargs: Any) -> _FakePopen:
            popen = _FakePopen(cmd, **kwargs)
            spawned.append(popen)
            return popen

        monkeypatch.setattr(subprocess, "Popen", _spawn)
        monkeypatch.setattr(daemon, "_wait_healthy", lambda url: False)

        with pytest.raises(daemon.DaemonError):
            daemon.start()
        assert spawned[0].terminated is True
        assert not config.pid_path().exists()


class TestStop:
    def test_stop_when_not_running_returns_false(self, acr_home: Path) -> None:
        assert daemon.stop() is False

    def test_stop_cleans_stale_pidfile(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config.pid_path().write_text("99999", encoding="utf-8")
        monkeypatch.setattr(os, "kill", _KillRecorder(alive=False))
        assert daemon.stop() is False
        assert not config.pid_path().exists()

    def test_stop_sends_sigterm_and_removes_pidfile(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config.pid_path().write_text("4242", encoding="utf-8")
        recorder = _KillRecorder(alive=True, dies_on=15)
        monkeypatch.setattr(os, "kill", recorder)

        assert daemon.stop() is True
        assert recorder.signals == [(4242, 15)]
        assert not config.pid_path().exists()

    def test_stop_escalates_to_sigkill_when_sigterm_ignored(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config.pid_path().write_text("4242", encoding="utf-8")
        recorder = _KillRecorder(alive=True, dies_on=9)
        monkeypatch.setattr(os, "kill", recorder)
        monkeypatch.setattr(daemon, "STOP_TIMEOUT_SECONDS", 0.05)

        assert daemon.stop() is True
        assert (4242, 15) in recorder.signals
        assert (4242, 9) in recorder.signals
        assert not config.pid_path().exists()

    def test_stop_tolerates_process_exiting_before_signal(
        self, acr_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process that vanishes between the liveness probe and the signal
        must not crash stop(): _signal swallows ProcessLookupError so the
        pidfile is still cleaned up (daemon.py:192-193).

        The kill stand-in reports the process as alive for signal 0 (the
        liveness probe) but raises ProcessLookupError for any real signal,
        modelling the race where the daemon dies just before SIGTERM lands.
        """
        config.pid_path().write_text("4242", encoding="utf-8")

        def _kill(pid: int, sig: int) -> None:
            if sig == 0:
                return
            raise ProcessLookupError(pid)

        monkeypatch.setattr(os, "kill", _kill)
        monkeypatch.setattr(daemon, "STOP_TIMEOUT_SECONDS", 0.05)

        assert daemon.stop() is True
        assert not config.pid_path().exists()


class TestWaitHealthy:
    def test_returns_true_when_health_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _ok(url: str, timeout: float) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"})

        monkeypatch.setattr(httpx, "get", _ok)
        assert daemon._wait_healthy("http://127.0.0.1:9321/health") is True

    def test_returns_false_when_never_reachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refused(url: str, timeout: float) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", _refused)
        monkeypatch.setattr(daemon, "HEALTH_TIMEOUT_SECONDS", 0.05)
        assert daemon._wait_healthy("http://127.0.0.1:9321/health") is False

    def test_non_200_keeps_polling_until_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _error(url: str, timeout: float) -> httpx.Response:
            return httpx.Response(503)

        monkeypatch.setattr(httpx, "get", _error)
        monkeypatch.setattr(daemon, "HEALTH_TIMEOUT_SECONDS", 0.05)
        assert daemon._wait_healthy("http://127.0.0.1:9321/health") is False
