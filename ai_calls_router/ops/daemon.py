"""Daemon lifecycle management for the ai-calls-router proxy.

Spawns the proxy as a detached "python -m ai_calls_router serve" process,
tracks it through a pidfile under the acr home directory, confirms startup
by polling /health on the configured port, and stops it with SIGTERM
escalating to SIGKILL when the process ignores the polite request.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

import httpx

from ai_calls_router._lib import config
from ai_calls_router.routing import decide as routing

logger = logging.getLogger("acr.daemon")

STOP_TIMEOUT_SECONDS = 5.0
HEALTH_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.1
HEALTH_REQUEST_TIMEOUT_SECONDS = 1.0


class DaemonError(Exception):
    """Raised when the proxy daemon cannot be started."""


def read_pid() -> int | None:
    """Read the daemon pid from the pidfile.

    Returns:
        The recorded pid, or None when the pidfile is missing or corrupt.
    """
    try:
        return int(config.pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def is_running(pid: int) -> bool:
    """Check whether a process with the given pid is alive.

    Uses signal 0, which performs the existence check without delivering
    a signal. A PermissionError means the process exists but belongs to
    another user, so it counts as running.

    Args:
        pid: Process id to probe.

    Returns:
        True when the process exists, False otherwise.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status() -> int | None:
    """Report the running daemon's pid, cleaning up stale pidfiles.

    Returns:
        The live daemon pid, or None when no daemon is running. A pidfile
        pointing at a dead process is removed as a side effect.
    """
    pid = read_pid()
    if pid is None:
        return None
    if is_running(pid):
        return pid
    config.pid_path().unlink(missing_ok=True)
    return None


def _health_url() -> str:
    """Build the daemon health endpoint URL from the active config.

    Returns:
        The /health URL on the configured host and port.
    """
    settings = config.server_settings(routing.load_routes())
    return f"http://{settings.host}:{settings.port}/health"


def _wait_healthy(url: str) -> bool:
    """Poll a health URL until it answers 200 or the timeout elapses.

    Args:
        url: Health endpoint to poll.

    Returns:
        True when the endpoint answered 200 within HEALTH_TIMEOUT_SECONDS,
        False otherwise.
    """
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while True:
        try:
            response = httpx.get(url, timeout=HEALTH_REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_SECONDS)


def start() -> int:
    """Start the proxy daemon if it is not already running.

    Spawns a detached "python -m ai_calls_router serve" process with output
    appended to the daemon log, records its pid, and waits for /health to
    answer before returning. Idempotent: a running daemon is left alone.

    Returns:
        The pid of the (possibly pre-existing) daemon process.

    Raises:
        DaemonError: When the spawned process never becomes healthy; the
            child is terminated and the pidfile removed before raising.
    """
    existing = status()
    if existing is not None:
        return existing

    config.home_dir().mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "ai_calls_router", "serve"]
    with config.log_path().open("ab") as log_handle:
        child = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    config.pid_path().write_text(str(child.pid), encoding="utf-8")

    if not _wait_healthy(_health_url()):
        child.terminate()
        config.pid_path().unlink(missing_ok=True)
        raise DaemonError(f"acr daemon did not become healthy; see {config.log_path()}")
    return child.pid


def stop() -> bool:
    """Stop the running proxy daemon.

    Sends SIGTERM and waits up to STOP_TIMEOUT_SECONDS for the process to
    exit, escalating to SIGKILL if it is still alive. The pidfile is always
    removed, including for stale entries pointing at dead processes.

    Returns:
        True when a live daemon was stopped, False when none was running.
    """
    pid = read_pid()
    if pid is None:
        return False
    if not is_running(pid):
        config.pid_path().unlink(missing_ok=True)
        return False

    _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while is_running(pid) and time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
    if is_running(pid):
        _signal(pid, signal.SIGKILL)

    config.pid_path().unlink(missing_ok=True)
    return True


def _signal(pid: int, sig: signal.Signals) -> None:
    """Send a signal, tolerating a process that already exited.

    Args:
        pid: Target process id.
        sig: Signal to deliver.
    """
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
