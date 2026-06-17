"""Configuration paths, defaults and validation for ai-calls-router.

Centralizes every filesystem location the proxy touches (config file, pid
file, daemon log, savings ledger) under ~/.ai-calls-router with env-var
overrides, and provides fail-open accessors for the server: and premium:
blocks of config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8747
DEFAULT_UPSTREAM = "https://api.anthropic.com"


class ConfigError(Exception):
    """Raised when config.yaml contains a configuration v1 cannot serve."""


@dataclass(frozen=True)
class ServerSettings:
    """Resolved server: block of config.yaml.

    Attributes:
        host: Bind address for the proxy daemon.
        port: Listen port for the proxy daemon.
        upstream: Premium passthrough target (no trailing slash).
    """

    host: str
    port: int
    upstream: str


def home_dir() -> Path:
    """Return the ai-calls-router state directory.

    Returns:
        $ACR_HOME when set, otherwise ~/.ai-calls-router.
    """
    override = os.environ.get("ACR_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ai-calls-router"


def config_path() -> Path:
    """Return the config.yaml location ($ACR_CONFIG overrides)."""
    override = os.environ.get("ACR_CONFIG")
    if override:
        return Path(override).expanduser()
    return home_dir() / "config.yaml"


def provider_config_dir() -> Path:
    """Return the directory holding per-provider YAML files."""
    return home_dir() / "config"


def provider_config_path(group: str) -> Path:
    """Return the per-provider YAML path for an agent group.

    Args:
        group: Agent group name, for example ``claude_code``.

    Returns:
        Path to the group's YAML file under ``provider_config_dir()``.
    """
    return provider_config_dir() / f"{group.replace('_', '-')}.yaml"


def pid_path() -> Path:
    """Return the daemon pidfile location."""
    return home_dir() / "acr.pid"


def log_path() -> Path:
    """Return the structured application log file location.

    Owned by the logging RotatingFileHandler configured in
    ``logging_setup.setup_logging``; carries the rich per-request log.
    """
    return home_dir() / "acr.log"


def daemon_log_path() -> Path:
    """Return the raw daemon stdout/stderr capture location.

    Distinct from ``log_path``: this file receives the daemon subprocess's
    unstructured stdout/stderr (uvicorn banner, early-startup crashes, stray
    prints) so those bytes never interleave with the structured ``acr.log``.
    """
    return home_dir() / "acr.daemon.out"


def ledger_path() -> Path:
    """Return the savings ledger location ($ACR_SAVINGS_LEDGER overrides)."""
    override = os.environ.get("ACR_SAVINGS_LEDGER")
    if override:
        return Path(override).expanduser()
    return home_dir() / "savings.jsonl"


def server_settings(routes: dict[str, Any]) -> ServerSettings:
    """Resolve the server: block, failing open to safe defaults.

    Args:
        routes: Parsed config.yaml mapping.

    Returns:
        ServerSettings with defaults substituted for missing or malformed
        values (a bad config must not stop the proxy from serving).
    """
    server = routes.get("server")
    if not isinstance(server, dict):
        server = {}

    host = server.get("host")
    if not isinstance(host, str) or not host:
        host = DEFAULT_HOST

    port = server.get("port")
    if not isinstance(port, int) or isinstance(port, bool):
        port = DEFAULT_PORT

    upstream = server.get("upstream")
    if not isinstance(upstream, str) or not upstream:
        upstream = DEFAULT_UPSTREAM

    return ServerSettings(host=host, port=port, upstream=upstream.rstrip("/"))


def validate_premium(routes: dict[str, Any]) -> None:
    """Validate the premium: block against v1 capabilities.

    v1 only supports the Anthropic passthrough; the block exists so future
    versions can add user-selectable premium providers without a schema break.

    Args:
        routes: Parsed config.yaml mapping.

    Raises:
        ConfigError: When the premium block is malformed or names a provider
            other than anthropic.
    """
    premium = routes.get("premium")
    if premium is None:
        return
    if not isinstance(premium, dict):
        raise ConfigError("premium: must be a mapping (e.g. {provider: anthropic})")
    provider = premium.get("provider", "anthropic")
    if provider != "anthropic":
        raise ConfigError(
            f"premium provider {provider!r} is not supported in v1; only 'anthropic' is"
        )
