"""Agent launcher helpers for routing local CLIs through acr.

Claude can be routed with a process environment override. Hermes needs a
persistent local config override because some flows ignore `OPENAI_BASE_URL`.
`unwrap` restores the byte-for-byte backup when the user wants to return Hermes
to its original upstream.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue


class WrapError(RuntimeError):
    """Raised when persistent wrapper setup cannot be applied."""


AGENT_COMMANDS: dict[str, str] = {
    "claude": "claude",
    "hermes": "hermes",
}

_HERMES_START_MARKER = "# >>> ai-calls-router hermes wrap >>>"
_HERMES_BACKUP_SUFFIX = ".acr-backup"


def hermes_config_path() -> Path:
    """Return the Hermes CLI config path.

    Returns:
        The user's `~/.hermes/config.yaml` path.
    """
    return Path.home() / ".hermes" / "config.yaml"


def hermes_backup_path(config_file: Path | None = None) -> Path:
    """Return the byte-for-byte backup path for a Hermes config file.

    Args:
        config_file: Optional Hermes config path override.

    Returns:
        The backup path used by `unwrap hermes`.
    """
    target = hermes_config_path() if config_file is None else config_file
    return target.with_name(f"{target.name}{_HERMES_BACKUP_SUFFIX}")


def enable_hermes_config(proxy_url: str, config_file: Path | None = None) -> Path:
    """Point Hermes OpenAI-compatible traffic at acr until `unwrap hermes`.

    Args:
        proxy_url: Base acr listen URL without `/v1`.
        config_file: Optional Hermes config path override for tests.

    Returns:
        The config path that was written.

    Raises:
        WrapError: When the config cannot be parsed or written.
    """
    target = hermes_config_path() if config_file is None else config_file
    backup = hermes_backup_path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _snapshot_original(target, backup)
        payload = _read_yaml_mapping(target)
        target.write_text(_hermes_config_content(payload, proxy_url), encoding="utf-8")
    except (OSError, yaml.YAMLError) as exc:
        raise WrapError(f"could not update Hermes config: {exc}") from exc
    return target


def disable_hermes_config(config_file: Path | None = None) -> Path:
    """Restore or remove acr's transient Hermes config change.

    Args:
        config_file: Optional Hermes config path override for tests.

    Returns:
        The config path that was restored or cleaned.

    Raises:
        WrapError: When the config cannot be restored.
    """
    target = hermes_config_path() if config_file is None else config_file
    backup = hermes_backup_path(target)
    try:
        if backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(target))
            return target
        if not target.exists():
            return target
        content = target.read_text(encoding="utf-8")
        if _HERMES_START_MARKER in content:
            target.unlink()
    except OSError as exc:
        raise WrapError(f"could not restore Hermes config: {exc}") from exc
    return target


def launch_env(agent: str, proxy_url: str) -> dict[str, str]:
    """Build the environment for one wrapped agent.

    Args:
        agent: One of `claude` or `hermes`.
        proxy_url: Base acr listen URL without `/v1`.

    Returns:
        A copy of the process environment with the relevant base URL override.
    """
    env = dict(os.environ)
    if agent == "claude":
        env["ANTHROPIC_BASE_URL"] = proxy_url
    elif agent == "hermes":
        env["OPENAI_BASE_URL"] = f"{proxy_url}/v1"
    return env


def _snapshot_original(config_file: Path, backup_file: Path) -> None:
    if backup_file.exists() or not config_file.exists():
        return
    content = config_file.read_text(encoding="utf-8")
    if _HERMES_START_MARKER in content:
        return
    shutil.copy2(config_file, backup_file)


def _read_yaml_mapping(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    loaded = cast("JsonValue", yaml.safe_load(content) or {})
    if not isinstance(loaded, dict):
        raise WrapError("Hermes config must be a YAML mapping")
    return loaded


def _hermes_config_content(payload: JsonObject, proxy_url: str) -> str:
    updated: JsonObject = dict(payload)
    model_value = updated.get("model")
    model: JsonObject = {}
    if isinstance(model_value, dict):
        model.update(model_value)
    model["base_url"] = f"{proxy_url}/v1"
    headers_value = model.get("default_headers")
    default_headers: JsonObject = {}
    if isinstance(headers_value, dict):
        default_headers.update(headers_value)
    default_headers["x-acr-agent"] = "hermes"
    model["default_headers"] = default_headers
    updated["model"] = model
    dumped = yaml.safe_dump(updated, sort_keys=False)
    return f"{_HERMES_START_MARKER}\n{dumped}"
