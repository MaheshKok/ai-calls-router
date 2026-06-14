"""Persistent Claude settings integration for desktop-style routing.

The desktop helper manages Claude's persistent settings JSON safely: it records the
previous ``ANTHROPIC_BASE_URL`` state in an acr-owned sidecar, writes only the
routing value it owns, and restores/removes that value on request.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ai_calls_router._lib import config

ENV_KEY = "ANTHROPIC_BASE_URL"
DEFAULT_SETTINGS_RELATIVE_PATH = Path(".claude") / "settings.json"
BACKUP_FILENAME = "desktop_backup.json"


class DesktopError(Exception):
    """Raised when persistent Claude settings cannot be read or written safely."""


@dataclass(frozen=True)
class DesktopResult:
    """Result of a desktop settings operation.

    Attributes:
        changed: True when the command intentionally changed a file.
        message: User-facing summary that avoids printing unrelated environment
            values, which may contain secrets.
    """

    changed: bool
    message: str


@dataclass(frozen=True)
class DesktopBackup:
    """acr-owned sidecar describing the state to restore on disable.

    Attributes:
        config_path: Absolute Claude settings path the backup belongs to.
        previous: Prior ``ANTHROPIC_BASE_URL`` string, or None when absent.
        env_existed: Whether the top-level ``env`` object existed before enable.
        file_existed: Whether the settings file existed before enable.
    """

    config_path: str
    previous: str | None
    env_existed: bool
    file_existed: bool

    def to_json(self) -> dict[str, Any]:
        """Return the stable JSON representation written to the sidecar."""
        return {
            "config_path": self.config_path,
            "env_existed": self.env_existed,
            "file_existed": self.file_existed,
            "previous": self.previous,
        }


def default_settings_path() -> Path:
    """Return the default persistent Claude settings path.

    Returns:
        ``~/.claude/settings.json`` expanded against the current user home.
    """
    return Path.home() / DEFAULT_SETTINGS_RELATIVE_PATH


def default_backup_path() -> Path:
    """Return the acr-owned backup sidecar path.

    Returns:
        ``$ACR_HOME/desktop_backup.json`` when ``ACR_HOME`` is set, otherwise
        ``~/.ai-calls-router/desktop_backup.json``.
    """
    return config.home_dir() / BACKUP_FILENAME


def resolve_settings_path(path: str | Path | None) -> Path:
    """Resolve a user-provided settings path or the safe default.

    Args:
        path: Optional CLI ``--config`` path.

    Returns:
        Absolute, user-expanded settings path. The file need not exist.
    """
    selected = default_settings_path() if path is None else Path(path)
    return selected.expanduser().resolve(strict=False)


def enable(
    *,
    settings_path: Path,
    proxy_url: str,
    backup_path: Path | None = None,
) -> DesktopResult:
    """Enable persistent Claude routing through the acr proxy.

    Args:
        settings_path: Claude settings JSON file to update.
        proxy_url: Proxy URL to store in ``env.ANTHROPIC_BASE_URL``.
        backup_path: Optional acr sidecar path for tests; defaults to
            :func:`default_backup_path`.

    Returns:
        Operation result with a safe user-facing message.

    Raises:
        DesktopError: If existing JSON is malformed, has an invalid shape, or a
            backup belongs to a different settings file.
    """
    resolved_settings = resolve_settings_path(settings_path)
    resolved_backup = _resolve_backup_path(backup_path)
    settings, file_existed = _read_settings(resolved_settings)
    env_existed, previous = _current_env_state(settings)

    backup = _read_backup(resolved_backup)
    if backup is None:
        _write_backup(
            resolved_backup,
            DesktopBackup(
                config_path=str(resolved_settings),
                previous=previous,
                env_existed=env_existed,
                file_existed=file_existed,
            ),
        )
    else:
        _ensure_backup_matches(backup, resolved_settings)

    updated = copy.deepcopy(settings)
    env = _ensure_env_mapping(updated)
    env[ENV_KEY] = proxy_url
    _write_settings(resolved_settings, updated)
    return DesktopResult(
        changed=True,
        message=(
            f"desktop routing enabled: {ENV_KEY}={proxy_url} in "
            f"{resolved_settings}. Start the proxy with `acr start`; "
            "check `acr status` before launching Claude."
        ),
    )


def disable(
    *,
    settings_path: Path,
    proxy_url: str,
    backup_path: Path | None = None,
) -> DesktopResult:
    """Disable persistent Claude routing and restore the backed-up state.

    Args:
        settings_path: Claude settings JSON file to restore.
        proxy_url: Current configured acr proxy URL, used only to diagnose an
            unmanaged live value when no backup exists.
        backup_path: Optional acr sidecar path for tests; defaults to
            :func:`default_backup_path`.

    Returns:
        Operation result with a safe user-facing message.

    Raises:
        DesktopError: If settings or backup JSON cannot be read safely.
    """
    resolved_settings = resolve_settings_path(settings_path)
    resolved_backup = _resolve_backup_path(backup_path)
    settings, _file_existed_now = _read_settings(resolved_settings)
    backup = _read_backup(resolved_backup)

    if backup is None:
        live = _current_base_url(settings)
        if live == proxy_url:
            return DesktopResult(
                changed=False,
                message=(
                    f"desktop routing is not enabled by acr: no backup exists, "
                    f"but {resolved_settings} still points at {proxy_url}; "
                    "leaving it untouched."
                ),
            )
        return DesktopResult(
            changed=False,
            message="desktop routing is not enabled by acr; nothing to restore.",
        )

    _ensure_backup_matches(backup, resolved_settings)
    updated = copy.deepcopy(settings)
    env = _ensure_env_mapping(updated)
    if backup.previous is None:
        env.pop(ENV_KEY, None)
        if not backup.env_existed and not env:
            updated.pop("env", None)
    else:
        env[ENV_KEY] = backup.previous

    if not backup.file_existed and not updated:
        if resolved_settings.exists():
            resolved_settings.unlink()
    else:
        _write_settings(resolved_settings, updated)
    resolved_backup.unlink(missing_ok=True)
    return DesktopResult(
        changed=True,
        message=f"desktop routing disabled for {resolved_settings}; previous setting restored.",
    )


def status(
    *,
    settings_path: Path,
    proxy_url: str,
    backup_path: Path | None = None,
) -> DesktopResult:
    """Report persistent desktop routing state without mutating files.

    Args:
        settings_path: Claude settings JSON file to inspect.
        proxy_url: Configured acr proxy URL used for comparison.
        backup_path: Optional acr sidecar path for tests; defaults to
            :func:`default_backup_path`.

    Returns:
        Operation result whose message reports on/off/mismatch state and the
        configured proxy URL. The operation never prints unrelated env values.
    """
    resolved_settings = resolve_settings_path(settings_path)
    resolved_backup = _resolve_backup_path(backup_path)
    try:
        settings, _file_existed = _read_settings(resolved_settings)
    except DesktopError as exc:
        return DesktopResult(changed=False, message=f"desktop routing status unknown: {exc}")

    live = _current_base_url(settings)
    live_display = live if live is not None else "unset"
    backup_display = "present" if resolved_backup.exists() else "absent"
    if live == proxy_url:
        state = "on"
    elif live is None:
        state = "off"
    else:
        state = "mismatch"
    return DesktopResult(
        changed=False,
        message=(
            f"desktop routing {state}: {ENV_KEY}={live_display}; "
            f"configured proxy={proxy_url}; backup={backup_display}; "
            f"settings={resolved_settings}"
        ),
    )


def _resolve_backup_path(path: Path | None) -> Path:
    selected = default_backup_path() if path is None else path
    return selected.expanduser().resolve(strict=False)


def _read_settings(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except JSONDecodeError as exc:
        raise DesktopError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise DesktopError(f"{path} must contain a JSON object")
    return parsed, True


def _write_settings(path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _current_env_state(settings: dict[str, Any]) -> tuple[bool, str | None]:
    if "env" not in settings:
        return False, None
    env = settings["env"]
    if not isinstance(env, dict):
        raise DesktopError("Claude settings env must be a JSON object")
    value = env.get(ENV_KEY)
    if value is None:
        return True, None
    if not isinstance(value, str):
        raise DesktopError(f"Claude settings env.{ENV_KEY} must be a string")
    return True, value


def _current_base_url(settings: dict[str, Any]) -> str | None:
    env = settings.get("env")
    if not isinstance(env, dict):
        return None
    value = env.get(ENV_KEY)
    return value if isinstance(value, str) else None


def _ensure_env_mapping(settings: dict[str, Any]) -> dict[str, Any]:
    if "env" not in settings:
        env: dict[str, Any] = {}
        settings["env"] = env
    else:
        env = settings["env"]
    if not isinstance(env, dict):
        raise DesktopError("Claude settings env must be a JSON object")
    return env


def _read_backup(path: Path) -> DesktopBackup | None:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise DesktopError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise DesktopError(f"{path} must contain a JSON object")

    config_path = parsed.get("config_path")
    previous = parsed.get("previous")
    env_existed = parsed.get("env_existed")
    file_existed = parsed.get("file_existed")
    if not isinstance(config_path, str):
        raise DesktopError(f"{path} is missing string config_path")
    if previous is not None and not isinstance(previous, str):
        raise DesktopError(f"{path} previous must be a string or null")
    if not isinstance(env_existed, bool):
        raise DesktopError(f"{path} is missing boolean env_existed")
    if not isinstance(file_existed, bool):
        raise DesktopError(f"{path} is missing boolean file_existed")
    return DesktopBackup(
        config_path=config_path,
        previous=previous,
        env_existed=env_existed,
        file_existed=file_existed,
    )


def _write_backup(path: Path, backup: DesktopBackup) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(backup.to_json(), indent=2) + "\n", encoding="utf-8")


def _ensure_backup_matches(backup: DesktopBackup, settings_path: Path) -> None:
    backup_settings_path = Path(backup.config_path).expanduser().resolve(strict=False)
    if backup_settings_path != settings_path:
        raise DesktopError(
            "desktop routing backup belongs to "
            f"{backup_settings_path}; run `acr desktop off --config {backup_settings_path}` "
            "before enabling a different settings file."
        )
