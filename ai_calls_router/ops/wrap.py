"""Agent launcher helpers for routing local CLIs through acr.

Codex and Hermes need persistent local config overrides because subscription
auth ignores `OPENAI_BASE_URL` in some flows. `unwrap` restores the byte-for-byte
backups when the user wants to return each agent to its original upstream.
"""

from __future__ import annotations

import json
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
    "codex": "codex",
    "hermes": "hermes",
}

_CODEX_START_MARKER = "# >>> ai-calls-router codex wrap >>>"
_CODEX_END_MARKER = "# <<< ai-calls-router codex wrap <<<"
_CODEX_BACKUP_SUFFIX = ".acr-backup"
_CODEX_TOP_LEVEL_OVERRIDES = ("model_provider", "openai_base_url")
_HERMES_START_MARKER = "# >>> ai-calls-router hermes wrap >>>"
_HERMES_BACKUP_SUFFIX = ".acr-backup"


def codex_config_path() -> Path:
    """Return the Codex CLI config path.

    Returns:
        The user's `~/.codex/config.toml` path.
    """
    return Path.home() / ".codex" / "config.toml"


def codex_backup_path(config_file: Path | None = None) -> Path:
    """Return the byte-for-byte backup path for a Codex config file.

    Args:
        config_file: Optional Codex config path override.

    Returns:
        The backup path used by `unwrap codex`.
    """
    target = codex_config_path() if config_file is None else config_file
    return target.with_name(f"{target.name}{_CODEX_BACKUP_SUFFIX}")


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


def hermes_auth_path(config_file: Path | None = None) -> Path:
    """Return the Hermes auth store path paired with a config file.

    Args:
        config_file: Optional Hermes config path override.

    Returns:
        The `auth.json` file used by that Hermes home.
    """
    target = hermes_config_path() if config_file is None else config_file
    return target.with_name("auth.json")


def hermes_auth_backup_path(auth_file: Path | None = None) -> Path:
    """Return the byte-for-byte backup path for a Hermes auth store.

    Args:
        auth_file: Optional Hermes auth store path override.

    Returns:
        The backup path used by `unwrap hermes`.
    """
    target = hermes_auth_path() if auth_file is None else auth_file
    return target.with_name(f"{target.name}{_HERMES_BACKUP_SUFFIX}")


def enable_codex_config(proxy_url: str, config_file: Path | None = None) -> Path:
    """Inject acr's Codex provider blocks into `config.toml`.

    Args:
        proxy_url: Base acr listen URL without `/v1`.
        config_file: Optional Codex config path override for tests.

    Returns:
        The config path that was written.

    Raises:
        WrapError: When the file cannot be written.
    """
    target = codex_config_path() if config_file is None else config_file
    backup = codex_backup_path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _snapshot_original(target, backup)
        content = target.read_text(encoding="utf-8") if target.exists() else ""
        cleaned = _strip_top_level_overrides(strip_codex_config(content))
        target.write_text(_codex_config_content(cleaned, proxy_url), encoding="utf-8")
    except OSError as exc:
        raise WrapError(f"could not update Codex config: {exc}") from exc
    return target


def disable_codex_config(config_file: Path | None = None) -> Path:
    """Restore or remove acr's Codex config injection.

    Args:
        config_file: Optional Codex config path override for tests.

    Returns:
        The config path that was restored or cleaned.

    Raises:
        WrapError: When the config cannot be restored.
    """
    target = codex_config_path() if config_file is None else config_file
    backup = codex_backup_path(target)
    try:
        if backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(target))
            return target
        if not target.exists():
            return target
        cleaned = strip_codex_config(target.read_text(encoding="utf-8"))
        if cleaned:
            target.write_text(cleaned, encoding="utf-8")
        else:
            target.unlink()
    except OSError as exc:
        raise WrapError(f"could not restore Codex config: {exc}") from exc
    return target


def enable_hermes_config(
    proxy_url: str,
    config_file: Path | None = None,
    *,
    auth_file: Path | None = None,
) -> Path:
    """Point Hermes openai-codex traffic at acr until `unwrap hermes`.

    Args:
        proxy_url: Base acr listen URL without `/v1`.
        config_file: Optional Hermes config path override for tests.
        auth_file: Optional Hermes auth store override for tests.

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
        _enable_hermes_auth(proxy_url, hermes_auth_path(target) if auth_file is None else auth_file)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise WrapError(f"could not update Hermes config: {exc}") from exc
    return target


def disable_hermes_config(
    config_file: Path | None = None,
    *,
    auth_file: Path | None = None,
) -> Path:
    """Restore or remove acr's transient Hermes config change.

    Args:
        config_file: Optional Hermes config path override for tests.
        auth_file: Optional Hermes auth store override for tests.

    Returns:
        The config path that was restored or cleaned.

    Raises:
        WrapError: When the config cannot be restored.
    """
    target = hermes_config_path() if config_file is None else config_file
    backup = hermes_backup_path(target)
    auth_target = hermes_auth_path(target) if auth_file is None else auth_file
    try:
        if backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(target))
            _disable_hermes_auth(auth_target)
            return target
        if not target.exists():
            _disable_hermes_auth(auth_target)
            return target
        content = target.read_text(encoding="utf-8")
        if _HERMES_START_MARKER in content:
            target.unlink()
        _disable_hermes_auth(auth_target)
    except OSError as exc:
        raise WrapError(f"could not restore Hermes config: {exc}") from exc
    return target


def launch_env(agent: str, proxy_url: str) -> dict[str, str]:
    """Build the environment for one wrapped agent.

    Args:
        agent: One of `claude`, `codex`, or `hermes`.
        proxy_url: Base acr listen URL without `/v1`.

    Returns:
        A copy of the process environment with the relevant base URL override.
    """
    env = dict(os.environ)
    if agent == "claude":
        env["ANTHROPIC_BASE_URL"] = proxy_url
    elif agent == "hermes":
        env["OPENAI_BASE_URL"] = f"{proxy_url}/v1"
        env["HERMES_CODEX_BASE_URL"] = f"{proxy_url}/v1"
    else:
        env["OPENAI_BASE_URL"] = f"{proxy_url}/v1"
    return env


def strip_codex_config(content: str) -> str:
    """Remove acr-managed Codex blocks from TOML text.

    Args:
        content: Existing `config.toml` text.

    Returns:
        Text without acr marker blocks.
    """
    stripped = _strip_marker_spans(content)
    return stripped.lstrip("\n").rstrip() + "\n" if stripped.strip() else ""


def _snapshot_original(config_file: Path, backup_file: Path) -> None:
    if backup_file.exists() or not config_file.exists():
        return
    content = config_file.read_text(encoding="utf-8")
    if _CODEX_START_MARKER in content or _CODEX_END_MARKER in content:
        return
    shutil.copy2(config_file, backup_file)


def _strip_marker_spans(content: str) -> str:
    text = content
    while _CODEX_START_MARKER in text and _CODEX_END_MARKER in text:
        start = text.index(_CODEX_START_MARKER)
        end_idx = text.index(_CODEX_END_MARKER, start)
        end = end_idx + len(_CODEX_END_MARKER)
        text = text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")
    return text.replace(f"{_CODEX_START_MARKER}\n", "").replace(f"{_CODEX_END_MARKER}\n", "")


def _strip_top_level_overrides(content: str) -> str:
    result: list[str] = []
    in_top_level = True
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        if in_top_level and stripped.startswith("["):
            in_top_level = False
        if in_top_level and _is_top_level_override(stripped):
            continue
        result.append(line)
    return "".join(result)


def _is_top_level_override(stripped_line: str) -> bool:
    return any(
        stripped_line.startswith(f"{key} ") or stripped_line.startswith(f"{key}=")
        for key in _CODEX_TOP_LEVEL_OVERRIDES
    )


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


def _enable_hermes_auth(proxy_url: str, auth_file: Path) -> None:
    if not auth_file.exists():
        return
    backup = hermes_auth_backup_path(auth_file)
    _snapshot_original(auth_file, backup)
    payload = _read_json_mapping(auth_file)
    _set_hermes_auth_base_url(payload, f"{proxy_url}/v1")
    auth_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _disable_hermes_auth(auth_file: Path) -> None:
    backup = hermes_auth_backup_path(auth_file)
    if backup.exists():
        auth_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup), str(auth_file))


def _read_json_mapping(path: Path) -> JsonObject:
    loaded = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        raise WrapError("Hermes auth store must be a JSON mapping")
    return loaded


def _set_hermes_auth_base_url(payload: JsonObject, base_url: str) -> None:
    providers = payload.get("providers")
    if isinstance(providers, dict):
        provider = providers.get("openai-codex")
        if isinstance(provider, dict):
            provider["base_url"] = base_url
    credential_pool = payload.get("credential_pool")
    if not isinstance(credential_pool, dict):
        return
    entries = credential_pool.get("openai-codex")
    if not isinstance(entries, list):
        return
    for entry in entries:
        if isinstance(entry, dict):
            entry["base_url"] = base_url


def _codex_config_content(user_content: str, proxy_url: str) -> str:
    top_level = (
        f"{_CODEX_START_MARKER}\n"
        'model_provider = "acr"\n'
        f'openai_base_url = "{proxy_url}/v1"\n'
        f"{_CODEX_END_MARKER}\n"
    )
    provider = (
        f"{_CODEX_START_MARKER}\n"
        "[model_providers.acr]\n"
        'name = "OpenAI via ai-calls-router"\n'
        f'base_url = "{proxy_url}/v1"\n'
        "supports_websockets = true\n"
        f"{_CODEX_END_MARKER}\n"
    )
    cleaned = user_content.strip()
    if not cleaned:
        return f"{top_level}\n{provider}"
    return f"{top_level}\n{cleaned}\n\n{provider}"
