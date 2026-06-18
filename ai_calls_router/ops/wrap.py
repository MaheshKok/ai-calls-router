"""Agent launcher helpers for routing local CLIs through acr.

Codex needs a persistent `~/.codex/config.toml` override because ChatGPT
subscription auth ignores `OPENAI_BASE_URL` in some flows. Claude and Hermes
only need environment overrides, so this module keeps their handling transient.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


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
