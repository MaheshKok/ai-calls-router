"""Tests for persistent Claude desktop settings routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_calls_router.ops import desktop

PROXY_URL = "http://127.0.0.1:8747"
ALT_PROXY_URL = "http://127.0.0.1:9999"
OLD_PROXY_URL = "http://127.0.0.1:8787"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_on_creates_settings_file_when_absent(tmp_path: Path) -> None:
    """Enabling desktop routing creates a minimal Claude settings file."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"

    result = desktop.enable(
        settings_path=settings_path,
        proxy_url=PROXY_URL,
        backup_path=backup_path,
    )

    assert result.changed is True
    assert _read_json(settings_path) == {"env": {"ANTHROPIC_BASE_URL": PROXY_URL}}
    assert _read_json(backup_path) == {
        "config_path": str(settings_path),
        "env_existed": False,
        "file_existed": False,
        "previous": None,
    }


def test_on_sets_base_url_to_configured_proxy(tmp_path: Path) -> None:
    """The configured proxy URL is the only persisted ANTHROPIC_BASE_URL value."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text('{"env": {}}\n', encoding="utf-8")

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert _read_json(settings_path)["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL


def test_on_preserves_unrelated_top_level_and_env_keys(tmp_path: Path) -> None:
    """Enabling desktop routing does not expose or rewrite unrelated settings."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(git status)"]},
                "model": "claude-sonnet-4",
                "env": {"SECRET_TOKEN": "do-not-print"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = desktop.enable(
        settings_path=settings_path,
        proxy_url=PROXY_URL,
        backup_path=backup_path,
    )

    updated = _read_json(settings_path)
    assert updated["permissions"] == {"allow": ["Bash(git status)"]}
    assert updated["model"] == "claude-sonnet-4"
    assert updated["env"]["SECRET_TOKEN"] == "do-not-print"
    assert updated["env"]["ANTHROPIC_BASE_URL"] == PROXY_URL
    assert "do-not-print" not in result.message


def test_on_backs_up_existing_base_url_value(tmp_path: Path) -> None:
    """The precedence-trap value is backed up before being overwritten."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": OLD_PROXY_URL}}, indent=2) + "\n",
        encoding="utf-8",
    )

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert _read_json(backup_path) == {
        "config_path": str(settings_path),
        "env_existed": True,
        "file_existed": True,
        "previous": OLD_PROXY_URL,
    }


def test_on_is_idempotent_and_keeps_original_backup(tmp_path: Path) -> None:
    """Running on repeatedly updates the live URL without replacing the backup."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": OLD_PROXY_URL}}, indent=2) + "\n",
        encoding="utf-8",
    )

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)
    first_backup = backup_path.read_text(encoding="utf-8")
    desktop.enable(settings_path=settings_path, proxy_url=ALT_PROXY_URL, backup_path=backup_path)

    assert backup_path.read_text(encoding="utf-8") == first_backup
    assert _read_json(settings_path)["env"]["ANTHROPIC_BASE_URL"] == ALT_PROXY_URL


def test_off_restores_previous_string_value(tmp_path: Path) -> None:
    """Disabling desktop routing restores the exact previous base URL value."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": OLD_PROXY_URL}}, indent=2) + "\n",
        encoding="utf-8",
    )

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)
    desktop.disable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert _read_json(settings_path) == {"env": {"ANTHROPIC_BASE_URL": OLD_PROXY_URL}}
    assert not backup_path.exists()


def test_off_removes_key_when_it_was_originally_absent(tmp_path: Path) -> None:
    """Disabling removes only the acr-added key when the env block already existed."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps({"model": "claude", "env": {"KEEP_ME": "1"}}, indent=2) + "\n",
        encoding="utf-8",
    )

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)
    desktop.disable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert _read_json(settings_path) == {"model": "claude", "env": {"KEEP_ME": "1"}}
    assert not backup_path.exists()


def test_off_with_no_backup_is_safe_noop(tmp_path: Path) -> None:
    """Disabling without an acr sidecar never guesses at user intent."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    original = json.dumps({"env": {"ANTHROPIC_BASE_URL": PROXY_URL}}, indent=2) + "\n"
    settings_path.write_text(original, encoding="utf-8")

    result = desktop.disable(
        settings_path=settings_path,
        proxy_url=PROXY_URL,
        backup_path=backup_path,
    )

    assert result.changed is False
    assert "no backup" in result.message
    assert settings_path.read_text(encoding="utf-8") == original


def test_malformed_settings_json_fails_without_writing(tmp_path: Path) -> None:
    """Malformed Claude settings fail loudly and remain byte-for-byte untouched."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    malformed = '{"env": '
    settings_path.write_text(malformed, encoding="utf-8")

    with pytest.raises(desktop.DesktopError):
        desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)
    with pytest.raises(desktop.DesktopError):
        desktop.disable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert settings_path.read_text(encoding="utf-8") == malformed
    assert not backup_path.exists()


def test_status_reports_on_off_and_mismatch_without_mutating(tmp_path: Path) -> None:
    """Status reports the live value and leaves settings untouched."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": OLD_PROXY_URL}}, indent=2) + "\n",
        encoding="utf-8",
    )
    before = settings_path.read_text(encoding="utf-8")

    result = desktop.status(
        settings_path=settings_path,
        proxy_url=PROXY_URL,
        backup_path=backup_path,
    )

    assert result.changed is False
    assert "mismatch" in result.message
    assert OLD_PROXY_URL in result.message
    assert settings_path.read_text(encoding="utf-8") == before


def test_off_removes_acr_added_empty_env_block_only_when_empty(tmp_path: Path) -> None:
    """If acr created the env block and it is empty after disable, remove it."""
    settings_path = tmp_path / "settings.json"
    backup_path = tmp_path / "desktop_backup.json"
    settings_path.write_text(json.dumps({"model": "claude"}, indent=2) + "\n", encoding="utf-8")

    desktop.enable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)
    desktop.disable(settings_path=settings_path, proxy_url=PROXY_URL, backup_path=backup_path)

    assert _read_json(settings_path) == {"model": "claude"}
    assert not backup_path.exists()
