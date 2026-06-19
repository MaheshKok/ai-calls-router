"""Tests for local agent wrapper config edits."""

from __future__ import annotations

import json
from pathlib import Path

from ai_calls_router.ops import wrap


def test_codex_default_paths_honor_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert wrap.codex_config_path() == tmp_path / ".codex" / "config.toml"
    assert wrap.codex_backup_path() == tmp_path / ".codex" / "config.toml.acr-backup"


def test_hermes_default_paths_honor_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert wrap.hermes_config_path() == tmp_path / ".hermes" / "config.yaml"
    assert wrap.hermes_backup_path() == tmp_path / ".hermes" / "config.yaml.acr-backup"


def test_enable_codex_config_writes_top_level_and_provider_blocks(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[profiles.default]\nmodel = "gpt-5"\n', encoding="utf-8")

    wrap.enable_codex_config("http://127.0.0.1:8747", config_file)

    content = config_file.read_text(encoding="utf-8")
    assert content.startswith(
        "# >>> ai-calls-router codex wrap >>>\n"
        'model_provider = "acr"\n'
        'openai_base_url = "http://127.0.0.1:8747/v1"\n'
    )
    assert "[model_providers.acr]" in content
    assert "supports_websockets = true" in content
    assert '[profiles.default]\nmodel = "gpt-5"' in content
    assert wrap.codex_backup_path(config_file).read_text(encoding="utf-8") == (
        '[profiles.default]\nmodel = "gpt-5"\n'
    )


def test_enable_codex_config_is_idempotent(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"

    wrap.enable_codex_config("http://127.0.0.1:8747", config_file)
    wrap.enable_codex_config("http://127.0.0.1:9321", config_file)

    content = config_file.read_text(encoding="utf-8")
    assert content.count("# >>> ai-calls-router codex wrap >>>") == 2
    assert "8747" not in content
    assert 'openai_base_url = "http://127.0.0.1:9321/v1"' in content


def test_enable_codex_config_replaces_existing_top_level_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'model_provider = "openai"\n'
        'openai_base_url = "https://chatgpt.com/backend-api/codex"\n'
        '[profiles.default]\nmodel_provider = "profile-local"\n',
        encoding="utf-8",
    )

    wrap.enable_codex_config("http://127.0.0.1:8747", config_file)

    content = config_file.read_text(encoding="utf-8")
    assert content.count("model_provider =") == 2
    assert content.count("openai_base_url =") == 1
    assert 'model_provider = "profile-local"' in content


def test_disable_codex_config_restores_backup_byte_for_byte(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    original = '[profiles.default]\nmodel = "gpt-5"\n'
    config_file.write_text(original, encoding="utf-8")

    wrap.enable_codex_config("http://127.0.0.1:8747", config_file)
    wrap.disable_codex_config(config_file)

    assert config_file.read_text(encoding="utf-8") == original
    assert not wrap.codex_backup_path(config_file).exists()


def test_disable_codex_config_without_backup_strips_marker_blocks(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        wrap.strip_codex_config("")
        + '# >>> ai-calls-router codex wrap >>>\nmodel_provider = "acr"\n'
        "# <<< ai-calls-router codex wrap <<<\n\n"
        '[profiles.default]\nmodel = "gpt-5"\n',
        encoding="utf-8",
    )

    wrap.disable_codex_config(config_file)

    assert config_file.read_text(encoding="utf-8") == '[profiles.default]\nmodel = "gpt-5"\n'


def test_disable_codex_config_without_existing_file_is_noop(tmp_path: Path) -> None:
    config_file = tmp_path / "missing.toml"

    assert wrap.disable_codex_config(config_file) == config_file

    assert not config_file.exists()


def test_disable_codex_config_removes_marker_only_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    wrap.enable_codex_config("http://127.0.0.1:8747", config_file)
    wrap.codex_backup_path(config_file).unlink(missing_ok=True)

    wrap.disable_codex_config(config_file)

    assert not config_file.exists()


def test_enable_hermes_config_sets_model_base_url_and_preserves_backup(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    original = (
        "model:\n"
        "  default: gpt-5.5\n"
        "  provider: openai-codex\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n"
        "  default_headers:\n"
        "    user-agent: hermes\n"
    )
    config_file.write_text(original, encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)

    content = config_file.read_text(encoding="utf-8")
    assert content.startswith("# >>> ai-calls-router hermes wrap >>>\n")
    assert "provider: openai-codex" in content
    assert "default: gpt-5.5" in content
    assert "base_url: http://127.0.0.1:8747/v1" in content
    assert "user-agent: hermes" in content
    assert "x-acr-agent: hermes" in content
    assert wrap.hermes_backup_path(config_file).read_text(encoding="utf-8") == original


def test_enable_hermes_config_sets_codex_auth_pool_base_url(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    auth_file = tmp_path / "auth.json"
    original_auth = {
        "providers": {"openai-codex": {"base_url": "http://127.0.0.1:8787/v1"}},
        "credential_pool": {
            "openai-codex": [
                {"id": "primary", "base_url": "http://127.0.0.1:8787/v1"},
                {"id": "secondary"},
            ]
        },
    }
    config_file.write_text("model:\n  provider: openai-codex\n", encoding="utf-8")
    auth_file.write_text(json.dumps(original_auth, indent=2) + "\n", encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file, auth_file=auth_file)

    patched = json.loads(auth_file.read_text(encoding="utf-8"))
    assert patched["providers"]["openai-codex"]["base_url"] == "http://127.0.0.1:8747/v1"
    assert patched["credential_pool"]["openai-codex"][0]["base_url"] == ("http://127.0.0.1:8747/v1")
    assert patched["credential_pool"]["openai-codex"][1]["base_url"] == ("http://127.0.0.1:8747/v1")
    assert json.loads(wrap.hermes_auth_backup_path(auth_file).read_text(encoding="utf-8")) == (
        original_auth
    )


def test_disable_hermes_config_restores_backup_byte_for_byte(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    original = "model:\n  provider: openai-codex\n"
    config_file.write_text(original, encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)
    wrap.disable_hermes_config(config_file)

    assert config_file.read_text(encoding="utf-8") == original
    assert not wrap.hermes_backup_path(config_file).exists()


def test_disable_hermes_config_restores_auth_backup_byte_for_byte(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    auth_file = tmp_path / "auth.json"
    original_auth = '{"credential_pool":{"openai-codex":[{"base_url":"old"}]}}\n'
    config_file.write_text("model:\n  provider: openai-codex\n", encoding="utf-8")
    auth_file.write_text(original_auth, encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file, auth_file=auth_file)
    wrap.disable_hermes_config(config_file, auth_file=auth_file)

    assert auth_file.read_text(encoding="utf-8") == original_auth
    assert not wrap.hermes_auth_backup_path(auth_file).exists()


def test_disable_hermes_config_removes_marker_only_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)
    wrap.disable_hermes_config(config_file)

    assert not config_file.exists()
