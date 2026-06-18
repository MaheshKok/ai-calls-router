"""Tests for local agent wrapper config edits."""

from __future__ import annotations

from pathlib import Path

from ai_calls_router.ops import wrap


def test_codex_default_paths_honor_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert wrap.codex_config_path() == tmp_path / ".codex" / "config.toml"
    assert wrap.codex_backup_path() == tmp_path / ".codex" / "config.toml.acr-backup"


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
