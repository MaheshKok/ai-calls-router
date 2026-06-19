"""Tests for local agent wrapper config edits."""

from __future__ import annotations

from pathlib import Path

from ai_calls_router.ops import wrap


def test_hermes_default_paths_honor_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert wrap.hermes_config_path() == tmp_path / ".hermes" / "config.yaml"
    assert wrap.hermes_backup_path() == tmp_path / ".hermes" / "config.yaml.acr-backup"


def test_enable_hermes_config_sets_model_base_url_and_preserves_backup(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    original = (
        "model:\n"
        "  default: gpt-4.1\n"
        "  provider: openai\n"
        "  base_url: https://api.openai.com/v1\n"
        "  default_headers:\n"
        "    user-agent: hermes\n"
    )
    config_file.write_text(original, encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)

    content = config_file.read_text(encoding="utf-8")
    assert content.startswith("# >>> ai-calls-router hermes wrap >>>\n")
    assert "provider: openai" in content
    assert "default: gpt-4.1" in content
    assert "base_url: http://127.0.0.1:8747/v1" in content
    assert "user-agent: hermes" in content
    assert "x-acr-agent: hermes" in content
    assert wrap.hermes_backup_path(config_file).read_text(encoding="utf-8") == original


def test_disable_hermes_config_restores_backup_byte_for_byte(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    original = "model:\n  provider: openai\n"
    config_file.write_text(original, encoding="utf-8")

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)
    wrap.disable_hermes_config(config_file)

    assert config_file.read_text(encoding="utf-8") == original
    assert not wrap.hermes_backup_path(config_file).exists()


def test_disable_hermes_config_removes_marker_only_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    wrap.enable_hermes_config("http://127.0.0.1:8747", config_file)
    wrap.disable_hermes_config(config_file)

    assert not config_file.exists()


def test_launch_env_sets_only_supported_agent_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    claude = wrap.launch_env("claude", "http://127.0.0.1:8747")
    hermes = wrap.launch_env("hermes", "http://127.0.0.1:8747")

    assert claude["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8747"
    assert hermes["OPENAI_BASE_URL"] == "http://127.0.0.1:8747/v1"
    assert "ANTHROPIC_BASE_URL" not in hermes
