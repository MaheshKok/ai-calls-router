"""Spec-derived tests for ai_calls_router.config.

Contract under test: path helpers honor their env-var overrides at call time,
server settings fail open to safe defaults on missing or malformed config, and
the premium block rejects anything but the anthropic provider in v1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_calls_router._lib import config


class TestPaths:
    """Path helpers resolve under ~/.ai-calls-router with env overrides."""

    def test_config_path_defaults_to_home_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACR_CONFIG", raising=False)
        monkeypatch.delenv("ACR_HOME", raising=False)
        assert config.config_path() == Path.home() / ".ai-calls-router" / "config.yaml"

    def test_config_path_honors_acr_config_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        override = tmp_path / "custom.yaml"
        monkeypatch.setenv("ACR_CONFIG", str(override))
        assert config.config_path() == override

    def test_acr_home_relocates_all_default_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("ACR_CONFIG", raising=False)
        monkeypatch.delenv("ACR_SAVINGS_LEDGER", raising=False)
        monkeypatch.setenv("ACR_HOME", str(tmp_path))
        assert config.config_path() == tmp_path / "config.yaml"
        assert config.pid_path() == tmp_path / "acr.pid"
        assert config.log_path() == tmp_path / "acr.log"
        assert config.ledger_path() == tmp_path / "savings.jsonl"

    def test_ledger_path_honors_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        override = tmp_path / "ledger.jsonl"
        monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(override))
        assert config.ledger_path() == override


class TestServerSettings:
    """server_settings reads server: block, failing open to defaults."""

    def test_defaults_on_empty_config(self) -> None:
        settings = config.server_settings({})
        assert settings.host == "127.0.0.1"
        assert settings.port == 8747
        assert settings.upstream == "https://api.anthropic.com"

    def test_reads_configured_values(self) -> None:
        routes = {
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
                "upstream": "https://example.com",
            }
        }
        settings = config.server_settings(routes)
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.upstream == "https://example.com"

    def test_malformed_server_block_falls_back_to_defaults(self) -> None:
        settings = config.server_settings({"server": "not-a-dict"})
        assert settings.port == 8747

    def test_non_numeric_port_falls_back_to_default(self) -> None:
        settings = config.server_settings({"server": {"port": "eight"}})
        assert settings.port == 8747

    def test_upstream_trailing_slash_stripped(self) -> None:
        settings = config.server_settings({"server": {"upstream": "https://x.com/"}})
        assert settings.upstream == "https://x.com"


class TestPremiumValidation:
    """v1 accepts only the anthropic premium provider."""

    def test_missing_premium_block_is_valid(self) -> None:
        config.validate_premium({})

    def test_anthropic_provider_is_valid(self) -> None:
        config.validate_premium({"premium": {"provider": "anthropic"}})

    def test_other_provider_raises(self) -> None:
        with pytest.raises(config.ConfigError):
            config.validate_premium({"premium": {"provider": "openai"}})

    def test_malformed_premium_block_raises(self) -> None:
        with pytest.raises(config.ConfigError):
            config.validate_premium({"premium": "anthropic"})
