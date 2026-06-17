"""Spec-derived tests for ai_calls_router.wizard.

Contract under test: run_wizard interviews the user through an injectable
ask function (provider preset or custom models, key env var, port -- with an
overwrite confirmation when config.yaml already exists), and writes a valid
config.yaml that the rest of the proxy can serve: the premium block passes
v1 validation, tiers carry provider-prefixed models with a key_env, the
default tool map routes cheap tools to tiers and editing tools to premium,
and empty answers fall back to the documented defaults. Phase 7 keeps the
router block in config.yaml and materializes per-provider agent YAML files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ai_calls_router._lib import config
from ai_calls_router.ops import wizard
from ai_calls_router.routing import decide, provider_config


class _ScriptedAsk:
    """Injectable ask function replaying canned answers."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.answers:
            return self.answers.pop(0)
        return ""


@pytest.fixture
def acr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the acr home directory at a temp dir; config under it."""
    monkeypatch.setenv("ACR_HOME", str(tmp_path))
    monkeypatch.delenv("ACR_CONFIG", raising=False)
    return tmp_path


def _written_config(path: Path) -> dict[str, object]:
    """Parse the YAML the wizard wrote."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestDefaults:
    def test_all_defaults_write_deepseek_preset(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        assert path == config.config_path()
        written = _written_config(path)
        assert written["server"]["port"] == config.DEFAULT_PORT
        for tier in ("fast", "structured", "code", "crud"):
            tier_cfg = written["tiers"][tier]
            assert tier_cfg["model"].startswith("deepseek/")
            assert tier_cfg["key_env"] == "DEEPSEEK_API_KEY"

    def test_generated_config_passes_v1_validation(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        written = _written_config(path)
        config.validate_premium(written)
        assert written["premium"]["provider"] == "anthropic"

    def test_generated_server_block_resolves(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        settings = config.server_settings(_written_config(path))
        assert settings.port == config.DEFAULT_PORT
        assert settings.upstream == config.DEFAULT_UPSTREAM

    def test_generated_config_contains_all_agent_groups(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        assert "agents" not in _written_config(path)
        loaded = provider_config.load_provider_files()
        assert set(loaded) == {"claude_code", "hermes", "codex"}
        for cfg in loaded.values():
            assert cfg["tools"]
            assert cfg["premium_tools"]
            assert cfg["upstream"]

    def test_generated_config_round_trips_through_load_routes(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        routes = provider_config.assemble_routes(
            decide.load_routes(path),
            provider_files=provider_config.load_provider_files(),
        )
        assert decide.tier_for_tools(["Bash"], routes, group="claude_code") == "fast"
        assert decide.tier_for_tools(["exec_command"], routes, group="codex") == "fast"
        assert decide.tier_for_tools(["terminal"], routes, group="hermes") == "fast"

    def test_default_tools_route_cheap_to_tiers_and_editing_to_premium(
        self, acr_home: Path
    ) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        assert path.exists()
        tools = yaml.safe_load(
            config.provider_config_path("claude_code").read_text(encoding="utf-8")
        )["tools"]
        assert tools["Bash"] == "fast"
        assert tools["BashOutput"] == "fast"
        assert tools["KillShell"] == "fast"
        assert tools["Read"] == "code"
        assert tools["LSP"] == "code"
        assert tools["TodoWrite"] == "crud"
        assert tools["TaskList"] == "crud"
        assert tools["TaskGet"] == "crud"
        for tool in ("Edit", "Write", "MultiEdit", "NotebookEdit", "Task"):
            assert tools[tool] == "premium"

    def test_settings_block_carries_routing_defaults(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk([]))
        written = _written_config(path)
        settings = written["settings"]
        assert written["router"]["endpoint_defaults"]["/v1/responses"] == "codex"
        assert written["router"]["fallback"] is None
        assert settings["escalate_on_premium_tools"] is True
        assert settings["tier_precedence"][0] == "premium"
        assert "premium_tools" not in settings


class TestAnswers:
    def test_chosen_port_written(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", "9100"]))
        assert _written_config(path)["server"]["port"] == 9100

    def test_invalid_port_falls_back_to_default(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", "not-a-port"]))
        assert _written_config(path)["server"]["port"] == config.DEFAULT_PORT

    def test_out_of_range_port_falls_back_to_default(self, acr_home: Path) -> None:
        """A numeric-but-out-of-range port (>65535) is rejected for the default.

        "70000" passes the isdigit() guard but fails the 1..65535 range check,
        exercising the second fallback branch in _ask_port (wizard.py:99).
        """
        path = wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", "70000"]))
        assert _written_config(path)["server"]["port"] == config.DEFAULT_PORT

    def test_unknown_provider_falls_back_to_default_preset(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["no-such-provider", "", ""]))
        written = _written_config(path)
        assert written["tiers"]["fast"]["model"].startswith("deepseek/")

    def test_custom_key_env_overrides_preset(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "MY_KEY", ""]))
        written = _written_config(path)
        for tier in ("fast", "code", "crud"):
            assert written["tiers"][tier]["key_env"] == "MY_KEY"

    def test_groq_preset_uses_groq_models_and_key(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["groq", "", ""]))
        written = _written_config(path)
        assert written["tiers"]["fast"]["model"].startswith("groq/")
        assert written["tiers"]["fast"]["key_env"] == "GROQ_API_KEY"


class TestCustomProvider:
    def test_custom_models_written_per_tier(self, acr_home: Path) -> None:
        answers = [
            "custom",
            "openrouter/qwen/qwen3-coder",
            "openrouter/qwen/qwen3-coder",
            "openrouter/moonshotai/kimi-k2",
            "openrouter/qwen/qwen3-coder",
            "OPENROUTER_API_KEY",
            "8800",
        ]
        path = wizard.run_wizard(ask=_ScriptedAsk(answers))
        written = _written_config(path)
        assert written["tiers"]["fast"]["model"] == "openrouter/qwen/qwen3-coder"
        assert written["tiers"]["structured"]["model"] == "openrouter/qwen/qwen3-coder"
        assert written["tiers"]["code"]["model"] == "openrouter/moonshotai/kimi-k2"
        assert written["tiers"]["crud"]["model"] == "openrouter/qwen/qwen3-coder"
        assert written["tiers"]["fast"]["key_env"] == "OPENROUTER_API_KEY"
        assert written["server"]["port"] == 8800

    def test_custom_tiers_carry_no_fabricated_prices(self, acr_home: Path) -> None:
        answers = ["custom", "x/m1", "x/m1", "x/m2", "x/m3", "X_KEY", ""]
        path = wizard.run_wizard(ask=_ScriptedAsk(answers))
        written = _written_config(path)
        for tier in ("fast", "structured", "code", "crud"):
            assert "input_cost_per_1m" not in written["tiers"][tier]
            assert "input_cached_cost_per_1m" not in written["tiers"][tier]
            assert "output_cost_per_1m" not in written["tiers"][tier]


class TestPresetPricing:
    """DeepSeek routes through its native endpoint (bypassing LiteLLM), so the
    generated config must carry cache-aware prices for it; LiteLLM-routed
    presets must not, leaving pricing to LiteLLM's own table."""

    def test_deepseek_preset_writes_cache_aware_prices(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", ""]))
        tiers = _written_config(path)["tiers"]
        for tier in ("fast", "structured", "code", "crud"):
            cfg = tiers[tier]
            miss = cfg["input_cost_per_1m"]
            hit = cfg["input_cached_cost_per_1m"]
            out = cfg["output_cost_per_1m"]
            assert all(isinstance(v, int | float) for v in (miss, hit, out))
            assert miss > 0
            assert hit > 0
            assert out > 0
            # The whole point of the direct path: a cache hit is far cheaper
            # than a miss. A swapped hit/miss rate fails here.
            assert hit < miss

    def test_deepseek_code_tier_pricier_than_fast_tier(self, acr_home: Path) -> None:
        # The code tier runs the pro model; it must not be cheaper than fast,
        # which would mean the flash/pro prices were transposed.
        tiers = _written_config(wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", ""])))["tiers"]
        assert tiers["code"]["input_cost_per_1m"] >= tiers["fast"]["input_cost_per_1m"]

    def test_deepseek_prices_match_example_config(self, acr_home: Path) -> None:
        # The wizard preset and the shipped config.example.yaml must agree so
        # the documented rates are exactly what `acr init` writes.
        example = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "config.example.yaml").read_text(
                encoding="utf-8"
            )
        )
        written = _written_config(wizard.run_wizard(ask=_ScriptedAsk(["deepseek", "", ""])))
        price_keys = (
            "input_cost_per_1m",
            "input_cached_cost_per_1m",
            "output_cost_per_1m",
        )
        for tier in ("fast", "structured", "code", "crud"):
            for key in price_keys:
                assert written["tiers"][tier][key] == example["tiers"][tier][key]

    def test_groq_preset_carries_no_price_overrides(self, acr_home: Path) -> None:
        path = wizard.run_wizard(ask=_ScriptedAsk(["groq", "", ""]))
        tiers = _written_config(path)["tiers"]
        for tier in ("fast", "structured", "code", "crud"):
            assert "input_cost_per_1m" not in tiers[tier]
            assert "input_cached_cost_per_1m" not in tiers[tier]
            assert "output_cost_per_1m" not in tiers[tier]


class TestOverwrite:
    def test_existing_config_kept_when_declined(self, acr_home: Path) -> None:
        config.config_path().parent.mkdir(parents=True, exist_ok=True)
        config.config_path().write_text("server:\n  port: 1234\n", encoding="utf-8")
        path = wizard.run_wizard(ask=_ScriptedAsk(["n"]))
        assert _written_config(path)["server"]["port"] == 1234

    def test_existing_config_overwritten_when_confirmed(self, acr_home: Path) -> None:
        config.config_path().parent.mkdir(parents=True, exist_ok=True)
        config.config_path().write_text("server:\n  port: 1234\n", encoding="utf-8")
        path = wizard.run_wizard(ask=_ScriptedAsk(["y", "deepseek", "", "9100"]))
        assert _written_config(path)["server"]["port"] == 9100
