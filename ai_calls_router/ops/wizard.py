"""Interactive configuration wizard behind the acr init command.

Interviews the user through an injectable ask function -- provider preset
(deepseek, groq, kimi, openrouter) or fully custom per-tier models, key
environment variable, and listen port -- and writes a complete config.yaml
that the proxy can serve immediately, with an overwrite confirmation when a
config already exists. LiteLLM-routed presets carry no price overrides (cost
accounting uses LiteLLM's own table); the DeepSeek preset carries cache-aware
overrides because its native Anthropic endpoint bypasses LiteLLM, leaving its
ids unpriced unless the config supplies the rates.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ai_calls_router._lib import config

DEFAULT_PROVIDER = "deepseek"

# Per-provider tier presets: model per tier plus the conventional key env
# var. Models are starting points the user can edit in config.yaml.
PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "fast": "deepseek/deepseek-v4-flash",
        "code": "deepseek/deepseek-v4-pro",
        "crud": "deepseek/deepseek-v4-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "groq": {
        "fast": "groq/llama-3.3-70b-versatile",
        "code": "groq/llama-3.3-70b-versatile",
        "crud": "groq/llama-3.1-8b-instant",
        "key_env": "GROQ_API_KEY",
    },
    "kimi": {
        "fast": "moonshot/kimi-k2-0905-preview",
        "code": "moonshot/kimi-k2-0905-preview",
        "crud": "moonshot/kimi-k2-0905-preview",
        "key_env": "MOONSHOT_API_KEY",
    },
    "openrouter": {
        "fast": "openrouter/qwen/qwen3-coder",
        "code": "openrouter/moonshotai/kimi-k2",
        "crud": "openrouter/qwen/qwen3-coder",
        "key_env": "OPENROUTER_API_KEY",
    },
}

# Cache-aware price overrides (USD per 1M tokens) for direct-endpoint presets.
# DeepSeek serves its native Anthropic endpoint directly, bypassing LiteLLM, so
# its fictional V4 ids are absent from LiteLLM's pricing table; without these
# the cache-aware ledger has nothing to price the routed side with and stays
# empty. Cache reads bill at input_cached_cost_per_1m, misses at
# input_cost_per_1m. PLACEHOLDERS -- verify against DeepSeek's published rates.
# LiteLLM-routed presets (groq, kimi, openrouter) intentionally appear here with
# no entry so they rely on LiteLLM's own table.
PRESET_PRICES: dict[str, dict[str, dict[str, float]]] = {
    "deepseek": {
        "fast": {
            "input_cost_per_1m": 0.14,
            "input_cached_cost_per_1m": 0.0028,
            "output_cost_per_1m": 0.28,
        },
        "code": {
            "input_cost_per_1m": 0.435,
            "input_cached_cost_per_1m": 0.003625,
            "output_cost_per_1m": 0.87,
        },
        "crud": {
            "input_cost_per_1m": 0.14,
            "input_cached_cost_per_1m": 0.0028,
            "output_cost_per_1m": 0.28,
        },
    },
}

TIER_MAX_TOKENS: dict[str, int] = {"fast": 8192, "code": 8192, "crud": 4096}

DEFAULT_TOOLS: dict[str, str] = {
    "Bash": "fast",
    "BashOutput": "fast",
    "KillShell": "fast",
    "WebFetch": "fast",
    "WebSearch": "fast",
    "Read": "code",
    "Grep": "code",
    "Glob": "code",
    "LSP": "code",
    "TodoWrite": "crud",
    "TaskList": "crud",
    "TaskGet": "crud",
    "Edit": "premium",
    "Write": "premium",
    "MultiEdit": "premium",
    "NotebookEdit": "premium",
    "Task": "premium",
    "ExitPlanMode": "premium",
    "AskUserQuestion": "premium",
}

PREMIUM_TOOLS: list[str] = [
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "Task",
    "ExitPlanMode",
    "AskUserQuestion",
]

AskFn = Callable[[str], str]


def _ask_port(ask: AskFn) -> int:
    """Ask for the proxy listen port, falling back to the default.

    Args:
        ask: Prompt function returning the user's raw answer.

    Returns:
        The chosen port, or config.DEFAULT_PORT when the answer is empty
        or not a valid port number.
    """
    answer = ask(f"Proxy port [{config.DEFAULT_PORT}]: ").strip()
    if not answer.isdigit():
        return config.DEFAULT_PORT
    port = int(answer)
    if not 1 <= port <= 65535:
        return config.DEFAULT_PORT
    return port


def _ask_models(ask: AskFn, provider: str) -> dict[str, str]:
    """Resolve per-tier models for the chosen provider.

    Presets answer directly; the custom flow asks one LiteLLM model id per
    tier, with empty answers falling back to the default preset's model.

    Args:
        ask: Prompt function returning the user's raw answer.
        provider: Normalized provider answer ("custom" or a preset name).

    Returns:
        Mapping of tier name to LiteLLM model id.
    """
    if provider != "custom":
        preset = PRESETS.get(provider, PRESETS[DEFAULT_PROVIDER])
        return {tier: preset[tier] for tier in TIER_MAX_TOKENS}

    fallback = PRESETS[DEFAULT_PROVIDER]
    models: dict[str, str] = {}
    for tier in TIER_MAX_TOKENS:
        answer = ask(f"LiteLLM model for tier '{tier}' (provider/model): ").strip()
        models[tier] = answer or fallback[tier]
    return models


def _default_key_env(provider: str, models: dict[str, str]) -> str:
    """Derive the default key env var for the chosen provider.

    Args:
        provider: Normalized provider answer ("custom" or a preset name).
        models: Resolved per-tier models (used to derive a name for custom
            providers from the fast model's LiteLLM prefix).

    Returns:
        The conventional API key environment variable name.
    """
    if provider != "custom":
        return PRESETS.get(provider, PRESETS[DEFAULT_PROVIDER])["key_env"]
    prefix = models["fast"].split("/", 1)[0]
    return f"{prefix.upper().replace('-', '_')}_API_KEY"


def _build_config(
    *, port: int, models: dict[str, str], key_env: str, provider: str
) -> dict[str, Any]:
    """Assemble the full config.yaml mapping.

    Args:
        port: Proxy listen port.
        models: Per-tier LiteLLM model ids.
        key_env: API key environment variable shared by all tiers.
        provider: Normalized provider answer; selects cache-aware price
            overrides from PRESET_PRICES for direct-endpoint presets.

    Returns:
        The config mapping ready to serialize as YAML. Direct-endpoint presets
        carry cache-aware price overrides; every other tier omits prices so
        LiteLLM's pricing table is the only cost source.
    """
    prices = PRESET_PRICES.get(provider, {})
    tiers = {
        tier: {
            "model": models[tier],
            "key_env": key_env,
            "max_tokens": TIER_MAX_TOKENS[tier],
            **prices.get(tier, {}),
        }
        for tier in TIER_MAX_TOKENS
    }
    return {
        "server": {
            "host": config.DEFAULT_HOST,
            "port": port,
            "upstream": config.DEFAULT_UPSTREAM,
        },
        "premium": {"provider": "anthropic"},
        "settings": {
            "tier_precedence": ["premium", "code", "fast", "crud"],
            "premium_tools": list(PREMIUM_TOOLS),
            "escalate_on_premium_tools": True,
        },
        "tiers": tiers,
        "tools": dict(DEFAULT_TOOLS),
    }


def run_wizard(ask: AskFn = input) -> Path:
    """Run the acr init interview and write config.yaml.

    Args:
        ask: Prompt function returning the user's raw answer (injectable
            for tests; defaults to builtin input).

    Returns:
        The config.yaml path -- freshly written, or the untouched existing
        file when the user declines to overwrite it.
    """
    path = config.config_path()
    if path.exists():
        answer = ask(f"{path} exists. Overwrite? [y/N]: ").strip().lower()
        if not answer.startswith("y"):
            return path

    providers = ", ".join([*PRESETS, "custom"])
    provider = ask(f"Cheap provider ({providers}) [{DEFAULT_PROVIDER}]: ").strip().lower()
    if provider != "custom" and provider not in PRESETS:
        provider = DEFAULT_PROVIDER

    models = _ask_models(ask, provider)
    default_key = _default_key_env(provider, models)
    key_env = ask(f"API key env var [{default_key}]: ").strip() or default_key
    port = _ask_port(ask)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            _build_config(port=port, models=models, key_env=key_env, provider=provider),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
