"""Tier credential resolution: the API key or OAuth mode for a serving tier.

Resolves a tier's credential from config.yaml and the environment: an OAuth
sentinel for ChatGPT/Codex OAuth tiers, otherwise the API key named by the
tier's ``key_env`` (process environment first, then the configured ``env_file``),
with an ``OPENAI_API_KEY`` fallback for Codex tiers. Key values are never logged
and any schema or read failure returns ``None`` so callers fail open to
passthrough. This module owns credentials only -- no tier selection and no
topology.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ai_calls_router.routing.config_schema import (
    CODEX_OAUTH_SENTINEL,
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject

logger = logging.getLogger("acr.routing")


@dataclass(frozen=True)
class TierCredential:
    """Resolved tier credential and auth mode."""

    value: str
    auth_mode: Literal["api_key", "oauth"]


def _search_env_file(env_path: Path, key_env: str) -> str | None:
    """Search a .env file for a variable, returning its value or None."""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        if name.strip() in (key_env, f"export {key_env}"):
            return raw.strip().strip("'\"") or None
    return None


def resolve_api_key(tier_cfg: JsonObject, settings: JsonObject) -> str | None:
    """Resolve the provider API key for a tier.

    Looks up the env var named by key_env in the process environment first,
    then in the env_file configured under settings (simple KEY=VALUE lines).
    Key values are never logged.

    Args:
        tier_cfg: Tier entry from config.yaml.
        settings: The settings: section of config.yaml.

    Returns:
        The key, or None when unavailable (callers must fail open).
    """
    key_env = tier_cfg.get("key_env")
    if not isinstance(key_env, str) or not key_env:
        return None
    value = os.environ.get(key_env)
    if value:
        return value

    env_file = settings.get("env_file")
    if not isinstance(env_file, str) or not env_file:
        return None
    try:
        env_path = Path(env_file).expanduser()
        return _search_env_file(env_path, key_env)
    except Exception as exc:
        logger.warning("acr: env_file read failed (%s)", exc, exc_info=True)
    return None


def _auth_key_env(tier_cfg: JsonObject) -> str | None:
    """Return flat or nested api-key env var name for a tier."""
    auth = tier_cfg.get("auth")
    if isinstance(auth, dict) and auth.get("mode") == "api_key_env":
        key_env = auth.get("key_env")
        return key_env if isinstance(key_env, str) and key_env else None
    key_env = tier_cfg.get("key_env")
    return key_env if isinstance(key_env, str) and key_env else None


def _auth_mode(tier_cfg: JsonObject) -> str:
    """Return nested tier auth mode."""
    auth = tier_cfg.get("auth")
    if isinstance(auth, dict):
        mode = auth.get("mode")
        return mode if isinstance(mode, str) else ""
    return ""


def resolve_tier_credential(tier_cfg: JsonObject, settings: JsonObject) -> TierCredential | None:
    """Resolve a tier credential and auth mode.

    Args:
        tier_cfg: Tier entry from config.yaml.
        settings: The settings: section of config.yaml.

    Returns:
        The resolved credential metadata, or None when unavailable.
    """
    try:
        parsed = parse_tier_config(tier_cfg)
    except ConfigSchemaError as exc:
        logger.warning("acr: tier schema validation failed (%s); passing through", exc)
        return None
    if _auth_mode(tier_cfg) == "oauth" or (
        parsed.key_env == CODEX_OAUTH_SENTINEL and is_codex_tier(tier_cfg)
    ):
        return TierCredential(value=CODEX_OAUTH_SENTINEL, auth_mode="oauth")
    credential = resolve_api_key({**tier_cfg, "key_env": _auth_key_env(tier_cfg)}, settings)
    if credential:
        return TierCredential(value=credential, auth_mode="api_key")
    if is_codex_tier(tier_cfg):
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return TierCredential(value=openai_key, auth_mode="api_key")
    return None
