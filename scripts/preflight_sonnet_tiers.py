"""Preflight check for the claude_code Sonnet-OAuth tier migration.

Replicates the server's runtime route assembly (assemble_routes over the global
config.yaml plus per-provider files) against the real on-disk config and asserts
that all four cheap claude_code tiers resolve to anthropic/claude-sonnet-4-6 on
the OAuth credential. Run before any billed live test to catch a schema or merge
regression without spending quota.
"""

from __future__ import annotations

import sys

from ai_calls_router.routing import provider_config
from ai_calls_router.routing.anthropic_oauth import is_anthropic_oauth_tier, native_model_id
from ai_calls_router.routing.credential_resolution import resolve_tier_credential
from ai_calls_router.routing.decide import load_routes

EXPECTED_MODEL = "claude-sonnet-4-6"
CHEAP_TIERS = ("fast", "code", "crud", "structured")


def main() -> int:
    """Assert the assembled claude_code tiers are Sonnet on OAuth; return exit code."""
    routes = provider_config.assemble_routes(
        load_routes(), provider_files=provider_config.load_provider_files()
    )
    agent = routes["agents"]["claude_code"]
    settings = routes.get("settings", {})
    tiers = agent["tiers"]

    failures: list[str] = []
    for name in CHEAP_TIERS:
        cfg = tiers.get(name)
        if cfg is None:
            failures.append(f"{name}: missing from assembled tiers")
            continue
        cred = resolve_tier_credential(cfg, settings)
        native = native_model_id(cfg) if is_anthropic_oauth_tier(cfg) else "<not-anthropic>"
        ok = (
            is_anthropic_oauth_tier(cfg)
            and native == EXPECTED_MODEL
            and cred is not None
            and cred.auth_mode == "oauth"
        )
        marker = "OK " if ok else "FAIL"
        cred_mode = cred.auth_mode if cred else None
        print(f"  [{marker}] {name}: native={native} auth={cred_mode}")
        if not ok:
            failures.append(f"{name}: native={native} cred={cred}")

    if failures:
        print("\nPREFLIGHT FAILED:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nPREFLIGHT OK: all four claude_code cheap tiers -> Sonnet 4.6 on OAuth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
