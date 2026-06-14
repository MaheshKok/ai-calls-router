# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [0.1.0] - 2026-06-14

First tagged release. Anthropic passthrough routing is verified working
end-to-end against Claude Code (CLI and desktop) via `ANTHROPIC_BASE_URL`.

### Added

#### Proxy and routing

- Standalone reverse proxy that routes Claude Code's tool-result-processing turns
  to a cheap LiteLLM-supported model while passing every decision-making turn
  through to Anthropic untouched.
- Tool-to-tier routing engine with premium-tool escalation and a fail-open
  passthrough on every error path.
- DeepSeek direct path that speaks Anthropic format natively, bypassing LiteLLM
  to preserve byte-for-byte prefix stability for provider-side prefix caching.
- Deterministic `tool_result` reducer on the DeepSeek direct path for stable,
  cache-friendly request prefixes.
- Built-in tool-result compressor with an optional `rtk` backend.
- Structured per-route logging: tier, model, route (direct/litellm), token and
  cache hit/miss counts, and request duration.

#### Accounting

- JSONL savings ledger and an `acr savings` report that prices routed models from
  LiteLLM's pricing table and omits models it cannot price.
- Cache-aware ledger pricing that accounts for cache-read versus cache-creation
  input tokens on supported providers.

#### CLI and operations

- `acr` command-line interface: `init`, `start`, `stop`, `status`, `serve`,
  `code`, `desktop`, `savings`, and `version`.
- Background daemon with pidfile, log, and health-check management.
- Interactive `acr init` configuration wizard with provider presets.
- `acr desktop` for persistent routing by managing the `ANTHROPIC_BASE_URL`
  setting in Claude's settings file, with `on`, `off`, and `status` actions.

#### Conversion fidelity

- Anthropic <-> OpenAI message conversion with golden-pair fidelity tests.

#### Packaging, tooling, and governance

- Layered package architecture (proxy, routing, accounting, ops) with the routed
  call engine split into engine and synthesis modules.
- Single-source versioning from `__init__.py`, PEP 561 `py.typed` marker, and
  reproducible sdist/wheel builds.
- CI/CD workflows plus a blocking quality-gate stack: ruff, pyright, pytest with
  98% coverage floor, deptry, pip-audit, radon/xenon complexity limits, bandit,
  and pre-commit hooks.
- OSS governance files: LICENSE, CONTRIBUTING, SECURITY, Code of Conduct, and
  issue/PR templates.

[0.1.0]: https://github.com/maheshkokare/ai-calls-router/releases/tag/v0.1.0
