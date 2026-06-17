# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [0.1.1](https://github.com/MaheshKok/ai-calls-router/compare/ai-calls-router-v0.1.0...ai-calls-router-v0.1.1) (2026-06-17)


### Features

* add desktop routing setup ([7dda4bf](https://github.com/MaheshKok/ai-calls-router/commit/7dda4bf6b504c9f49c183b8b970fb3211997cea2))
* add deterministic tool_result reducer on DeepSeek direct path ([2031c5b](https://github.com/MaheshKok/ai-calls-router/commit/2031c5b35f1d51352d6565528cc2b5517d381963))
* add industry-grade quality tooling stack (phases 1-3) ([e8c8ee3](https://github.com/MaheshKok/ai-calls-router/commit/e8c8ee312bef98ac60f97ca25322be05173411cf))
* add live /metrics endpoint, per-request counters, and savings-ledger enrichment ([90f898d](https://github.com/MaheshKok/ai-calls-router/commit/90f898de2255cbe08634439eba791381d2d40506))
* add pre-commit bandit hook and comprehensive CI quality gates ([232abaa](https://github.com/MaheshKok/ai-calls-router/commit/232abaa9ed8536e5b6501d8f672e6e75fd470e19))
* add structured usage logging to routing engine ([2788627](https://github.com/MaheshKok/ai-calls-router/commit/2788627c78b0d9e282c381d61dc9b13e2d19be05))
* add wemake-python-styleguide (flake8) configuration and pre-commit hook ([9ab14d3](https://github.com/MaheshKok/ai-calls-router/commit/9ab14d31aa6b060ead919b837d246551acb23cde))
* DeepSeek direct Anthropic-format routing with cache-aware ledger pricing ([0a14203](https://github.com/MaheshKok/ai-calls-router/commit/0a14203452e65203c1524e0c7940108db2f43392))
* Phase 3 - savings ledger and accounting ([8aa07c4](https://github.com/MaheshKok/ai-calls-router/commit/8aa07c42c3a67c80994f2041256675c5487094f7))
* Phase 4 - built-in compression for routed request bodies ([dd2fb8f](https://github.com/MaheshKok/ai-calls-router/commit/dd2fb8f32e71d0bbeb1f6189d0a367c420e7f2a6))
* Phase 5 - routed call engine ([7d04c4f](https://github.com/MaheshKok/ai-calls-router/commit/7d04c4f3c88ec0d59799b0753ed6551cba913592))
* Phase 6 - passthrough proxy and routing server ([1a9e7b8](https://github.com/MaheshKok/ai-calls-router/commit/1a9e7b8488f772b7b2a51f97789eaec15d7c8839))
* Phase 7 - daemon, CLI, and init wizard ([d7fcf53](https://github.com/MaheshKok/ai-calls-router/commit/d7fcf53df9b02809d4eaf861b4d43885615577c9))
* Phase 8 - integration suite, pricing wiring, packaging, and docs ([b271b4d](https://github.com/MaheshKok/ai-calls-router/commit/b271b4d68965b52fa2286132157e0f4efd11664d))
* replace mypy with pyright, expand Ruff rules, add hardening tools ([5f90e22](https://github.com/MaheshKok/ai-calls-router/commit/5f90e22888638641d88af9a815de184482b7b14a))
* scaffold project with config and routing engine ([3dc4105](https://github.com/MaheshKok/ai-calls-router/commit/3dc41053ef4c081fa90d23f933b145a0f53b4c3e))
* single-source version from __init__.py, add CI/CD workflows ([ddd8a2f](https://github.com/MaheshKok/ai-calls-router/commit/ddd8a2f13919067fb4423567f86caadb1ba3665c))


### Bug Fixes

* use fully-qualified imports for acr_testkit to resolve PyCharm errors ([c876843](https://github.com/MaheshKok/ai-calls-router/commit/c876843dbeff829ad2eee29dc6a790060835f759))

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
