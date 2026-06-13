# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [0.1.0] - 2026-06-13

### Added

- Standalone reverse proxy that routes Claude Code's tool-result-processing turns
  to a cheap LiteLLM-supported model while passing every decision-making turn
  through to Anthropic untouched.
- Tool-to-tier routing engine with premium-tool escalation and a fail-open
  passthrough on every error path.
- Built-in tool-result compressor with an optional `rtk` backend.
- JSONL savings ledger and an `acr savings` report that prices routed models from
  LiteLLM's pricing table and omits models it cannot price.
- `acr` command-line interface: `init`, `start`, `stop`, `status`, `serve`,
  `code`, `savings`, and `version`.
- Background daemon with pidfile, log, and health-check management.
- Interactive `acr init` configuration wizard with provider presets.

[0.1.0]: https://github.com/maheshkokare/ai-calls-router/releases/tag/v0.1.0
