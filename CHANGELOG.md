# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [Unreleased]

### Added

- Added ChatGPT-OAuth routed serving for Hermes. Decision and premium turns pass
  through to `https://chatgpt.com/backend-api/codex` with the client's own OAuth;
  cheap tool-result turns are served by smaller GPT models (`gpt-5.4-mini`,
  `gpt-5.3-codex-spark`) on the same ChatGPT plan. Routing reuses the existing
  plan quota — it changes which model serves a turn, not the billing source.
- Added `auth.mode: oauth` tier authentication so agent-local tiers can serve on
  the client's ChatGPT OAuth instead of an API key.
- Added `POST /v1/responses` inbound support so Hermes can route on either the
  OpenAI Chat Completions or the Responses wire, with conversion between the
  Anthropic and Responses formats.

### Fixed

- Fixed a startup schema-validation failure (`Input should be 'api_key_env'`)
  that rejected `auth.mode: oauth` tiers and disabled routing, forcing all
  traffic to premium passthrough.
- Fixed Codex OAuth Responses SSE parsing so routed turns rebuild
  `response.output` from streamed text deltas when the final completed event
  reports an empty output array.

### Notes

- This branch (`codex/support-codex-hermes`) originally targeted both a
  standalone `codex` client and Hermes. The standalone `codex` agent group and
  its experimental ChatGPT WebSocket transport were dropped before release; only
  Hermes ships and all routing is HTTP-only. The Responses serving that codex
  used is folded into the `hermes` group.

## [0.5.0] - 2026-06-17

### Added

- Added provider-specific YAML files under `~/.ai-calls-router/config/` for
  `claude-code.yaml` and `hermes.yaml`.
- Added startup bootstrap that creates missing provider config files without
  overwriting operator-edited files.
- Added canonical assembly from global `config.yaml` plus provider YAML files
  into the existing `routes["agents"]` shape used by the decision core.
- Added validation that rejects cheap-tier `key_env` entries in provider files;
  cheap provider credentials stay under tier config.
- Added `router:` identity policy with endpoint defaults, `user_agent_map`, and
  fail-closed `fallback: null`.
- Added strict JSON structural typing and stricter static checks across package
  and tests.

### Changed

- Passthrough now targets the resolved agent group's configured upstream instead
  of always using the global premium upstream.
- `acr init` now seeds router policy and relies on bootstrap for provider files.
- `config.example.yaml` now shows agent-local tiers so Claude Code and Hermes
  can route the same semantic tier names to different models.

### Fixed

- Fixed cross-provider passthrough so Hermes and Claude Code non-routed turns go
  to their own configured upstreams.
- Fixed unresolved identity handling so the router returns `400` before any
  upstream request when a strict router policy cannot attribute the agent.
- Removed unrecognized type-checker config keys while keeping strict checking
  active.

### Tests

- Added provider-config tests for YAML assembly, missing and malformed provider
  files, identity precedence, fail-closed attribution, key rejection, and
  immutability.
- Added bootstrap tests proving provider files are create-only, idempotent,
  valid, and free of cheap-tier keys.
- Added compatibility smoke coverage for Claude Code and Hermes routed and
  premium turns.

## [0.4.0] - 2026-06-17

### Added

- Added `POST /v1/chat/completions` support for Hermes sessions that speak the
  OpenAI Chat Completions wire format.
- Added an OpenAI Chat adapter with Hermes as its default agent group.
- Added deterministic Chat-to-Anthropic request conversion for routed turns.
- Added Anthropic-to-Chat response synthesis for non-streaming routed responses.
- Added Chat Completions SSE synthesis for streamed routed responses.
- Added Hermes pending-tool extraction from the last run of OpenAI tool-result
  messages.

### Changed

- Generalized server handling so adapter-backed endpoints share the same routing
  and passthrough path.
- Threaded the real request path through routing metrics instead of hardcoding
  the Anthropic Messages path.

### Tests

- Added Chat conversion, SSE, adapter, and server integration tests.
- Verified malformed Chat requests fail open to passthrough.
- Verified Hermes premium-tool responses escalate to passthrough.

## [0.3.0] - 2026-06-14

### Added

- Added the OpenAI compatibility roadmap and Phase 0 findings for multi-client
  routing.
- Added per-agent tool-map planning for Claude Code and Hermes.
- Added risk notes for credential isolation, fail-open routing, cache stability,
  and provider-file drift.

## [0.2.0] - 2026-06-14

### Added

- Added the initial Claude Code routing proxy.
- Added DeepSeek direct Anthropic-native serving for cache-stable routed calls.
- Added savings ledger and live dashboard foundations.
