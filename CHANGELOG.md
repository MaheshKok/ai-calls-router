# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [0.5.2](https://github.com/MaheshKok/ai-calls-router/compare/ai-calls-router-v0.5.1...ai-calls-router-v0.5.2) (2026-06-23)


### Features

* compress tool_result blocks on the Anthropic OAuth routed path ([0108d1d](https://github.com/MaheshKok/ai-calls-router/commit/0108d1da7639905e8a14202fce122abe1a3c3f1e))
* compress tool_result blocks on the Anthropic OAuth routed path ([34d7951](https://github.com/MaheshKok/ai-calls-router/commit/34d7951717163fa5854ab0207fedc7e310b12106))

## [0.5.1](https://github.com/MaheshKok/ai-calls-router/compare/ai-calls-router-v0.5.0...ai-calls-router-v0.5.1) (2026-06-23)


### Features

* add Claude Desktop embedded Claude Code shim ([a0f26d8](https://github.com/MaheshKok/ai-calls-router/commit/a0f26d89c8e6890244af651eff1d7aeb5533e752))
* add comprehensive logging system with request correlation ([896a95a](https://github.com/MaheshKok/ai-calls-router/commit/896a95a8f369bd957286fabef4359aa136bca4a9))
* add desktop routing setup ([7dda4bf](https://github.com/MaheshKok/ai-calls-router/commit/7dda4bf6b504c9f49c183b8b970fb3211997cea2))
* add deterministic tool_result reducer on DeepSeek direct path ([2031c5b](https://github.com/MaheshKok/ai-calls-router/commit/2031c5b35f1d51352d6565528cc2b5517d381963))
* add industry-grade quality tooling stack (phases 1-3) ([e8c8ee3](https://github.com/MaheshKok/ai-calls-router/commit/e8c8ee312bef98ac60f97ca25322be05173411cf))
* add lifetime routed-calls-per-model counter ([6274490](https://github.com/MaheshKok/ai-calls-router/commit/6274490a40e5029e84858f5d3555ca277450dacc))
* add live /metrics endpoint, per-request counters, and savings-ledger enrichment ([90f898d](https://github.com/MaheshKok/ai-calls-router/commit/90f898de2255cbe08634439eba791381d2d40506))
* add live dashboard, agent/session metadata, and metrics persistence ([c982d23](https://github.com/MaheshKok/ai-calls-router/commit/c982d2390edb2d57d5d6fe8be07b6e820e1fe79a))
* add pre-commit bandit hook and comprehensive CI quality gates ([232abaa](https://github.com/MaheshKok/ai-calls-router/commit/232abaa9ed8536e5b6501d8f672e6e75fd470e19))
* add structured usage logging to routing engine ([2788627](https://github.com/MaheshKok/ai-calls-router/commit/2788627c78b0d9e282c381d61dc9b13e2d19be05))
* add tool_result compression observability ([619e0b1](https://github.com/MaheshKok/ai-calls-router/commit/619e0b18c9dbf26e147ff056ce30afa81334499e))
* add wemake-python-styleguide (flake8) configuration and pre-commit hook ([9ab14d3](https://github.com/MaheshKok/ai-calls-router/commit/9ab14d31aa6b060ead919b837d246551acb23cde))
* bootstrap premium token totals from request_events DB on restart ([406cb09](https://github.com/MaheshKok/ai-calls-router/commit/406cb097fe15599f64a87fd8aabd9054d5da482b))
* canonicalize tool names against the premium tool list ([bdb6913](https://github.com/MaheshKok/ai-calls-router/commit/bdb6913cc1ad387994a4108909f356297f5e2bd9))
* **codex-ws:** hybrid sticky-route WS router + frame synthesizer (flag OFF) ([e5f4671](https://github.com/MaheshKok/ai-calls-router/commit/e5f46712ef673f553eff903a5fc787d1fbc425fd))
* **codex:** add CodexSession store + strip image_generation for routed tiers ([f0f6f42](https://github.com/MaheshKok/ai-calls-router/commit/f0f6f421db4d1aee8af246d25c30762d518f28b2))
* **compression:** add per-tier text_ml_compression flag, wire through both serving paths ([684f967](https://github.com/MaheshKok/ai-calls-router/commit/684f96763f0554dcfb19303e24e8199250f2b15c))
* DeepSeek direct Anthropic-format routing with cache-aware ledger pricing ([0a14203](https://github.com/MaheshKok/ai-calls-router/commit/0a14203452e65203c1524e0c7940108db2f43392))
* default bootstrap to Sonnet 4.6 OAuth (claude_code) + Codex OAuth (hermes) ([782d324](https://github.com/MaheshKok/ai-calls-router/commit/782d3242125495077c01077ea026a4ac8819f4a2))
* extend complexity checks, usage tracking, and dashboard metrics ([fe599d3](https://github.com/MaheshKok/ai-calls-router/commit/fe599d3694b58b559cd6035d3157a27c170a5c84))
* implement premium guard callback and tool tracking ([d8e8c8d](https://github.com/MaheshKok/ai-calls-router/commit/d8e8c8df55036ec5803a08956b93b2782e5403a0))
* limit recent requests display to 20 and update status messages accordingly ([4d1b514](https://github.com/MaheshKok/ai-calls-router/commit/4d1b5147c2aca132fcb75cda15b18bbacc0a33d5))
* **metrics:** persist request history in SQLite ([f6fd82e](https://github.com/MaheshKok/ai-calls-router/commit/f6fd82ef98182933f61325e1d010c4d7dcbd0bea))
* per-row tool_result shrinkage in dashboard + skip zero-input rows ([0282a41](https://github.com/MaheshKok/ai-calls-router/commit/0282a41d8146cfca6036bde2abe4cf2f2134a1ea))
* per-row tool_result shrinkage in dashboard + skip zero-input rows ([7dbd774](https://github.com/MaheshKok/ai-calls-router/commit/7dbd774586d885851cb320d48e2f1fce5b5d7123))
* Phase 3 - savings ledger and accounting ([8aa07c4](https://github.com/MaheshKok/ai-calls-router/commit/8aa07c42c3a67c80994f2041256675c5487094f7))
* Phase 4 - built-in compression for routed request bodies ([dd2fb8f](https://github.com/MaheshKok/ai-calls-router/commit/dd2fb8f32e71d0bbeb1f6189d0a367c420e7f2a6))
* Phase 5 - routed call engine ([7d04c4f](https://github.com/MaheshKok/ai-calls-router/commit/7d04c4f3c88ec0d59799b0753ed6551cba913592))
* Phase 6 - passthrough proxy and routing server ([1a9e7b8](https://github.com/MaheshKok/ai-calls-router/commit/1a9e7b8488f772b7b2a51f97789eaec15d7c8839))
* Phase 7 - daemon, CLI, and init wizard ([d7fcf53](https://github.com/MaheshKok/ai-calls-router/commit/d7fcf53df9b02809d4eaf861b4d43885615577c9))
* Phase 8 - integration suite, pricing wiring, packaging, and docs ([b271b4d](https://github.com/MaheshKok/ai-calls-router/commit/b271b4d68965b52fa2286132157e0f4efd11664d))
* **proxy:** route OpenAI chat completions ([4f26224](https://github.com/MaheshKok/ai-calls-router/commit/4f26224d4177a1f60576f2a0e1b3d6846e085143))
* **proxy:** route OpenAI Responses requests ([2b4bbb0](https://github.com/MaheshKok/ai-calls-router/commit/2b4bbb0b8b1fa91b251c97abef974878bc762ad6))
* **proxy:** use per-agent passthrough upstreams ([399bfbf](https://github.com/MaheshKok/ai-calls-router/commit/399bfbf7ceef2f5b66e10a73724e35bd9f773625))
* redesign dashboard — less clutter, same signal ([4a1c787](https://github.com/MaheshKok/ai-calls-router/commit/4a1c787a275770b1da42cf98629c0657d4ebb6f1))
* redesign dashboard and add routed-by-model panel ([0e16020](https://github.com/MaheshKok/ai-calls-router/commit/0e16020091d4f0734468511bdaaf102f7ac1ba52))
* replace mypy with pyright, expand Ruff rules, add hardening tools ([5f90e22](https://github.com/MaheshKok/ai-calls-router/commit/5f90e22888638641d88af9a815de184482b7b14a))
* **routing:** add agent-local OAuth tiers ([2191c53](https://github.com/MaheshKok/ai-calls-router/commit/2191c533b037cf2a712c7dfeac899787e3ff2623))
* **routing:** add per-agent tool config ([d532f0d](https://github.com/MaheshKok/ai-calls-router/commit/d532f0d043f8a0b9fc34c138cbcf5c9d6dbbb61f))
* **routing:** add provider-specific YAML routing ([7e860d6](https://github.com/MaheshKok/ai-calls-router/commit/7e860d602f13a7712ac958119c3d708185a62f8b))
* **routing:** compress forwarded bodies on premium and codex paths ([03029c4](https://github.com/MaheshKok/ai-calls-router/commit/03029c4769b7882ad07baa456b0ab0be62d628cb))
* **routing:** make routed reasoning effort configurable per tier ([09588ec](https://github.com/MaheshKok/ai-calls-router/commit/09588ec3efd76dd800cf2a9c282db2c04d4f8a45))
* **routing:** restore ChatGPT-OAuth serving for Hermes ([6f8a9f9](https://github.com/MaheshKok/ai-calls-router/commit/6f8a9f9d78027f5aab860a96a5c2d200dc971dba))
* **routing:** route Claude Code cheap turns to Sonnet via OAuth ([0d522ff](https://github.com/MaheshKok/ai-calls-router/commit/0d522ff86f5e07b7f020f6f6060ce18b10778656))
* scaffold project with config and routing engine ([3dc4105](https://github.com/MaheshKok/ai-calls-router/commit/3dc41053ef4c081fa90d23f933b145a0f53b4c3e))
* single-source version from __init__.py, add CI/CD workflows ([ddd8a2f](https://github.com/MaheshKok/ai-calls-router/commit/ddd8a2f13919067fb4423567f86caadb1ba3665c))
* support Codex WebSocket routing ([9e84a19](https://github.com/MaheshKok/ai-calls-router/commit/9e84a1988b3d68fa1ef19d0f06366953c6b4b5d4))


### Bug Fixes

* add adapter seam for routed requests ([9c407df](https://github.com/MaheshKok/ai-calls-router/commit/9c407dfe7cb68a976698badbf05c8fe4a3229196))
* add output column to dashboard, correct cache miss and billing semantics ([5428d9f](https://github.com/MaheshKok/ai-calls-router/commit/5428d9f3591bcc03155deaaaddfa64e13b303fea))
* **cache:** enforce deterministic tool-call serialization for prefix cache ([ca8f54b](https://github.com/MaheshKok/ai-calls-router/commit/ca8f54b5b3bb2f7d079dbce4569d794bbb76413f))
* codex "No tool output found" 400 and missing WS dashboard metrics ([df5afad](https://github.com/MaheshKok/ai-calls-router/commit/df5afadf2f968869d84c38e5848fb2242e017f8b))
* **codex:** report routed tier model ([d4b20d6](https://github.com/MaheshKok/ai-calls-router/commit/d4b20d6d515191788a19fbb1e5b5784eb56bccdc))
* **codex:** strip stale content-length from OAuth headers ([fc1d9fa](https://github.com/MaheshKok/ai-calls-router/commit/fc1d9fa7604a2b6ad5761ca93f347d07cb3efc9f))
* **codex:** support OAuth routed responses ([2783376](https://github.com/MaheshKok/ai-calls-router/commit/2783376e8c4f54871877cd119b39f53499834819))
* correct tool_result flattening, CLI error output, agent label ([761be48](https://github.com/MaheshKok/ai-calls-router/commit/761be488c5db0080a4d5f47611c5295ec69c0fc4))
* harden OAuth header forwarding, timeout validation, and preflight credential logging ([198b3e4](https://github.com/MaheshKok/ai-calls-router/commit/198b3e46f8e8a4e2e16c25edd995b33a3250bfa4))
* **metrics:** single token source of truth for compression, fix request total bug, wire incr_error ([254e59a](https://github.com/MaheshKok/ai-calls-router/commit/254e59ac2fcde3805205515c99a801df66f22e3c))
* **passthrough:** strip opus[1m] / context-1m long-context opt-in so OAuth subscription stops 429ing large turns ([6ac300e](https://github.com/MaheshKok/ai-calls-router/commit/6ac300e963f3c8d01d80f8da2c6252b72367083c))
* **proxy:** harden routed accounting and passthrough paths ([37d4f99](https://github.com/MaheshKok/ai-calls-router/commit/37d4f99701afb85e5154e6257bc91a489298bec2))
* **proxy:** surface Codex WebSocket routing ([a2fedaf](https://github.com/MaheshKok/ai-calls-router/commit/a2fedaf4989fa8fe8fdc9a27e6ab2143ca711512))
* **routing:** normalize Claude system messages ([b44ddc3](https://github.com/MaheshKok/ai-calls-router/commit/b44ddc3fba4d8990c9eb34b06a94f731a359bb3e))
* **routing:** strip blank text blocks to stop empty-content 400 ([4b83de4](https://github.com/MaheshKok/ai-calls-router/commit/4b83de465f5169012c9ce56faa7dde3e8d63220a))
* update coverage threshold to fail under 95% ([d7e707a](https://github.com/MaheshKok/ai-calls-router/commit/d7e707a4539edf0d1daeb81ad85baa59750188d3))
* use fully-qualified imports for acr_testkit to resolve PyCharm errors ([c876843](https://github.com/MaheshKok/ai-calls-router/commit/c876843dbeff829ad2eee29dc6a790060835f759))

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
