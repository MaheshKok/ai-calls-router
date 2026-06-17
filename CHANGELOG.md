# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Releases are automated by [release-please](https://github.com/googleapis/release-please)
from [Conventional Commit](https://www.conventionalcommits.org/) messages, so new
entries are added here when a release pull request is merged.

## [0.5.0] - 2026-06-17

### Added

#### OpenAI Responses / Codex routing

- Added `POST /v1/responses` support through a dedicated OpenAI Responses
  adapter. Codex sessions can now send OpenAI Responses requests to the router
  and receive Responses-shaped output while the routed core continues to operate
  on Anthropic Messages internally.
- Added deterministic Responses-to-Anthropic request conversion for self-contained
  `input[]` conversations, including:
  - bare string input shorthand;
  - `message` items with user, assistant, and system roles;
  - `function_call` and `function_call_output` items;
  - `custom_tool_call` and `custom_tool_call_output` items for freeform tools
    such as `apply_patch`;
  - tool definitions for both function tools and custom/freeform tools;
  - `instructions` to Anthropic top-level `system`;
  - `max_output_tokens` to Anthropic `max_tokens`;
  - deterministic stripping of Responses `reasoning.encrypted_content` on the
    routed path so DeepSeek cache keys never include encrypted, provider-specific
    reasoning bytes.
- Added Anthropic-to-Responses response synthesis for non-streaming routed
  responses, including text output items, function-call output items, usage
  mapping, and `max_tokens` to incomplete-response mapping.
- Added Responses SSE synthesis for streaming clients. The stream emits
  Responses event frames such as `response.created`,
  `response.output_item.added`, `response.output_text.delta`,
  `response.function_call_arguments.delta`, and `response.completed`, then
  terminates in the OpenAI-compatible stream form.
- Added Codex pending-tool extraction from the last run of
  `function_call_output` and `custom_tool_call_output` items. The extractor
  resolves `call_id` values back to prior function/custom tool-call names,
  deduplicates in order, skips hosted-tool calls, and returns `<unknown>` for
  unresolved call IDs so uncertain turns safely escalate.
- Added `/v1/responses` routing through the existing `_try_route` and
  `routed_call` machinery, including streamed and non-streamed responses.

#### Provider-specific configuration

- Added provider-specific YAML files under `~/.ai-calls-router/config/`:
  - `claude-code.yaml`;
  - `codex.yaml`;
  - `hermes.yaml`.
- Added startup bootstrap that creates missing provider config files without
  overwriting operator-edited files.
- Added a canonical assembly layer that merges global `config.yaml` plus
  provider-specific YAML files back into the existing in-memory
  `routes["agents"]` shape consumed by the routing decision core.
- Added validation that rejects cheap-tier `key_env` entries in per-provider
  files. Cheap provider credentials remain global under `tiers.*.key_env`;
  provider files describe the premium side only.
- Added reserved provider-file fields for future-compatible configuration:
  `auth`, `wire`, `endpoints`, `model_defaults`, `tool_choice`, `reasoning`,
  and `fallback`.
- Added `router:` identity policy in global config:
  - `endpoint_defaults` for `/v1/messages`, `/v1/chat/completions`, and
    `/v1/responses`;
  - case-insensitive `user_agent_map` contains-rules;
  - `fallback`, including `null` for fail-closed identity attribution.
- Added identity precedence:
  `x-acr-agent` header > `router.user_agent_map` > `router.endpoint_defaults`
  > `router.fallback` > adapter default.
- Added fail-closed identity behavior when a `router:` block is present,
  `fallback: null`, and no identity rule matches. The server returns
  `400 {"error": "unresolved agent identity"}` before any upstream request is
  attempted.

#### Documentation and examples

- Updated `README.md` for all three live endpoints:
  - `POST /v1/messages` for `claude_code`;
  - `POST /v1/chat/completions` for `hermes`;
  - `POST /v1/responses` for `codex`.
- Documented the per-provider file layout, runtime-consumed fields, reserved
  fields, router identity policy, `x-acr-agent` override header, and the
  operational meaning of unresolved identity failures.
- Documented Codex setup against the router using `/v1/responses` and the
  client-owned OpenAI credential.
- Documented Hermes setup against the router using `/v1/chat/completions`.
- Updated `config.example.yaml` to reflect the global cheap-side config plus
  provider-specific premium-side layout.
- Marked the OpenAI compatibility documentation phase complete in the plan.

#### Type safety

- Added shared JSON structural aliases for parsed request/config payloads.
- Re-enabled strict Pyright checking for the package.
- Enabled Ruff `ANN401` so `Any` annotations are rejected.
- Reworked package and test annotations to avoid `Any` and unparameterized
  builtin generics.

### Changed

- Passthrough now targets the resolved agent group's own configured upstream
  instead of always using the global premium upstream:
  - `claude_code` passthrough goes to the Anthropic upstream;
  - `codex` passthrough goes to its OpenAI-compatible upstream;
  - `hermes` passthrough goes to its configured Hermes upstream.
- Passthrough still forwards the client request body and headers verbatim.
  Routed cheap-provider calls still use only the tier key resolved from global
  tier config, preserving credential isolation.
- The server route table now includes `/v1/responses`.
- Metrics and request accounting preserve the real request path for
  `/v1/responses` instead of labeling those requests as `/v1/messages`.
- DeepSeek direct serving remains Anthropic-native and unchanged; OpenAI-shaped
  clients are converted only at the edge.
- `acr init` now seeds the global router policy and relies on bootstrap to
  materialize provider files.

### Fixed

- Fixed the cross-provider passthrough bug where non-routed Codex or Hermes
  turns would otherwise be sent to the single global Anthropic upstream.
- Fixed ambiguous identity handling so the router does not guess an upstream
  and accidentally forward a client credential to the wrong provider.
- Removed unrecognized Pyright config keys while keeping strict type checking
  active.
- Resolved the merge conflict with `main` in metrics accounting by keeping the
  strict JSON typing path.

### Tests

- Added conversion tests for Responses request and response handling, including
  malformed input fail-open paths, custom tool calls, reasoning stripping,
  usage mapping, and deterministic serialization.
- Added Responses SSE golden tests for text and function-call streams.
- Added adapter tests for Codex pending-tool extraction, unresolved IDs,
  hosted-tool skipping, and custom tool output routing.
- Added server integration tests for `/v1/responses` routed streaming,
  malformed-body passthrough, path-aware metrics, and credential isolation.
- Added provider-config tests for per-provider YAML assembly, missing and
  malformed provider files, router identity precedence, fail-closed unresolved
  identity, cheap-key rejection, and immutability.
- Added bootstrap tests proving provider config files are create-only,
  idempotent, validate successfully, and contain no cheap-tier keys.
- Added end-to-end compatibility smoke coverage for Claude Code, Hermes, and
  Codex routed and premium turns.
- Verified the full quality gate after the merge-conflict fix:
  `make lint`, `make type`, and `make test` all pass.

## [0.4.0] - 2026-06-17

### Added

#### Hermes Chat Completions routing

- Added `POST /v1/chat/completions` support for Hermes sessions that speak the
  OpenAI Chat Completions wire format.
- Added an OpenAI Chat adapter with Hermes as its default agent group.
- Added deterministic Chat-to-Anthropic request conversion for routed turns,
  including:
  - system messages to Anthropic top-level `system`;
  - user and assistant text messages to Anthropic text content blocks;
  - assistant `tool_calls[]` to Anthropic `tool_use` blocks;
  - `role: "tool"` messages to Anthropic `tool_result` blocks;
  - OpenAI function tool schemas to Anthropic `input_schema`;
  - OpenAI `tool_choice` mapping where supported;
  - `max_tokens` and `max_completion_tokens` handling.
- Added Anthropic-to-Chat response synthesis for non-streaming routed responses,
  including text content, tool calls, finish reasons, usage mapping, and
  deterministic IDs.
- Added Chat Completions SSE synthesis for streamed routed responses, including
  role deltas, text deltas, tool-call deltas, final finish chunks, and
  `[DONE]`.
- Added Hermes pending-tool extraction from the last run of OpenAI tool-result
  messages, resolving `tool_call_id` back to the earlier assistant tool-call
  name.
- Added `/v1/chat/completions` route handling through the existing adapter
  abstraction, decision layer, routed-call core, and passthrough fallback.

#### Per-agent routing foundation

- Added the `ClientAdapter` protocol and adapter registry so endpoint-specific
  wire formats can share the same routing core.
- Added the Anthropic Messages identity adapter for the existing
  `/v1/messages` Claude Code path.
- Added per-agent default tool maps and premium-tool lists for:
  - `claude_code`;
  - `hermes`;
  - `codex`.
- Moved Claude Code default tool configuration out of the wizard and into a
  data-only defaults module.
- Added per-agent `agents:` config support while preserving old flat config
  compatibility through a compatibility shim.
- Added per-agent `tier_for_tools(..., group=...)` behavior so the same tool
  name can route differently for Claude Code, Hermes, and Codex.
- Added per-agent premium-tool escalation so a premium tool in one group does
  not trigger escalation for another group.
- Added a `structured` tier for Hermes structured-operation tools such as
  `write_file`, `skill_manage`, and `cronjob`.
- Relaxed tier-name validation so custom tier names from config are accepted
  instead of being limited to the original `fast`, `code`, `crud`, and
  `premium` names.

#### Wizard and example config

- Updated `acr init` output to seed per-agent config data.
- Updated generated config tests so wizard output round-trips through route
  loading and routing decisions.
- Updated example config with all agent groups, tool maps, premium tools, and
  the new `structured` tier.

### Changed

- The routed serving path now converts inbound client formats to Anthropic
  canonical only when a cheap route is selected. Passthrough remains a verbatim
  relay in the caller's original format.
- `_try_route` and `routed_call` now operate through the adapter interface while
  preserving the existing Claude Code behavior for `/v1/messages`.
- Response-side premium-tool escalation now uses the active agent group's
  premium-tool list.
- Tier selection now reads tools from the active agent group's tool map and
  falls back to built-in defaults when config is missing or malformed.
- Existing flat configs without `agents:` keep routing as Claude Code configs
  through the compatibility shim.
- Metrics recording now receives the real request path for Chat Completions
  traffic.

### Fixed

- Fixed legacy config compatibility so old `tools:` plus
  `settings.premium_tools` configs continue to route Claude Code tool-result
  turns without user edits.
- Fixed cross-agent premium escalation so Codex-only premium tools such as
  `apply_patch` do not force Hermes or Claude Code turns to passthrough unless
  those groups mark the tool premium.
- Fixed unknown-tool handling to safely select the premium tier.
- Fixed mixed-tool batches so the highest-precedence tier wins.
- Fixed malformed or missing agent tool config paths to fail open to defaults
  instead of raising into serving.

### Tests

- Added tests for all default agent tool maps and premium-tool lists.
- Added tier-resolution tests for Claude Code, Hermes, Codex, unknown tools,
  mixed batches, custom tier names, and the old flat-config compatibility shim.
- Added routed-call tests proving per-agent premium escalation behavior.
- Added Chat conversion tests for tool calls, tool results, system messages,
  text-only turns, tool schemas, malformed arguments, immutability, and
  byte-stable Anthropic request construction.
- Added Chat SSE golden tests for text and tool-call streams.
- Added Hermes Chat server integration tests for routed DeepSeek direct calls,
  streamed responses, premium-tool passthrough, malformed-body passthrough, and
  request-path metrics.

## [0.3.0] - 2026-06-14

### Added

- Added the Phase 0 OpenAI compatibility findings document.
- Documented Codex's OpenAI Responses API wire mode and the decision to support
  `/v1/responses`.
- Documented that standard Codex/OpenAI provider turns send full `input[]`
  conversation state, making routing feasible without server-side
  `previous_response_id` state.
- Documented Hermes' multi-wire provider behavior:
  Chat Completions by default, Responses for OpenAI-Codex/XAI style providers,
  and Anthropic Messages where applicable.
- Captured the Hermes tool-to-tier map and premium-tool expectations from the
  existing Hermes routing config.
- Captured the Codex tool vocabulary and initial tier mapping for shell,
  stdin, planning, patch, plugin, and agent-spawn style tools.
- Added provider-routing plan notes for the later split between global cheap
  tier config and per-provider premium-side YAML files.

### Changed

- Updated the OpenAI compatibility roadmap with the edge-conversion design:
  inbound client wire format to Anthropic canonical, unchanged routed core,
  DeepSeek direct serving, then response conversion back to the caller's wire
  format.
- Recorded the invariant that passthrough must stay verbatim and routed
  DeepSeek requests must remain deterministic to preserve prefix-cache
  stability.
- Clarified that OpenAI compatibility would be implemented phase-by-phase:
  adapter seam, per-agent tool config, Hermes Chat Completions, Codex
  Responses, per-agent passthrough upstreams, docs, and provider-specific YAML.

### Tests

- No runtime behavior changed in this release; the work was investigation and
  planning for the OpenAI/Hermes compatibility implementation.

## [0.2.0] - 2026-06-14

### Added

- Live dashboard at `/dashboard` with per-request table, per-agent grouping,
  per-session breakdown, and auto-refresh. Removed `tier` and `route` columns;
  split cache into `cache hit` (read) and `cache miss` (write) columns.
- Model-column prefix stripping (shows `deepseek-v4-flash` instead of
  `deepseek/deepseek-v4-flash`).
- Agent identification from `User-Agent` header with friendly display labels
  (🖥️ Claude-Code CLI, 💻 Claude Desktop, 🔧 API).
- Session fingerprinting: SHA-256 hash of the first system message content,
  first 12 hex chars, shown in the dashboard.
- Provider identification from model string: maps arbitrary model IDs to a
  dashboard-friendly provider label (anthropic, openai, deepseek, google, aws,
  azure, meta, mistral, cohere, groq, fireworks, perplexity, together, unknown).
- Metrics persistence: `bootstrap()` replays `savings.jsonl` on proxy startup to
  restore routed-token counters and recent-request history, so the dashboard
  survives restarts.
- Cache-Control `no-cache, no-store, must-revalidate` headers on `/dashboard` to
  prevent stale cache after HTML updates.

### Fixed

- Dashboard JavaScript syntax error caused by escaped quotes in the `esc()`
  sanitizer function — rewritten with clean quoting and validated via
  `node --check`.

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

[0.5.0]: https://github.com/maheshkokare/ai-calls-router/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/maheshkokare/ai-calls-router/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/maheshkokare/ai-calls-router/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/maheshkokare/ai-calls-router/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/maheshkokare/ai-calls-router/releases/tag/v0.1.0
