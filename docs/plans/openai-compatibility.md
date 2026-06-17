# OpenAI Compatibility Plan

## Phase 0 Investigation

# Plan: OpenAI-call compatibility (Codex + Hermes) with per-agent tool config

Status: APPROVED FOR EXECUTION — implement phase by phase, in order.
Owner: routing layer.
Audience: any engineer or LLM agent executing a single phase cold, with no prior
context from the design conversation. Everything needed is in this document plus
the cited `file:line` references.

---

## 0. How to work in this repo (read first)

- Python target: 3.12 syntax. The local venv interpreter is `.venv/bin/python`
  (CPython 3.13). Always invoke tests through the venv, never bare `python`.
- Prefer Makefile targets. Relevant ones (run `make help` for the full list):
  - `make test` — run the suite.
  - `make lint` — `ruff check`.
  - `make format` — `ruff format` source + tests.
  - `make type` — pyright.
  - `make coverage` — tests with coverage; **fails under 95%**.
  - `make build` — sdist + wheel.
- Direct equivalents if a target is missing: `.venv/bin/python -m pytest`,
  `ruff check`, `ruff format`, `pyright`.
- Coding standards enforced in this repo (do not deviate):
  - Google-format docstrings on every module, class, function, method.
  - A 2–4 sentence module docstring at the top of every new file describing it.
  - Built-in generics (`list[str]`, `dict[str, Any]`); `typing` only for
    `Protocol`, `Callable`, `Literal`, etc.
  - **Immutability**: never mutate an input body/message/dict. Build new objects;
    share unchanged sub-objects; return the original object by identity on no-op
    (see `_prepare_routed_body` in `routing/engine.py` for the build-new pattern
    and the no-op passthrough in `routing/compression.py` for return-by-identity).
  - No nested functions; helpers are module-level.
  - A public function with more than two caller-supplied positional arguments
    makes the rest keyword-only with `*` (see `routed_call` in
    `ai_calls_router/routing/engine.py`).
  - One significant public class per file; `ClassName` -> `class_name.py`.
  - Complexity budget: keep new functions at radon grade A/B.
- Tests are spec-derived and adversarial (see `tests/unit/test_compression.py` for
  the house style): derive expected values from the contract, cover boundaries and
  error paths, never mirror the implementation, never snapshot-match.

### Non-negotiable serving invariants (every phase must preserve all of these)

1. **Fail open.** Any error in decision, conversion, or the routed call resolves
   to passthrough. Routing must never break a turn. Pattern: catch broadly,
   log at WARNING, return `None`/passthrough (see `_try_route`
   `ai_calls_router/proxy/server.py:255` and `routed_call`
   `ai_calls_router/routing/engine.py:318`).
2. **Credential isolation.** The client's premium credential (Anthropic OAuth /
   OpenAI key) must NEVER reach the routed cheap provider. The routed call uses
   only the tier key resolved from `key_env`/`env_file`
   (`resolve_api_key` `ai_calls_router/routing/decide.py:261`). Passthrough
   forwards the client's own headers verbatim to the client's own upstream.
3. **Accounting never raises into serving.** Savings recording is best-effort and
   must not propagate exceptions into the response path
   (`ai_calls_router/accounting/savings.py`).
4. **DeepSeek cache byte-stability.** Anything that builds the body sent to a
   native-Anthropic (DeepSeek) endpoint must be a deterministic pure function of
   the inbound request, so the cacheable prefix is byte-identical across turns.
   This is the entire reason routed DeepSeek calls bypass LiteLLM and compression
   today (`_serve_via_direct` `ai_calls_router/routing/engine.py:209`). Violating
   it silently destroys the prefix cache — the highest-value property in the
   system.
5. **No hardcoded secrets.** Keys come from env / env_file only.

---

## 1. Goal

Make the router serve tool-result turns for three agent families, not just one:

| Agent group  | Members (collapsed into one group) | Wire format(s)                          | Inbound endpoint(s)        |
| ------------ | ---------------------------------- | --------------------------------------- | -------------------------- |
| `claude_code`| Claude Code CLI + desktop          | Anthropic Messages                      | `POST /v1/messages`        |
| `hermes`     | Hermes CLI + desktop               | **Multi-wire, per-provider** (Phase 0): Chat Completions (default) · Responses · Anthropic Messages | `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages` |
| `codex`      | Codex CLI + desktop                | OpenAI Responses API (confirmed; Phase 0) | `POST /v1/responses`       |

Hard requirements stated by the product owner:

- **R1.** DeepSeek routing always uses Anthropic style, regardless of inbound
  format, so the DeepSeek prefix cache stays hot (invariant 4).
- **R2.** Reuse the existing tool-router decision logic for Hermes (the
  `headroom-tool-router` port already living in `routing/decide.py`).
- **R3.** Codex integration must be investigated and grounded before coding
  (Phase 0) — its wire protocol and statefulness decide feasibility.
- **R4.** Tool→tier configuration lives in YAML and is consumed by the code; it is
  not hardcoded in the wizard. Defaults are seeded per agent group from a code
  data-module, written into config by the wizard, and read from config at
  runtime.
- **R5.** Default tools are configured per agent group: `codex`, `claude_code`,
  `hermes` (CLI and desktop of each share one group).

---

## 2. Key architectural fact (do not re-derive)

The internal canonical representation is **already Anthropic Messages format**.
The serving pipeline operates entirely on Anthropic bodies; it converts to OpenAI
only for the LiteLLM provider path and converts the response back
(`ai_calls_router/_lib/conversion.py:169`, `:218`). The DeepSeek direct path
sends Anthropic-native verbatim (`ai_calls_router/routing/direct.py`); it bypasses
the LiteLLM conversion and applies no compression, which is what keeps the
cacheable prefix byte-identical across turns
(`_serve_via_direct` `ai_calls_router/routing/engine.py:209`).

Therefore the design is **edge conversion, unchanged core**:

```
inbound (any format) --ingress adapter--> Anthropic canonical
        --> existing decide + routed_call + DeepSeek-direct (UNCHANGED)
        --> Anthropic response --egress adapter--> caller's format (any)
```

Only the routed (DeepSeek) path converts. Passthrough stays a verbatim byte relay
because the client's format already equals its own upstream's format.

### Target flow

```mermaid
flowchart TD
    A[Inbound request] --> B{Endpoint -> format; header/endpoint -> agent group}
    B -->|/v1/messages · claude_code| C[AnthropicMessagesAdapter]
    B -->|/v1/chat/completions · hermes| D[OpenAIChatAdapter]
    B -->|/v1/responses · codex| E[OpenAIResponsesAdapter]
    C --> F[extract_pending_tools]
    D --> F
    E --> F
    F --> G{tier_for_tools using per-agent tool map}
    G -->|premium / unmapped| H[Passthrough verbatim to agent's own upstream]
    G -->|cheap tier| I[adapter.to_anthropic_request — deterministic canonical]
    I --> J[routed_call -> _serve_via_direct: DeepSeek native Anthropic, no compression]
    J --> K{escalates? using per-agent premium_tools}
    K -->|yes| H
    K -->|no| L[adapter.to_client_response / to_client_sse: Anthropic -> client format]
    L --> M[Response to agent in its native format]
```

---

## 3. File map (what each phase creates or touches)

New:
- `ai_calls_router/routing/agent_defaults.py` — per-agent default tool→tier maps
  + premium lists (data module; Phase 2).
- `ai_calls_router/routing/adapters/__init__.py` — adapter registry + resolution
  (Phase 1).
- `ai_calls_router/routing/adapters/base.py` — `ClientAdapter` Protocol (Phase 1).
- `ai_calls_router/routing/adapters/anthropic_messages.py` — Phase 1.
- `ai_calls_router/routing/adapters/openai_chat.py` — Phase 3.
- `ai_calls_router/routing/adapters/openai_responses.py` — Phase 4.
- `ai_calls_router/_lib/openai_inbound.py` — OpenAI→Anthropic request + tool-def
  conversion, and Anthropic→OpenAI response conversion (Phase 3/4). Keep
  `_lib/conversion.py` focused on the existing LiteLLM direction.
- `ai_calls_router/routing/synthesis_openai.py` — OpenAI chat + Responses SSE
  renderers (Phase 3/4).
- `ai_calls_router/routing/provider_config.py` — assemble the per-provider YAML
  files into the canonical `agents:` routes dict + identity resolution (Phase 7).
- `ai_calls_router/ops/bootstrap.py` — materialize missing per-provider YAMLs from
  `agent_defaults.py` before serving; idempotent (Phase 7).
- Per-provider config templates on disk under `~/.ai-calls-router/config/`:
  `claude-code.yaml`, `codex.yaml`, `hermes.yaml` (Phase 7; written by bootstrap,
  not committed source).
- Tests: `tests/unit/test_agent_defaults.py`, `tests/unit/test_adapters_*.py`,
  `tests/unit/test_openai_inbound.py`, `tests/unit/test_synthesis_openai.py`,
  `tests/unit/test_provider_config.py`, `tests/unit/test_bootstrap.py` (Phase 7),
  plus integration cases in `tests/` mirroring `test_routed_call.py`.

Changed:
- `ai_calls_router/proxy/server.py` — add routes; thread adapter + agent group
  through `_try_route`; per-agent passthrough upstream.
- `ai_calls_router/routing/decide.py` — `tier_for_tools(..., group)`; per-agent
  tool-map selection; `agents:` schema + flat-config compat shim.
- `ai_calls_router/routing/engine.py` — `escalates(..., group)` reads per-agent
  premium tools; `routed_call` threads `group`.
- `ai_calls_router/ops/wizard.py` — emit `agents:` sections from
  `agent_defaults.py` instead of the hardcoded `DEFAULT_TOOLS`/`PREMIUM_TOOLS`.
- `config.example.yaml` — replace flat `tools:`/`premium_tools:` with `agents:`.
- `ai_calls_router/_lib/config.py` — add `provider_config_dir()`,
  `provider_config_path(group)`, `KNOWN_GROUPS` (Phase 7).
- `ai_calls_router/proxy/server.py` — call `bootstrap.ensure_provider_configs()` in
  `_lifespan` before serving; load via `provider_config.assemble_routes()`; resolve
  the agent group per request and fail closed on unattributable identity (Phase 7).
- `ai_calls_router/ops/wizard.py` — `acr init` also runs bootstrap and seeds the
  global `router:` block (Phase 7).
- `docs/` / `README` — document the new endpoints, config, and per-provider layout.

---

## 4. Config schema change

### Before (current, flat — single agent assumed)

`config.example.yaml:85` `tools:` is one flat map; `settings.premium_tools`
(`:41`) is one flat list; `server.upstream` (`:15`) is the single passthrough
target.

### After (per agent group — logical schema)

This `agents:` block is the **in-memory canonical shape** the decision core
consumes (`routes["agents"][group]`). Phase 2 ships it as one block in one
`config.yaml`; Phase 7 repackages the same shape onto three per-provider files on
disk and merges them back into this identical dict (see §4.1 and Phase 7). The
decision core is indifferent to which packaging produced the dict.

```yaml
agents:
  claude_code:
    upstream: https://api.anthropic.com
    premium: { provider: anthropic }
    tools:                       # Claude Code tool vocabulary (current map)
      Bash: fast
      Read: code
      Grep: code
      Glob: code
      TodoWrite: crud
      Edit: premium
      Write: premium
      # ...full map from ops/wizard.py DEFAULT_TOOLS
    premium_tools: [Edit, Write, MultiEdit, NotebookEdit, Task, ExitPlanMode, AskUserQuestion]

  hermes:
    # Premium = the Hermes session model (provider openai-codex, via Hermes
    # OAuth) — NOT a fixed anthropic endpoint. Resolved per Phase 0 findings.
    upstream: <hermes session-model upstream>    # see phase0-findings.md §6.2
    premium: { provider: openai-codex }
    tools:                       # REAL Hermes vocabulary — routes.yaml (Phase 0)
      terminal: fast
      process: fast
      read_file: code
      search_files: code
      execute_code: code
      skill_view: code
      todo: crud
      memory: crud
      session_search: crud
      skills_list: crud
      write_file: structured     # NOTE: 'structured' tier — see below
      skill_manage: structured
      cronjob: structured
      patch: premium
      clarify: premium
      delegate_task: premium
      browser_vision: premium
      browser_*: premium
    premium_tools: [patch, clarify, delegate_task]

  codex:
    upstream: https://api.openai.com             # default OpenAI provider (Phase 0)
    premium: { provider: openai }
    tools:                       # REAL Codex surface (thin) — Phase 0 §5
      exec_command: fast
      local_shell: fast
      shell: fast
      shell_command: fast
      write_stdin: fast
      update_plan: crud
      get_context_remaining: crud
      apply_patch: premium       # custom/freeform tool; see Phase 4 gaps
      spawn_agent: premium
      request_user_input: premium
      request_plugin_install: premium
      # No native code tier: Codex reads/searches via the shell. A 'code' tier
      # only appears if a filesystem MCP is wired in.
    premium_tools: [apply_patch, spawn_agent, request_user_input, request_plugin_install]
```

`settings:` (env_file, tier_precedence, compression,
escalate_on_premium_tools) stay global and unchanged. **`tiers:` stays global
but must gain a `structured` tier** for the `hermes` group (`write_file`,
`skill_manage`, `cronjob` reference it) — add it to the config `tiers:` schema or
remap those three onto `code`/`crud`. The per-agent config and tier validation
must therefore accept tier names beyond `fast`/`code`/`crud`.

### Compatibility shim (mandatory)

If `agents:` is absent, synthesize a single `claude_code` group from the existing
top-level `tools:`, `settings.premium_tools`, and `server.upstream`. Existing
configs must keep working with zero edits. Add a test that an old flat config
routes identically before and after this change.

### 4.1 On-disk layout (Phase 7 — provider-specific YAML files)

Phase 7 splits the `agents:` block onto one self-contained file per provider
family, keeping the cheap side global. The decision core still consumes the §4
canonical dict; only the *source on disk* changes.

```
~/.ai-calls-router/
  config.yaml            # global: server:, settings:, tiers:, router:  (cheap side + identity policy)
  config/
    claude-code.yaml     # agents.claude_code payload only
    codex.yaml           # agents.codex payload only
    hermes.yaml          # agents.hermes payload only
```

Three packaging modes coexist and all converge to one canonical dict (KISS/DRY —
no fork of the decision core):

- (a) **legacy flat single file** → Phase 2 compat shim → one `claude_code` group;
- (b) **single file with an `agents:` block** → Phase 2 schema verbatim;
- (c) **three per-provider files + `router:` map** → Phase 7, the target layout.

`provider_config.assemble_routes()` (Phase 7) reads whichever mode is present and
returns the identical `{"server", "settings", "tiers", "agents"}` dict.

Hard rules for the per-provider files (enforced by bootstrap validation):

- **The cheap-provider credential lives only in global `tiers.*.key_env`.** A
  per-provider file MUST NOT carry a cheap `key_env` (invariants 2 and 5). This
  keeps the DeepSeek key single-sourced and unreachable from the premium side.
- Each per-provider file owns its **premium side only**: `upstream`, `auth`,
  `wire`, model defaults, `tools`, `premium_tools`, `tool_choice`, `reasoning`,
  `fallback`.
- `auth` names a credential *source*, never a secret:
  `auth: { mode: oauth_passthrough }` (forward the client's own Authorization
  header verbatim — the default; invariant 2) or
  `auth: { mode: api_key_env, key_env: SOME_ENV }` (names an env var only).

Per-provider file schema (`claude-code.yaml` shown):

```yaml
group: claude_code
upstream: https://api.anthropic.com
auth: { mode: oauth_passthrough }     # forward client credential verbatim; never store a secret
wire: anthropic_messages              # anthropic_messages | openai_chat | openai_responses
endpoints: [/v1/messages]             # endpoints this family may arrive on
model_defaults: {}                    # optional premium model hints (passthrough is verbatim; informational)
tool_choice: passthrough              # provider tool-choice policy (informational in v1)
reasoning: strip                      # strip | preserve — codex/hermes-codex strip Reasoning items (invariant 4)
tools:                                # == agents.claude_code.tools (§4); seeded from agent_defaults
  Bash: fast
  Read: code
  # ...full map from agent_defaults.AGENT_DEFAULT_TOOLS["claude_code"]
premium_tools: [Edit, Write, MultiEdit, NotebookEdit, Task, ExitPlanMode, AskUserQuestion]
fallback: passthrough                 # per-family serving fallback when not cheap-routed
```

Router map (global `config.yaml` `router:` block) — drives §5 resolution:

```yaml
router:
  endpoint_defaults:
    /v1/messages: claude_code
    /v1/chat/completions: hermes
    /v1/responses: codex
  user_agent_map:                     # desktop-shim identity -> family (CLI and desktop share one group)
    - { contains: claude,  group: claude_code }
    - { contains: codex,   group: codex }
    - { contains: hermes,  group: hermes }
  fallback: null                      # null => fail closed on ambiguity; a group name => explicit passthrough policy
```

---

## 5. Agent group vs adapter (important distinction)

- **Adapter** = wire-format handler, chosen by **endpoint**. Converts request and
  response shapes. One adapter per format.
- **Agent group** = identity used to pick the tool map / premium list / upstream.
  Resolved by `resolve_agent_group(*, path, headers, routes, adapter_default)` in
  precedence order (Phase 7 finalizes this against the `router:` block, §4.1):
  1. `x-acr-agent` request header if present and valid (`codex|claude_code|hermes`).
  2. else `router.user_agent_map` — first `contains`-match (case-insensitive)
     against the inbound `User-Agent` (the desktop-shim signal: CLI and desktop of
     one family share one group, so this maps the shim's UA onto the family, never
     to a new group). Skipped when no `router:` block is present.
  3. else `router.endpoint_defaults[path]` when a `router:` block is present
     (`/v1/messages`→`claude_code`, `/v1/chat/completions`→`hermes`,
     `/v1/responses`→`codex`).
  4. else `router.fallback` if set to a known group (an explicit operator opt-in to
     passthrough for a named group).
  5. else `adapter_default` — the adapter's own `default_agent_group`. This is the
     SINGLE source of the endpoint→group map for the legacy flat file and the
     single-file `agents:` block (modes a/b), which carry no `router:` block: those
     deployments resolve all three known endpoints exactly as before and NEVER
     return `None`. `resolve_agent_group` returns `None` ONLY when a `router:` block
     IS present, `fallback` is `null`, and steps 1–3 all miss — i.e. the operator
     opted into router-based routing and the request is genuinely ambiguous. On
     `None` the handler rejects with `400` rather than guess an upstream (invariant
     2: forwarding the client's own credential to a guessed upstream risks a
     cross-provider leak). Fail-closed is an *attribution-layer* decision and does
     not violate invariant 1 (fail open), which governs the *serving* layer: once a
     group resolves, any later error still falls back to that group's passthrough.
     The endpoint→group map lives in exactly two places by design —
     `adapter.default_agent_group` (modes a/b) and `router.endpoint_defaults`
     (mode c) — and is never duplicated a third time inside `router_map`. See Phase 7.

They are NOT 1:1 — the override is load-bearing, because Hermes is multi-wire
(Phase 0). A Hermes session can arrive on ANY of the three endpoints depending on
its provider: `/v1/chat/completions`, `/v1/responses`, or `/v1/messages`. On
`/v1/responses` both `hermes` and `codex` share the `OpenAIResponsesAdapter` and
must be told apart by `x-acr-agent` to pick the right tool map (Hermes'
`patch`/`terminal`/... vs Codex's `apply_patch`/`shell`/...); likewise a Hermes
Chat session and a future Chat client share `/v1/chat/completions`. So the
endpoint default must be overridable by header. Keep adapter and group decoupled.
(Codex itself is Responses-only; only Hermes spans all three wires.)

---

## 6. Adapter contract (Phase 1 defines; Phases 3–4 implement)

`ai_calls_router/routing/adapters/base.py`:

```python
from collections.abc import Iterator
from typing import Any, Protocol


class ClientAdapter(Protocol):
    """Wire-format handler for one inbound API shape.

    Converts an inbound request to the Anthropic canonical the serving core
    expects, extracts the pending tool names the routing decision needs, and
    converts the Anthropic response back to the caller's native format. All
    methods are pure functions of their arguments and never mutate inputs.
    """

    default_agent_group: str

    def extract_pending_tools(self, body: dict[str, Any]) -> list[str]:
        """Return ordered, deduplicated names of tools whose results this
        request is processing; [] for a turn opener or on shape surprises."""

    def to_anthropic_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Convert the inbound request to an Anthropic Messages body,
        deterministically (see invariant 4). Never mutates `body`."""

    def to_client_response(self, anthropic_body: dict[str, Any]) -> dict[str, Any]:
        """Convert an Anthropic response body to the caller's native
        non-streaming response object."""

    def to_client_sse(self, anthropic_body: dict[str, Any]) -> Iterator[bytes]:
        """Render an Anthropic response body as the caller's native SSE stream."""
```

The `AnthropicMessagesAdapter` is mostly identity: `extract_pending_tools`
delegates to today's `pending_tool_names`, `to_anthropic_request` returns the
body unchanged, `to_client_response` returns the Anthropic body, `to_client_sse`
delegates to the existing `synthesis.synthesize_sse`.

---

## Phase 0 — Investigation (blocking; no code)

**STATUS: COMPLETE — results in `docs/plans/phase0-findings.md`.** Resolved: Codex
is Responses-only (Phase 4 IN); Codex sends full `input[]` each turn (routable);
Hermes tool map/premium/tiers grounded in the real `routes.yaml`; Hermes inbound
wire is **per-provider multi-wire** (`hermes-agent` `runtime_provider.py:240-341`)
— Chat Completions (default), Responses (openai-codex/xai), or Anthropic Messages
(anthropic/minimax). **Both Phase 3 and Phase 4 have real Hermes consumers; both
ship.** No open items remain (bedrock/codex_app_server out of scope, passthrough).

**Objective.** Replace every "confirm in Phase 0" placeholder with verified fact,
and settle whether Phase 4 (Responses adapter) is needed.

**Steps.**
1. OpenAI Chat Completions tool-call contract — confirm the exact shapes for:
   assistant message `tool_calls[].id` / `.function.name` / `.function.arguments`
   (JSON string); `role:"tool"` result message `tool_call_id` + `content`; the
   `tools[]` function-definition schema; streaming `chat.completion.chunk` deltas
   for tool calls. Source: OpenAI docs via Context7 (`/openai/openai-openapi` or
   the platform docs), cross-checked against the existing Hermes tool router.
2. OpenAI Responses API contract — confirm: `POST /v1/responses` request body;
   `input[]` item types `function_call` (`name`, `arguments`, `call_id`) and
   `function_call_output` (`call_id`, `output`); Codex built-in tool item types
   (e.g. `local_shell`/`local_shell_call`, `apply_patch`) and their output items;
   the streaming event sequence (`response.created`,
   `response.output_item.added`, `response.function_call_arguments.delta`,
   `response.completed`, etc.).
3. **Codex statefulness (decision-critical).** Determine whether Codex sends the
   full `input[]` history every turn (`store:false`) or relies on
   `previous_response_id` server-side state. Inspect the Codex source / config.
   - If full-input: proceed with Phase 4 as written.
   - If server-state-only: the router cannot see history → cannot extract pending
     tools → Codex can only passthrough. Document the required Codex config to
     force full-input mode, or descope Codex routing to passthrough-only.
4. **Codex wire mode (scope lever).** Determine whether the user will point Codex
   at the router via the Responses API or via a Chat-Completions model-provider
   (`wire_api = "chat"`). If Chat, Phase 4 is dropped; Codex reuses the
   `OpenAIChatAdapter` with `x-acr-agent: codex`.
5. Extract the Hermes tool→tier map and premium list from the existing Hermes
   tool router (`~/.hermes/plugins/tool-router/routes.yaml`); record the Hermes
   premium upstream + provider. **DONE:** distinct vocab (`terminal`/`read_file`/
   `patch`/...), adds a `structured` tier, premium=`[patch, clarify, delegate_task]`,
   provider `openai-codex`, premium upstream = Hermes session model.
6. Enumerate the Codex tool vocabulary and assign each tool a tier + premium
   flag. **DONE:** `shell` family→fast, `apply_patch`→premium, multi-agent/
   user-facing/plugin-install→premium, `update_plan`/`get_context_remaining`→crud.
   NOTE: Codex has NO native code-introspection tools (`read_file`/`grep`/`glob`
   are not built in — it reads/searches via the shell), so the `code` tier is
   empty for codex unless a filesystem MCP is wired in.

**Constraints.** Cite a source (doc URL or repo `file:line`) for every shape and
every tool-name decision. No assumptions carried into later phases.

**Verification / done-when.**
- A short `docs/plans/phase0-findings.md` exists containing: verified Chat +
  Responses tool-call schemas with sources; the Codex statefulness answer with
  source; the Codex wire-mode decision; the Hermes tool map + upstream; the Codex
  tool map. Every placeholder in this plan is resolvable from it.
- Decision recorded: Phase 4 IN or OUT.

---

## Phase 1 — Inbound adapter abstraction (no behavior change)

**Objective.** Introduce the adapter seam and route the existing Anthropic path
through it, with zero behavioral change. Pure refactor.

**Preconditions.** None (can start in parallel with Phase 0).

**Steps.**
1. Create `adapters/base.py` with the `ClientAdapter` Protocol (§6).
2. Create `adapters/anthropic_messages.py` implementing the adapter by delegating
   to existing functions (`pending_tool_names`, identity request,
   `synthesis.synthesize_sse`).
3. Create `adapters/__init__.py` with:
   - `adapter_for_path(path: str) -> ClientAdapter | None`
   - `resolve_agent_group(default_group: str, headers) -> str` (§5). NOTE: this
     header-only form is the Phase 1/5 shape; Phase 7 replaces it with the
     router-aware `provider_config.resolve_agent_group(*, path, headers, routes,
     adapter_default)` and deletes this one.
4. Refactor `proxy/server.py`:
   - Generalize `_try_route(body_bytes)` to
     `_try_route(body_bytes, *, adapter, group)`; replace the direct
     `pending_tool_names` call (`server.py:277`) with
     `adapter.extract_pending_tools(body)`; build the Anthropic body via
     `adapter.to_anthropic_request(body)` before `routed_call`; convert the
     result via `adapter.to_client_response` / `to_client_sse` (`server.py:321-331`).
   - Keep `/v1/messages` wired to `AnthropicMessagesAdapter` +
     `claude_code` group. No new endpoints yet.

**Constraints.** Preserve all five serving invariants. Do not change routing
decisions, savings, or SSE bytes for the Anthropic path. Immutability:
`to_anthropic_request` for Anthropic returns the same object (identity).

**Verification / done-when.**
- `make test` — the full existing suite passes unchanged (no test edits needed
  beyond import wiring). 95% coverage holds.
- `make lint`, `make type` clean; new functions radon A/B.
- Mutation check: temporarily make `extract_pending_tools` return `[]` and
  confirm an existing routing test fails (seam is load-bearing), then revert.

---

## Phase 2 — Per-agent tool config

**Objective.** Move tool defaults out of `wizard.py` into a data module, add the
`agents:` schema, make the decision read per-agent maps, keep old configs working.

> **Canonical-merge contract (load-bearing for Phase 7).** The `agents:` block
> defined here is the single in-memory canonical shape `routes["agents"][group]`.
> Phase 7 changes only *where the bytes live on disk* (three per-provider files +
> a `router:` map), then merges them back into a dict byte-equivalent to this one
> before `decide`/`engine` see it. Do not bake "one file" assumptions into the
> decision core — read only from `routes["agents"][group]`, never from a file path.

**Preconditions.** Phase 1 merged. Phase 0 Hermes/Codex tool maps available (for
seeding defaults; can stub if Phase 0 still running, but real maps required before
Phases 3/4).

**Steps.**
1. Create `routing/agent_defaults.py` (seed from Phase 0 / §4 config example):
   ```python
   AGENT_DEFAULT_TOOLS: dict[str, dict[str, str]] = {
       "claude_code": { ... },   # move ops/wizard.py:84 DEFAULT_TOOLS here verbatim
       "hermes": { ... },        # routes.yaml map: terminal→fast, read_file→code,
                                 #   write_file/skill_manage/cronjob→structured,
                                 #   patch/clarify/delegate_task→premium, browser_*→premium
       "codex": { ... },         # shell family→fast, apply_patch→premium,
                                 #   update_plan/get_context_remaining→crud (no code tier)
   }
   AGENT_DEFAULT_PREMIUM_TOOLS: dict[str, list[str]] = {
       "claude_code": [ ... ],                        # ops/wizard.py PREMIUM_TOOLS
       "hermes": ["patch", "clarify", "delegate_task"],
       "codex": ["apply_patch", "spawn_agent", "request_user_input", "request_plugin_install"],
   }
   ```
   **`structured` tier:** the `hermes` map references a `structured` tier absent
   from the current `tiers:`. Phase 2 must either add `structured` to the config
   `tiers:` schema (and seed it in the wizard) or remap those three tools onto an
   existing tier. Tier-name validation must not hardcode `fast`/`code`/`crud`.
2. `routing/decide.py`:
   - Add `agent_tools(routes, group) -> dict[str, str]` returning
     `routes["agents"][group]["tools"]`, falling back to
     `AGENT_DEFAULT_TOOLS[group]`.
   - Add the **compat shim** (§4): when `routes` has no `agents` key, synthesize a
     `claude_code` group from top-level `tools` / `settings.premium_tools` /
     `server.upstream`.
   - Change `tier_for_tools(names, routes)` ->
     `tier_for_tools(names, routes, *, group)`, using `agent_tools(routes, group)`
     instead of `routes.get("tools")` (`decide.py:229`). `tier_precedence` stays
     global.
3. `routing/engine.py`:
   - Change `escalates(response_body, settings)` ->
     `escalates(response_body, settings, *, premium_tools)` (or thread `group`),
     reading the per-agent premium list instead of `settings.premium_tools`
     (`engine.py:149`). The DeepSeek response names the agent's own tools, so the
     premium check must use the agent's list.
   - Thread `group` through `routed_call` to `escalates`.
4. `proxy/server.py`: pass `group` into `tier_for_tools` and `routed_call`.
5. `ops/wizard.py`: replace `DEFAULT_TOOLS`/`PREMIUM_TOOLS` usage
   (`wizard.py:84`, `:106`, `:226`, `:222`) with an `agents:` block built from
   `agent_defaults.py`. Add per-agent `upstream`/`premium`.
6. `config.example.yaml`: replace the flat `tools:` and `settings.premium_tools`
   with the `agents:` block (§4). Keep `tiers:`/`settings:` global.

**Constraints.** Immutability and fail-open preserved. The compat shim must make a
pre-existing flat config route byte-identically to before. `agent_defaults.py`
holds data only — no IO, no logic. Per-agent premium-tool list drives the
escalation guard correctly (Claude `Edit`, Codex `apply_patch`, etc.).

**Verification / done-when.**
- New `tests/unit/test_agent_defaults.py`: every group has a non-empty tool map;
  every tier referenced exists in `tiers`; every premium tool maps to `premium`.
- `decide.py` tests: `tier_for_tools` with `group="codex"` resolves Codex names;
  with `group="claude_code"` matches today's behavior; unknown tool → premium;
  mixed batch → highest precedence; **compat shim**: a config lacking `agents:`
  resolves the same tier for the same Claude tools as before.
- `engine.py` tests: a routed response calling a group's premium tool escalates
  (returns `None`); calling another group's premium-but-not-this-group tool does
  not.
- `wizard.py` tests (`tests/unit/test_wizard.py`): generated config contains all
  three `agents:` groups with tools + premium_tools + upstream; round-trips
  through `decide.load_routes`.
- `make test`/`lint`/`type` clean; coverage ≥ 95%.

---

## Phase 3 — OpenAI Chat Completions adapter (Hermes)  [IN per Phase 0]

**Objective.** Serve `POST /v1/chat/completions` for Hermes end to end: OpenAI
request → Anthropic canonical → DeepSeek direct → OpenAI response (non-stream +
SSE).

**Preconditions.** Phases 1–2 merged; Phase 0 Chat contract verified. **Confirmed
consumer:** a Hermes session on any chat-completions provider — qwen/gemini/nous/
openrouter or any custom OpenAI-compatible provider, which is the DEFAULT Hermes
`api_mode` (`hermes-agent` `runtime_provider.py:308-341`). Not droppable.

**Steps.**
1. `_lib/openai_inbound.py` — request direction (inverse of the existing
   `_lib/conversion.py` emitters at `:126`–`:159`):
   - `openai_chat_to_anthropic_messages(messages) -> list[dict]`: assistant
     `tool_calls` → `tool_use` content blocks (`id`=`tool_calls[].id` verbatim,
     `name`, `input`=parsed `arguments`); `role:"tool"` messages →
     `tool_result` blocks (`tool_use_id`=`tool_call_id` verbatim); `system` →
     Anthropic `system`; user/assistant text → text blocks.
   - `openai_tool_to_anthropic(tool) -> dict` (inverse of `convert_anthropic_tool`
     `conversion.py:34`): `function.name/description/parameters` →
     `name/description/input_schema`.
   - `chat_request_to_anthropic(body) -> dict`: assemble the full Anthropic body
     (model, messages, system, tools, tool_choice, max_tokens, temperature,
     top_p, stop). **Deterministic**: fixed key order, verbatim id passthrough,
     preserve argument-object key order, inject nothing nondeterministic.
2. `_lib/openai_inbound.py` — response direction:
   - `anthropic_to_chat_response(anthropic_body) -> dict`: Anthropic `content`
     (text + `tool_use`) → `choices[0].message` (`content` + `tool_calls` with
     `arguments` re-serialized to JSON string); map `stop_reason`→`finish_reason`;
     map usage.
3. `routing/synthesis_openai.py`:
   - `synthesize_chat_sse(anthropic_body) -> Iterator[bytes]`: emit
     `chat.completion.chunk` events (role delta, content/tool-call deltas,
     final `finish_reason`, `[DONE]`).
4. `adapters/openai_chat.py`: `OpenAIChatAdapter` wiring the above;
   `extract_pending_tools` = last `role:"tool"` message's `tool_call_id`s matched
   to prior assistant `tool_calls[].function.name` (return `["<unknown>"]` on an
   unresolvable id, mirroring `pending_tool_names` `decide.py:138`).
5. `proxy/server.py`: add `Route("/v1/chat/completions", chat_completions,
   methods=["POST"])`; the handler resolves the adapter + group and calls the
   generalized `_try_route`, then `_serve_passthrough` to the **hermes** upstream
   on `None`.

**Constraints.**
- Invariant 4 (byte-stability): `chat_request_to_anthropic` must be a pure
  deterministic function — verbatim `tool_call_id` passthrough, stable field
  order, preserved argument key order. No `uuid`/timestamp in the request path.
- Invariant 1: malformed OpenAI body, unconvertible message, or conversion error
  → return `None` → passthrough.
- Reuse, do not reimplement, the decision and serving core. The routed body goes
  through `_serve_via_direct`, which sends Anthropic-native verbatim with no
  compression, preserving the byte-stable DeepSeek prefix.

**Verification / done-when.**
- `tests/unit/test_openai_inbound.py` (spec-derived): tool_call→tool_use round
  trip; tool result message→tool_result; tool-def conversion; system handling;
  text-only turns; **byte-stability across two turns** — convert turn N and turn
  N+1 sharing a prefix, assert the serialized shared prefix is byte-identical
  (mirror the direct-path verbatim guard `test_direct_path_sends_tool_result_verbatim`
  in `tests/unit/test_routed_call.py`); immutability (input not mutated); arguments with
  ordered keys survive round trip; unparseable arguments handled.
- `extract_pending_tools` tests: resolvable ids → names; unresolvable id →
  `["<unknown>"]`; turn opener → `[]`.
- `test_synthesis_openai.py`: SSE byte sequence matches a hand-written golden for
  a text response and a tool-call response.
- Integration (mirror `tests/unit/test_routed_call.py` harness): a Hermes
  `/v1/chat/completions` tool-result turn routes to DeepSeek direct and returns a
  valid `chat.completion`; a premium-tool turn passes through to the hermes
  upstream.
- `make test`/`lint`/`type` clean; coverage ≥ 95%.

---

## Phase 4 — OpenAI Responses adapter (Codex, maybe Hermes)  [IN per Phase 0]

**Objective.** Same as Phase 3 for `POST /v1/responses`. Phase 0 confirmed Phase
4 = IN (Codex is Responses-only and sends full input each turn). **This adapter
also serves Hermes sessions running on `openai-codex`/`xai`** (which emit
`codex_responses` — D3), resolved to the `hermes` tool map via `x-acr-agent`.
Phase 3 still ships independently for Hermes chat-completions sessions.

**Preconditions.** Phases 1–2 merged. Independent of Phase 3 — Phases 3 and 4 are
separate adapters that can land in either order; do NOT couple them.

**Steps.**
1. `_lib/openai_inbound.py` — Responses request direction:
   - `responses_input_to_anthropic_messages(input) -> list[dict]`: `function_call`
     items → `tool_use` blocks (`id`=`call_id` verbatim); `function_call_output`
     items → `tool_result` blocks (`tool_use_id`=`call_id`); Codex built-in tool
     items (`local_shell`/`apply_patch` and their outputs) mapped per Phase 0
     findings; `instructions`/system → Anthropic `system`; message items → text.
   - **Strip inbound `Reasoning` items** (they carry `encrypted_content`) before
     building the canonical body — mirror `_strip_thinking_from_messages`
     (`engine.py:41`). Leaving them in makes the DeepSeek prefix non-deterministic
     and breaks the cache (invariant 4).
   - `responses_request_to_anthropic(body) -> dict`: assemble deterministically
     (same byte-stability rules as Phase 3).
2. `_lib/openai_inbound.py` — Responses response direction:
   - `anthropic_to_responses(anthropic_body) -> dict`: Anthropic content →
     Responses `output[]` items (`message` + `function_call`); map usage + status.
3. `routing/synthesis_openai.py`: `synthesize_responses_sse(anthropic_body)` —
   emit the Responses event sequence verified in Phase 0. **Streaming is the live
   path:** Codex ALWAYS streams, so this synthesizer (not a JSON body) is the real
   egress; the non-streaming path is effectively dead for codex but kept for tests.
4. `adapters/openai_responses.py`: `OpenAIResponsesAdapter`;
   `extract_pending_tools` = `function_call_output.call_id`s matched to
   `function_call.name`. **`apply_patch` is a custom/freeform tool** — its result
   can arrive as `CustomToolCallOutput`; the call/output name-match (and the
   premium-escalation guard) must cover custom-tool items, not only
   `FunctionCall`/`FunctionCallOutput`. **Hosted-tool items** (`WebSearchCall`,
   `ImageGenerationCall`) are not `function_call_output`, carry no routable name,
   and are inert for tier selection — skip them without crashing.
5. `proxy/server.py`: add `Route("/v1/responses", responses, methods=["POST"])`;
   passthrough to the **codex** upstream on `None`.

**Constraints.** Same as Phase 3, plus: if `extract_pending_tools` cannot see the
matching `function_call` for an output (history not sent), return `["<unknown>"]`
so the turn safely passes through rather than mis-routing.

**Verification / done-when.** Same battery as Phase 3, retargeted to Responses
shapes: round-trip conversions, byte-stability across two turns, immutability,
golden Responses SSE, integration through DeepSeek direct, premium passthrough to
the codex upstream. `make test`/`lint`/`type` clean; coverage ≥ 95%.

---

## Phase 5 — Per-agent premium upstream / passthrough

**Objective.** A non-routed (premium/unmapped/opener) turn replays to the agent's
own upstream in the agent's own format, not always Anthropic.

**Preconditions.** Phase 2 merged (config carries per-agent `upstream`); Phases
3/4 for the OpenAI groups.

**Steps.**
1. `proxy/server.py` `_serve_passthrough` (`server.py:99`): select the upstream
   from `routes["agents"][group]["upstream"]` (compat shim → `server.upstream`)
   instead of the single `settings.upstream`.
2. Thread `group` into `_serve_passthrough` from each endpoint handler.
3. Confirm credential isolation (invariant 2): passthrough still forwards the
   client's own headers verbatim to the client's own upstream — no tier key, no
   cross-provider credential leakage.

**Constraints.** No conversion on passthrough (client format == upstream format).
Fail-open preserved.

**Verification / done-when.**
- Tests: a `claude_code` passthrough hits the anthropic upstream; a `hermes`/
  `codex` passthrough hits its configured upstream; client headers are forwarded
  unchanged; no tier key appears in passthrough requests.
- `make test`/`lint`/`type` clean; coverage ≥ 95%.

---

## Phase 6 — Docs, example config, end-to-end checks

**Objective.** Make the feature usable and documented.

**Steps.**
1. `config.example.yaml`: finalized `agents:` block for all three groups with real
   tool maps from Phase 0.
2. `README` / `docs/`: document the three endpoints, the `agents:` schema, the
   `x-acr-agent` override, and how to point Codex / Hermes at the router.
3. End-to-end smoke (manual or scripted): drive a real tool-result turn from each
   agent through the router; confirm a routed DeepSeek response and a sane premium
   passthrough.

**Verification / done-when.**
- `acr init` (wizard) produces a config that `decide.load_routes` parses and that
  routes all three groups in tests.
- Example config validates; docs reviewed.
- Full suite green at ≥ 95% coverage; `make lint`/`type` clean.

---

## 7. Cross-cutting verification (run after every phase)

```
make format
make lint
make type
make coverage     # must report >= 95%
```

Plus, for any code that builds a body destined for a native-Anthropic endpoint,
a **byte-stability test**: convert two consecutive turns that share a prefix and
assert the serialized shared prefix is byte-identical. This is the executable
proxy for "the DeepSeek prefix cache still hits." Treat a failure here as a
release blocker, not a warning.

---

## 8. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | Codex relies on `previous_response_id` server state | Router can't see history → can't route Codex | RESOLVED in Phase 0: standard OpenAI provider sends full `input[]` each turn (`store:false`). Only Azure stores server-side → Azure descoped to passthrough. |
| 2 | OpenAI→Anthropic conversion is non-deterministic | Silent DeepSeek cache misses (defeats the whole point) | Pure functions, verbatim id passthrough, stable field order, byte-stability tests (invariant 4) |
| 3 | Three SSE dialects (Anthropic / chat.chunk / Responses) | Streaming clients break | Golden-pair SSE tests per dialect (Phase 3/4) |
| 4 | Wrong per-agent premium-tool mapping | Escalation guard misfires; cheap model performs premium action | Per-agent `premium_tools`; escalation tests per group (Phase 2) |
| 5 | Argument JSON round-trip reorders keys/normalizes floats | Cache misses | Preserve key order; do not reformat; test |
| 6 | Flat-config users regress | Existing deployments break | Mandatory compat shim + regression test (Phase 2) |
| 7 | A cheap-tier `key_env` leaks into a per-provider file | Cheap key reachable from premium side; breaks single-sourcing | Bootstrap/load validation rejects any cheap `key_env` in a per-provider file; cheap creds live only in global `tiers.*.key_env` (invariants 2, 5; Phase 7) |
| 8 | Ambiguous caller identity mis-routes to the wrong upstream | Client credential forwarded to a guessed provider → cross-provider leak | Fail closed: unresolved identity → `400`, never a guessed upstream, unless `router.fallback` is explicitly set (Phase 7) |
| 9 | Bootstrap overwrites an operator's edited per-provider file | Silent loss of hand-tuned config | Bootstrap is create-if-absent only; never rewrites an existing file; idempotent; test asserts an existing file is untouched (Phase 7) |
| 10 | Partial config (some per-provider files present, some missing) | Routing silently degrades for the missing family | Bootstrap materializes the full known set before serving; `load_provider_files` skips a missing file and `assemble_routes` falls back to the compat shim for that group (fail open, invariant 1) — a missing file degrades to premium-default behavior, never a crash (Phase 7) |

---

## 9. Open decisions (Phase 0 results recorded in phase0-findings.md)

- **D1. Codex wire mode** — RESOLVED: Responses API only (Codex removed Chat;
  `wire_api="chat"` hard-errors). Phase 4 IN.
- **D2. Codex statefulness** — RESOLVED: full `input[]` history sent every turn
  (`store:false` except Azure). Codex routing is feasible (risk #1 retired for the
  standard OpenAI provider; Azure is descoped to passthrough).
- **D3. Hermes wire details** — RESOLVED: tool map / premium list / tiers
  GROUNDED in the real `~/.hermes/plugins/tool-router/routes.yaml` (distinct vocab;
  adds a `structured` tier; premium=`[patch, clarify, delegate_task]`; premium
  upstream = Hermes session model, NOT anthropic). Inbound wire is **per-provider
  multi-wire** (`hermes-agent` `runtime_provider.py:240-341`): `chat_completions`
  (default + qwen/gemini/nous/openrouter), `codex_responses` (openai-codex/xai),
  `anthropic_messages` (anthropic/minimax). Both Phase 3 and Phase 4 have real
  Hermes consumers — both ship; `/v1/messages` already covers the Anthropic case.
  `bedrock_converse` / `codex_app_server` out of scope (passthrough).
- **D4. Content reduction on the routed path** — SUPERSEDED. The old
  `reduce.reduce_text` module is gone; content reduction now runs as headroom
  compression on the LiteLLM (non-direct) path, gated by the `compress_routed`
  setting (default on) and respecting headroom's coding-agent tool exclusions
  (`routing/compression.py`, `_serve_via_litellm` `engine.py:179`, compression
  applied at `engine.py:213`). The direct
  (DeepSeek) path stays verbatim by design (invariant 4). No per-tier/per-tool
  dup-line knob is planned; not required for any phase here.
- **D5. On-disk config layout** — DECIDED (Phase 7): three per-provider files
  (`config/{claude-code,codex,hermes}.yaml`) plus a global `config.yaml`
  (`server`/`settings`/`tiers`/`router`) are the target layout. The single-file
  `agents:` block (Phase 2) and the legacy flat file are both retained as
  compat-only load modes; all three converge to one canonical `agents:` dict via
  `provider_config.assemble_routes()`. Rationale: per-family isolation and
  ownership (R5) without forking the decision core (KISS/DRY). Cheap-tier creds
  stay single-sourced in global `tiers:`; per-provider files carry the premium
  side only.
- **D6. Fail-closed vs invariant 1** — DECIDED (Phase 7): fail-OPEN remains the
  serving-layer rule (a resolved group's turn never breaks; errors fall back to
  that group's passthrough). Fail-CLOSED applies only at the attribution layer:
  an unresolved/ambiguous identity returns `400` rather than forward the client's
  credential to a guessed upstream (invariant 2). `router.fallback: <group>` is
  the explicit operator opt-out back into passthrough. The two rules do not
  conflict because they govern different layers.


## Phase 7 — Provider-specific YAML routing (drafted as "Phase X")

**Objective.** Make provider identity a first-class routing dimension: Claude Code,
Codex, and Hermes (CLI and desktop of each share one group, R5) each resolve
through their own on-disk YAML file with isolated premium-side settings, driven by
an explicit `router:` map, while the cheap-routing decision core stays byte-for-byte
unchanged. This is the on-disk repackaging of the Phase 2 `agents:` canonical shape
(§4.1) — not a second decision engine.

**Relationship to Phase 2 (read before coding).** Phase 2 defined the canonical
in-memory shape `routes["agents"][group]` and a flat-config compat shim. Phase 7
adds a third load mode that assembles three per-provider files into that *same*
dict. After assembly, `decide.tier_for_tools`, `engine.escalates`, `routed_call`,
and `_serve_passthrough` are untouched. If Phase 7 requires any change to the
decision core, the design is wrong — stop and reconcile against §4.

**Fail-closed vs invariant 1 (decision, D6).** Fail-OPEN stays at the serving
layer: once a group resolves, every downstream error falls back to *that group's*
passthrough (invariant 1 intact). Fail-CLOSED applies only to identity
attribution: a request whose group cannot be resolved (no header, no UA match, no
endpoint default, no `router.fallback`) is rejected `400` rather than forwarded to
a guessed upstream — because forwarding the client's own premium credential to the
wrong provider is a credential leak (invariant 2). The two never fire on the same
condition.

**Preconditions.** Phases 1–5 merged (adapters, per-agent `agents:` schema +
compat shim, per-agent passthrough upstream). Phase 7 is the last structural phase
before Phase 6 docs.

### On-disk layout (target)

```
~/.ai-calls-router/
  config.yaml            # server:, settings:, tiers:, router:   (cheap side + identity policy)
  config/
    claude-code.yaml     # agents.claude_code payload (premium side only)
    codex.yaml           # agents.codex payload
    hermes.yaml          # agents.hermes payload
```

Per-provider file schema and `router:` block schema: see §4.1. Hard rule
(invariants 2, 5): a per-provider file carries the **premium side only** and MUST
NOT contain a cheap-tier `key_env`; the cheap credential lives only in global
`tiers.*.key_env`. `auth:` names a credential *source* (`oauth_passthrough` |
`api_key_env`), never a literal secret.

### Steps

1. **`ai_calls_router/_lib/config.py`** — add path resolvers only (no IO beyond
   path construction, no routing imports — `config.py` is the lowest layer):
   ```python
   def provider_config_dir() -> Path:
       """Directory holding per-provider YAMLs ($ACR_HOME/config)."""
       return home_dir() / "config"

   def provider_config_path(group: str) -> Path:
       """Path to one group's per-provider YAML (claude_code -> claude-code.yaml)."""
       return provider_config_dir() / f"{group.replace('_', '-')}.yaml"
   ```
   Do **not** add `KNOWN_GROUPS` here. It is single-sourced at
   `adapters/base.py:15` (`frozenset({"claude_code", "codex", "hermes"})`);
   `provider_config_path` takes the group as an argument (no iteration), so
   `config.py` stays free of any routing dependency. Callers that need to
   iterate the set (`provider_config`, `bootstrap`) import `KNOWN_GROUPS` from
   `adapters.base`.

2. **`ai_calls_router/routing/provider_config.py`** (new; assembly + identity).
   Separate the impure read from the pure merge — the doc's earlier "pure
   functions that read each file" was self-contradictory. `provider_config`
   imports `KNOWN_GROUPS` from `adapters.base` and `with_agent_compat` +
   `agent_upstream` from `decide`; it must **not** be imported by `decide`
   (that would cycle).
   ```python
   class ProviderConfigError(ValueError):
       """A per-provider file violates an invariant (e.g. carries a cheap
       key_env). Raised by assemble_routes; the caller fails open by skipping
       the offending payload and logging a WARNING."""

   def load_provider_files() -> dict[str, dict[str, Any]]:
       """Impure: read every present provider_config_path(group).

       Returns {group: parsed_payload} for files that exist and parse.
       A missing file is simply absent from the result (fail open). A
       malformed file is skipped with a WARNING (never raises into serving,
       invariant 1). Does no merging and no invariant validation — that is
       assemble_routes' job."""

   def assemble_routes(
       base: dict[str, Any], *, provider_files: dict[str, dict[str, Any]]
   ) -> dict[str, Any]:
       """Pure: merge provider_files into a NEW canonical routes dict.

       For each group present in provider_files, validate the payload
       (reject any cheap-tier or top-level key_env -> ProviderConfigError;
       require group/upstream/wire) and place it at result['agents'][group].
       Any group without a file falls back to base['agents'] / the Phase 2
       compat shim via with_agent_compat(base). Never mutates base. When
       provider_files is empty, returns with_agent_compat(base) verbatim.
       The caller catches ProviderConfigError, drops that one payload, logs
       a WARNING, and keeps serving (invariant 1; risk #7)."""

   def router_map(routes: dict[str, Any]) -> dict[str, Any] | None:
       """Return routes['router'] if a router: block is present, else None.

       Does NOT synthesize endpoint_defaults — the endpoint->group map lives
       only in adapter.default_agent_group (modes a/b) and router.endpoint_defaults
       (mode c), never a third place (see §5)."""

   def resolve_agent_group(
       *,
       path: str,
       headers: Mapping[str, str],
       routes: dict[str, Any],
       adapter_default: str,
   ) -> str | None:
       """Resolve the agent group by §5 precedence:
       x-acr-agent > router.user_agent_map contains-match (case-insensitive)
       > router.endpoint_defaults[path] > router.fallback > adapter_default.
       Steps 2-4 are skipped when no router: block is present, so modes a/b
       fall straight through to adapter_default and never return None for a
       known endpoint. Returns None ONLY when a router: block is present,
       fallback is null, and steps 1-3 all miss (caller fails closed -> 400).
       Validates header/result against KNOWN_GROUPS."""
   ```
   Drop the planned `provider_upstream` — per-group upstream is already served by
   Phase 5's `decide.agent_upstream(routes, group)`; reuse it (DRY). `auth.key_env`
   naming a *premium* source is allowed; a bare or cheap-tier `key_env` is rejected
   by `assemble_routes` (risk #7).

3. **`ai_calls_router/ops/bootstrap.py`** (new; materialize-if-absent, idempotent):
   ```python
   def ensure_provider_configs(*, ask: Callable[[str], str] | None = None) -> list[Path]:
       """Create any missing per-provider YAML from agent_defaults.py.

       For each group in KNOWN_GROUPS whose provider_config_path is absent,
       write a template (group, upstream, auth=oauth_passthrough, wire,
       endpoints, tools from AGENT_DEFAULT_TOOLS[group], premium_tools from
       AGENT_DEFAULT_PREMIUM_TOOLS[group], fallback=passthrough). NEVER
       overwrite an existing file (risk #9). Returns the paths created
       (empty list when all present)."""
   ```
   Templates derive entirely from `agent_defaults.py` (DRY) — no tool maps
   duplicated here. Writes are create-only (`if not path.exists()`), so re-running
   is a no-op.

4. **`ai_calls_router/proxy/server.py`** (assembly happens at the server boundary,
   NOT inside `decide.load_routes` — `provider_config` imports `decide`, so calling
   `assemble_routes` from inside `decide` would cycle):
   - In `_lifespan` (`server.py:508`), before serving, call
     `bootstrap.ensure_provider_configs()` so missing files are materialized once
     at startup (risk #10).
   - Add one private helper `_load_assembled_routes()` at the server boundary that
     does `assemble_routes(routing.load_routes(), provider_files=load_provider_files())`
     and memoizes on the **max** mtime of the global `config.yaml` and every present
     per-provider file (so editing any `config/*.yaml` invalidates the cache).
     `load_routes` keeps its own mtime cache for the global file; this helper layers
     the provider-file mtimes on top.
   - Replace the two `routing.load_routes()` serving call sites with
     `_load_assembled_routes()`: `_serve_passthrough` (`server.py:99`) and
     `_resolve_tier_config` (def `server.py:187`, the `load_routes()` call at
     `server.py:197`). Every handler then sees the merged canonical dict.
   - At the `_messages` handler (call site `server.py:443`), replace the Phase 5
     `resolve_agent_group(adapter.default_agent_group, request.headers)` with
     `provider_config.resolve_agent_group(path=path, headers=request.headers,
     routes=routes, adapter_default=adapter.default_agent_group)`. On `None`, reject
     `400` (`{"error": "unresolved agent identity"}`) BEFORE any credential is
     attached — do NOT passthrough to a guessed upstream (invariant 2 / D6). Thread
     the resolved `group` into `_try_route` / `_serve_passthrough` exactly as Phases
     2/5 already do.
   - Delete the superseded header-only `resolve_agent_group(default_group, headers)`
     from `adapters/__init__.py:49` and drop it from `__all__`; the
     `provider_config` version (router-aware, fail-closed) replaces it. `server.py`
     stops importing it from `...adapters`.

5. **`ai_calls_router/ops/wizard.py`** — `acr init` also calls
   `bootstrap.ensure_provider_configs()` and seeds the global `router:` block
   (`endpoint_defaults` from §5, the three `user_agent_map` contains-rules, and
   `fallback: null` = fail closed). The global `config.yaml` it writes keeps
   `server`/`settings`/`tiers`; the per-provider `agents:` content now lives in the
   `config/` files, not inline.

6. **Docs** (folds into Phase 6): provider→YAML mapping, the `router:` precedence
   rules (§5), ambiguous/unknown handling (`400` unless `router.fallback`), and how
   the three families differ in config ownership and endpoints.

### Constraints
- **Decision core unchanged.** `assemble_routes` output must be a drop-in for
  Phase 2's `routes` dict; no signature in `decide.py`/`engine.py` changes in this
  phase.
- **Invariants 2 + 5.** Per-provider files never carry a cheap key; validation
  rejects them; `auth:` names a source, not a secret.
- **Invariant 1 vs fail-closed (D6).** Read/assembly errors fail open (skip file,
  keep serving). Only unresolved *identity* fails closed (`400`).
- **Idempotent, non-destructive bootstrap.** Create-if-absent only; never rewrite
  an edited file.
- Immutability, no nested functions, Google docstrings, radon A/B as elsewhere.

### Verification / done-when

`tests/unit/test_provider_config.py` (spec-derived, adversarial):
- `assemble_routes` merges three present files into
  `routes["agents"][group]` for each group; result is byte-equivalent to a
  hand-written single-file `agents:` block (proves the canonical-merge contract).
- A per-provider file containing a cheap `key_env` is **rejected** (raises/skipped
  + WARNING); the cheap key never appears in the assembled dict (risk #7).
- A missing file for one group falls back to the compat shim / `agent_defaults`
  without dropping the other groups (risk #10).
- `resolve_agent_group` precedence: `x-acr-agent` wins over UA and endpoint;
  UA `contains`-match is **case-insensitive** and wins over endpoint default;
  endpoint default applies when no header/UA; with no router: block (modes a/b)
  it falls straight through to `adapter_default` and never returns `None`;
  **unresolved → `None`** ONLY when a router: block is present, `fallback` is unset,
  and steps 1-3 miss; `router.fallback` set → returns that group; invalid
  `x-acr-agent` value → ignored, falls through (not honored).
- `agent_upstream(routes, group)` (Phase 5, reused — no new `provider_upstream`)
  returns the per-group upstream; compat shim → `server.upstream`.
- Immutability: `assemble_routes` does not mutate `base`.

`tests/unit/test_bootstrap.py`:
- All files absent → `ensure_provider_configs` creates exactly the KNOWN_GROUPS
  set; each parses via `decide.load_routes` and assembles cleanly.
- One file already present and edited → it is **left untouched** (byte-compare
  before/after); only the missing ones are created (risk #9).
- Second run is a no-op (returns `[]`).
- Generated templates carry no cheap `key_env` and pass `provider_config`
  validation (invariants 2, 5).

Server-level (mirror existing handler tests):
- A request with `x-acr-agent: codex` on `/v1/responses` resolves the `codex`
  group; a Hermes UA on `/v1/chat/completions` resolves `hermes`.
- A request that resolves no group (unknown UA, no header, no endpoint default,
  `fallback: null`) returns `400` and forwards no credential upstream (assert the
  client header never reaches any upstream — invariant 2 / risk #8).
- A resolved-but-premium turn still passes through to that group's own upstream
  (Phase 5 behavior unchanged).

`make test`/`lint`/`type` clean; coverage ≥ 95%; byte-stability suite (§7) still
green (assembly must not perturb the routed-body prefix).
