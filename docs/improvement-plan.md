# Codebase Improvement Plan

Findings from a full-codebase architecture review (2026-07-02): directory
structure, design patterns, coupling, DRY, redundancy, KISS, extensibility,
file sizes, and consistency. Security was explicitly out of scope. Scope:
`ai_calls_router/` — 12,210 production LOC across 58 files, reviewed against
the seams documented in `CLAUDE.md` and `.claude/rules/routing.md`.

Sections 1-9 are the analysis; section 10 is the ranked action list.

## 1. Directory structure

Domain-based split (`proxy/`, `routing/`, `accounting/`, `ops/`, `_lib/`) is
the right call at this scale. Two soft spots:

- **`_lib/` is a junk-drawer name.** `conversion.py` (394 LOC) and
  `responses_inbound.py` (744 LOC) are wire-format translation logic — the
  same concern as `routing/adapters/`, parked in a different package with no
  doc explaining why. "Where does client-format translation live" currently
  has two correct-looking answers.
- **`routing/` is overloaded** (19 files): decision-making (`decide`,
  `tier_selection`, `agent_topology`, `credential_resolution`), four serving
  paths, two compression modules, and three synthesis modules, all flat in
  one folder. Module docstrings mostly compensate; a
  `routing/{decision,serving,synthesis}/` second-level split would read
  cleaner if the folder keeps growing.

## 2. Design patterns

- **Facade** — `routing/decide.py` is textbook: explicitly re-exports
  `agent_topology` / `tier_selection` / `credential_resolution` symbols, and
  its docstring states the split is internal. Best-documented pattern in the
  repo.
- **Chain of Responsibility** — `route_dispatch.try_native_or_oauth_route`
  walks `try_anthropic_oauth_route -> try_codex_direct_route ->
  try_oauth_responses_route -> _serve_routed_litellm`; each `None` return
  means "not mine, try next". Codified as a contract in
  `.claude/rules/routing.md`, not just accidental code shape.
- **Adapter** — `routing/adapters/{anthropic_messages,openai_chat,
  openai_responses}.py` implement the same `ClientAdapter` Protocol
  uniformly (`extract_pending_tools`, `to_anthropic_request`,
  `to_client_response`, `to_client_sse`). Cleanest pattern application in
  the codebase.
- **Not a strategy pattern, despite looking like one**: the three
  OAuth/direct serving functions (`anthropic_oauth.messages_call`,
  `codex_direct.responses_call`, `direct.direct_call`) have different
  signatures and different return shapes (`JsonObject | None` vs
  `tuple[JsonObject, tuple[int, int, int, int], ShrinkStats] | None`).
  Copy-paste-adapt, not a shared interface — acceptable given protocol
  heterogeneity, but it should not be mistaken for polymorphism.

## 3. Coupling

The declared seam (`server.py` = transport, `orchestrator.py` = policy,
`route_dispatch.py` = dispatch, serving paths = wire) holds up under import
tracing. `server.py` never imports `engine` / `codex_direct` /
`anthropic_oauth` directly; `orchestrator.py` never imports `engine`
directly. No circular imports found.

Two real cracks:

- **`proxy/server.py:47-48`**:

  ```python
  codex_direct = route_dispatch.codex_direct
  _try_codex_direct_route = route_dispatch.try_codex_direct_route
  ```

  A transport-layer module holding direct references to serving-layer
  internals it never calls, contradicting its own docstring ("transport
  concerns only"). If tests still reach through this, they should import
  `route_dispatch` directly; otherwise delete the re-exports.
- **Ambiguous alias**: `route_dispatch.py` and `orchestrator.py` both do
  `from ai_calls_router.routing import decide as routing`, so the local name
  `routing` means "the `decide.py` facade", not "the `routing/` package", in
  exactly the two files most likely to be read alongside package-level
  imports. `provider_config.py` uses named imports with no alias. One
  convention should win.

`accounting.metrics.get_metrics()` is a module-level singleton reached from
`engine.py`, `orchestrator.py`, and `server.py._lifespan`. Legitimate for
process-wide counters, but it is temporal coupling (callers assume
`bootstrap()` already ran) that is invisible from any signature.

## 4. DRY violations

- **Anthropic-shape usage extraction duplicated.**
  `routing/anthropic_oauth.py:92` (`_response_usage`) and
  `routing/engine.py:512` (`_usage_from_anthropic`) perform the identical
  four-field extraction (`input_tokens`, `output_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens`, same
  `jsonnum.int_value(..., minimum=0)` calls); one returns a tuple, the other
  wraps it in `RouteUsage`. Collapse into one shared function and have
  `engine.py` wrap the result. The Responses-shape extractor in
  `codex_direct.py` reads different field names and legitimately stays
  separate.
- **`native_model_id` — one name, three meanings.**
  `anthropic_oauth.py:74`, `codex_direct.py`, and `direct.py` each define
  `native_model_id` with different stripping rules (prefix constant vs
  `/`-split vs provider-specific). Worse than duplication: a false cognate.
  Rename to disambiguate (`strip_anthropic_prefix`, `strip_codex_prefix`,
  `strip_direct_prefix` or similar) even if the logic cannot be unified.
- **`try_*_route` boilerplate in `route_dispatch.py`.** Four functions of
  roughly 90-100 lines each — about 350 of the file's ~700 lines — repeat
  the same skeleton: call serving path, check `None`, extract tool names,
  build `RouteUsage`, `record_route_outcome`, shape the client response,
  return `RouteAttempt`. They differ only in which serving function they
  call and how the response is shaped. This is the single highest-value DRY
  target in the codebase: extract one
  `_dispatch_serve(serve_fn, tool_name_extractor, response_shaper,
  route_label)` helper. It also reduces per-function cognitive complexity
  (project gate is <= 15).

Counter-examples done right: `route_json.py` and `jsonnum.py` correctly
centralize repeated coercion logic used across 6+ files. The failure above
is that the same discipline was not applied to usage extraction.

## 5. Redundancy — justified vs not

- **Three serving paths, two compression call sites, three synthesis
  modules: all justified.** They exist because of documented hard
  constraints — cache byte-determinism forbids mutating the DeepSeek direct
  path, Anthropic Messages OAuth and OpenAI Responses OAuth are genuinely
  different wire protocols, and the three SSE grammars differ per client
  format. `CLAUDE.md` states this explicitly and the code matches. Leave
  these alone.
- **The `try_*_route` boilerplate is the opposite case** — the same dispatch
  shape repeated per provider, with no protocol heterogeneity forcing it.
  It grew by copy-paste and is the one redundancy to remove (section 4).

## 6. KISS

Generally restrained: no premature ABCs, no dependency-injection framework,
`ClientAdapter` correctly uses `Protocol` rather than `ABC`. Two minor
over-engineering candidates, neither urgent:

- `prepare_route` returns a `RouteDecision | RouteAttempt` union that every
  caller `isinstance`-checks. Workable (fits the fail-open,
  return-values-over-exceptions philosophy), but each consumer repeats the
  same guard. One typed exception caught once in `try_route` would remove
  the propagated union.
- `_RoutesCache` in `server.py` is a two-field dataclass wrapping what is
  functionally a `tuple | None` module global. Single-use, single-file,
  harmless.

## 7. Extensibility

- **New client wire adapter** (e.g. Gemini): implement `ClientAdapter`,
  register in `adapter_for_path`. No changes to `orchestrator.py`,
  `route_dispatch.py`, or `engine.py`. Clean.
- **New routing tier**: pure config change, zero code. Best extension point
  in the system.
- **New direct provider** (DeepSeek-like native Anthropic endpoint): one
  entry in `direct.py::DIRECT_ANTHROPIC_ENDPOINTS`. Trivial.
- **New OAuth protocol**: a new serving module, a new `try_*_route`
  function following the ~90-line boilerplate, wiring into the dispatch
  chain, and a tier predicate in `config_schema.py`. Bounded and documented
  in `.claude/rules/routing.md`, but the 90-line step exists only because of
  the section 4 duplication — after the `_dispatch_serve` extraction it
  becomes a ~15-line closure.

## 8. Oversized files

| File | LOC | Verdict |
|---|---|---|
| `accounting/metrics.py` | 929 | Split. SQLite persistence (`_ensure_request_db` ... `_load_premium_token_totals`) is a self-contained DB layer, candidate `metrics_db.py`. `identify_agent` / `identify_provider` / `session_fingerprint` are pure functions unrelated to `_Metrics` internals, candidate `identity.py`. Result: three ~300-line single-responsibility files. |
| `_lib/responses_inbound.py` | 744 | Borderline but defensible — one wire format's conversion logic, inherently verbose (many item types), under the 800-line ceiling. Leave it. |
| `proxy/route_dispatch.py` | ~700 | Split along the obvious seam: the four OAuth/Codex `try_*_route` functions (~350 lines) move to `route_dispatch_oauth.py`; the core contract (`RouteAttempt`, `RouteDecision`, `prepare_route`, `try_route`) stays. Low risk — no reverse dependency from the `try_*` functions into the rest of the file. The test layout already implies this split (`test_route_dispatch_anthropic_oauth.py` covers only the OAuth slice). |
| `routing/engine.py` | ~700 | Cohesive: single serving path plus escalation guard plus outcome recording. Within tolerance. |
| `routing/codex_direct.py` | 481 | The SSE-reconstruction helpers (~170 lines) are a separable "rebuild a Responses object from SSE" concern. Not urgent; split if it grows. |

None of these mix unrelated domains — they grew by accretion of cases, which
is a better failure mode than mixed-concern bloat. `metrics.py` and
`route_dispatch.py` have still crossed the point where splitting reduces
real cognitive load.

## 9. Inconsistencies

- `native_model_id` naming collision (section 4) — the sharpest one.
- Alias inconsistency: `decide as routing` in two files, named imports
  elsewhere, for the same module.
- Undocumented convention split: serving paths return `None` on failure;
  config/schema layers raise typed exceptions. Both are individually fine
  and internally consistent, but "serving paths return None, config paths
  raise" is written nowhere — it is only inferable by reading several files.
  Worth one line in `.claude/rules/routing.md`.
- Test-to-file mapping is strong everywhere except `route_dispatch.py`,
  which again argues for the section 8 split.

## 10. Action list, ranked by impact

1. **Extract the `try_*_route` dispatch skeleton** into a shared
   `_dispatch_serve` helper (sections 4, 7). Biggest DRY win: ~350 lines
   collapse to ~150, and per-function cognitive complexity drops.
2. **Split `proxy/route_dispatch.py`** along the OAuth/core seam
   (section 8). Low risk, halves the largest proxy-layer file, matches the
   existing test structure. Natural to do together with item 1.
3. **Split `accounting/metrics.py`** into `metrics.py` (counters +
   `_Metrics`), `metrics_db.py` (SQLite persistence), and `identity.py`
   (agent/provider identification, session fingerprint).
4. **Unify the Anthropic-shape usage extractors and rename the three
   `native_model_id` functions** (section 4). Small effort, removes a real
   false-cognate trap.
5. **Remove or redirect the `server.py:47-48` re-exports** of
   `route_dispatch` internals (section 3). Cheapest fix on the list; closes
   the one crack in the transport/policy seam.

Deliberately deferred: folding `_lib/`'s wire-conversion files into
`routing/` (section 1). It is a real ownership ambiguity, but the fix
touches ~15 files' imports for no functional payoff — park it until a
broader restructure is happening anyway.

Standing constraints for any of the above: routed-body byte determinism and
prefix stability (cache safety), fail-open serving paths, the
transport/policy/dispatch seam, and all `make qa` gates (coverage >= 95%,
pyrefly strict, cognitive complexity <= 15).
