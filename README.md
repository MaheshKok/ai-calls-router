<div align="center">

```
   █████╗   ██████╗ ██████╗
  ██╔══██╗ ██╔════╝ ██╔══██╗
  ███████║ ██║      ██████╔╝
  ██╔══██║ ██║      ██╔══██╗
  ██║  ██║ ╚██████╗ ██║  ██║
  ╚═╝  ╚═╝  ╚═════╝ ╚═╝  ╚═╝
```

**The model right-sizing router for coding agents**

route mechanical turns to an efficient model · Claude Code · Hermes · subscription-OAuth · fail-open · local-first · zero agent changes · honest accounting

[![CI](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml/badge.svg)](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Install](#install) · [How it works](#how-it-works-30-seconds) · [Get started](#get-started-60-seconds) · [Configure](#configure) · [Compared to](#compared-to)

</div>

---

## What it does

A coding agent spends most of its turns *not* thinking — it reads a file, runs a
grep, runs a shell command, then spends the next turn just digesting that
output. Those tool-result turns are mechanical. They do not need a flagship
model's reasoning, but they are billed at the flagship's price, and on a
subscription they burn the flagship's quota.

ai-calls-router sits between your agent and the upstream API and splits the
stream:

- **Routed** — tool-result turns (digesting `Read`, `Grep`, `Bash`, … output)
  go to an efficient model.
- **Passthrough** — decision turns (a fresh prompt, an `Edit`, a `Write`,
  anything it cannot classify) go straight to the premium upstream, untouched.

The agent never knows. The response always claims the model it asked for, so
model display and session restore stay correct. Same session, fraction of the
premium spend.

**What's inside:**

- **Proxy** — a local Starlette + uvicorn daemon on `127.0.0.1:8747`.
- **Per-agent routing** — independent tool→tier maps and upstreams for
  `claude_code` and `hermes`.
- **Subscription-OAuth serving** — route routine turns to Sonnet / Codex on your
  *subscription* bearer, off the premium quota.
- **Savings ledger + live dashboard** — every routed turn recorded against its
  true model, never a guessed number.

---

## How it works (30 seconds)

```
  Claude Code / Hermes
          │
          ▼
   ┌─────────────────┐   resolve agent group (endpoint + x-acr-agent)
   │   acr daemon    │   classify turn (pending tool_result? tool → tier)
   └─────────────────┘
          │
    ┌─────┴───────────────────────────┐
    ▼                                  ▼
  ROUTED                          PASSTHROUGH
  tool_result → routed tier       fresh / premium / unknown / any failure
  tier model + tier credential    client headers forwarded verbatim
    │                                  │
    ▼                                  ▼
  premium tool in response? ──► escalate to passthrough
    │
    ▼
  savings.jsonl (true model) → mask response model → client-format JSON/SSE
```

A request carrying `tool_result` blocks is digesting a tool's output. The proxy
finds which tool produced each result, maps it to a tier (`tools:` in the
config), and serves that turn on the tier's efficient model. Everything else streams
straight through in the client's native format — the proxy never translates a
passthrough body.

---

## Get started (60 seconds)

```bash
# 1 — Install (isolated)
uv tool install ai-calls-router      # or: pipx install ai-calls-router

# 2 — Write default config + provider files to ~/.ai-calls-router/
acr init

# 3 — Start the daemon
acr start
acr status                           # prints pid + URL

# 4 — Point Claude Code at it
ANTHROPIC_BASE_URL=http://127.0.0.1:8747 claude -p "explain this repo"

# 5 — Watch what routed and what it saved
open http://127.0.0.1:8747/dashboard
acr savings
```

Hermes instead of Claude Code:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8747/v1 hermes
```

---

## Proof

ai-calls-router does not ask you to trust a marketing number — it proves the
savings on your own traffic, locally, and refuses to invent figures it cannot
back.

- **Live dashboard** at `http://127.0.0.1:8747/dashboard` — recent requests,
  routed-vs-passthrough counts, per-session grouping, cache read/write tokens,
  compression stats, cumulative savings.
- **Savings ledger** at `~/.ai-calls-router/savings.jsonl` — one row per routed
  turn, recorded against the **true** routed model (never the masked client id).
- **`acr savings`** — aggregated report from that ledger.

> Pricing is best-effort and honest. A routed model is priced from LiteLLM's own
> table first; a model it still cannot price is **omitted** from the ledger, not
> estimated. Savings you see are savings that actually happened.

---

## When to use · When to skip

**Use it when** your agent does real tool-heavy work — reading code, grepping,
running commands, processing MCP tool output — and you want those turns off the
premium model. The more tool churn, the more it saves.

**Skip it when** your workload is mostly short conversational turns with little
tool use; there is little mechanical work to route, so the win is small.

---

## Agent compatibility

| Agent              | Endpoint(s)                                  | Routed model (default)                                                 | Auth                              | Status |
| ------------------ | -------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------- | :----: |
| **Claude Code**    | `POST /v1/messages`                          | `claude-sonnet-4-6`                                                    | Anthropic subscription OAuth      |   ✅   |
| **Hermes**         | `POST /v1/chat/completions`, `/v1/responses` | `gpt-5.4-mini` (fast) · `gpt-5.3-codex-spark` (code/crud/structured)  | ChatGPT subscription OAuth        |   ✅   |
| **Claude Desktop** | via embedded shim                            | same as Claude Code                                                    | Anthropic subscription OAuth      |   ✅   |

- Claude Code's routed tiers go to **Sonnet 4.6 on the subscription OAuth
  bearer** — separate Sonnet quota, premium (Opus) passthrough untouched, and a
  Sonnet-quota `429` fails open to it.
- Provider files are **create-only**: the proxy never overwrites an edited file.
  Change a model, tier, or tool mapping and your edit survives every restart.

---

## Install

Requires Python 3.11 or newer.

```bash
# Isolated (recommended)
uv tool install ai-calls-router
pipx install ai-calls-router

# Plain pip also works
pip install ai-calls-router

# Local dev from a checkout
make install          # creates .venv if missing, installs acr + dev deps
```

**Granular extras** — optional headroom compression for the routed serving path
(a deterministic no-op if not installed):

```bash
uv tool install "ai-calls-router[compression]"
```

### Point Claude Code at the proxy

`ANTHROPIC_BASE_URL` is how Claude Code finds the proxy. Pick one — mixing them
is redundant.

```bash
# A — per-invocation (testing)
ANTHROPIC_BASE_URL=http://127.0.0.1:8747 claude -p "..."

# B — shell-level (~/.zshrc or ~/.bashrc)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8747

# C — persistent Claude settings (terminal + IDE + Desktop)
acr desktop on        # write into ~/.claude/settings.json (backs up any existing value)
acr desktop status
acr desktop off       # restore the previous value

# D — launcher that also boots the daemon
acr code -- -p "explain this repo"
```

> **Claude Desktop** does not inherit your shell env and may reset the API base
> URL. Install the embedded shim: `scripts/desktop-shim/apply.sh`
> (`--revert` to remove). It refuses to clobber an existing `claude-real`
> without `--force`.

**Not routing?** `acr desktop status` (conflicting base URL?) · `acr status`
(daemon up?) · `~/.ai-calls-router/acr.log` (who got the traffic?).

---

## Configure

Everything is reloaded by file mtime on each request — edits apply with no
restart. (Code changes still need a daemon restart.)

```
~/.ai-calls-router/
├── config.yaml              # global: server, router rules, tier precedence
├── config/
│   ├── claude-code.yaml     # claude_code: upstream, tiers, tool→tier map
│   └── hermes.yaml          # hermes:      upstream, tiers, tool→tier map
├── .env                     # tier credentials (optional)
└── savings.jsonl            # savings ledger
```

`config.example.yaml` in the repo is the documented template.

**Tool → tier map** — an exact tool name wins over a trailing-`*` glob, so new
MCP tools are covered automatically:

```yaml
tools:
  Bash: fast
  Read: code
  Edit: premium                     # premium = always passthrough
  mcp__lean-ctx__*: code            # glob covers every lean-ctx tool…
  mcp__lean-ctx__ctx_edit: premium  # …except the ones named explicitly
```

**Tier** — model, credential, and reasoning effort:

```yaml
tiers:
  code:
    provider: anthropic
    model: anthropic/claude-sonnet-4-6
    auth: { mode: oauth }           # or { mode: api_key_env, key_env: MY_KEY }
    effort: high                    # low | medium | high | max
    input_cost_per_1m: 3.0          # optional — only if LiteLLM cannot price it
    output_cost_per_1m: 15.0
```

OAuth tiers use the agent's own subscription bearer — no separate key. API-key
tiers read `auth.key_env` from the environment or `~/.ai-calls-router/.env`.

---

## Compared to

| | Scope | Where it runs | Reversible | Agent changes |
| --- | --- | --- | --- | --- |
| **Raw premium (no proxy)** | every turn billed at premium | — | — | — |
| **Headroom** | compresses tokens *within* each call | local proxy / lib / MCP | yes | none |
| **ai-calls-router** | routes whole mechanical *turns* to an efficient model | local proxy | yes (fail-open) | none |

Complementary, not competing: ai-calls-router can call Headroom's compressor on
its routed path (`[compression]` extra) to shrink the body *before* it serves
the turn on the routed model.

---

## Guarantees

1. The response always claims the model the client asked for. Routing is
   invisible.
2. Your premium credential is never forwarded to a routed provider — routed
   calls carry only the tier credential.
3. Any serving failure on the routing path falls back to passthrough.
   Unresolved *identity* is stricter: `400 {"error": "unresolved agent identity"}`
   before any upstream call.
4. The ledger records the real routed model; only the client-facing response is
   masked.
5. Cost numbers are never fabricated — an unpriceable model is left out of the
   ledger.

---

## Commands

| Command | Purpose |
| --- | --- |
| `acr init` | Generate `config.yaml` and provider files. |
| `acr start` / `acr stop` | Start / stop the background daemon. |
| `acr restart` | Restart the daemon. |
| `acr status` | Daemon state, pid, URL. |
| `acr code [-- ARGS]` | Boot the daemon and launch `claude` through it. |
| `acr wrap AGENT [-- ARGS]` | Launch an agent through the proxy. |
| `acr unwrap AGENT` | Remove persistent agent wrap state. |
| `acr desktop on/off/status` | Manage the persistent `ANTHROPIC_BASE_URL`. |
| `acr savings` | Aggregated routing-savings report. |
| `acr serve` | Run in the foreground (used by the daemon). |
| `acr version` | Print the version. |

---

## Contributing

See `CONTRIBUTING.md` for dev setup, `CODE_OF_CONDUCT.md` for community
standards, and `SECURITY.md` for reporting vulnerabilities. Run `make qa`
(lint + type + tests, all blocking) before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
