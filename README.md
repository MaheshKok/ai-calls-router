# ai-calls-router

[![CI](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml/badge.svg)](https://github.com/maheshkokare/ai-calls-router/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/ai-calls-router.svg)](https://pypi.org/project/ai-calls-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)

Per-tool-result model routing proxy for Claude Code. It serves the cheap
tool-result-processing turns of a Claude Code session on any LiteLLM-supported
model (DeepSeek, Groq, Kimi, OpenRouter, and others) while keeping every
decision-making turn on your premium Anthropic model and OAuth subscription.

Claude Code spends a large share of its API turns doing mechanical work:
reading a file you just grepped, interpreting a `Bash` result, summarizing a
web fetch. Those turns do not need a frontier model. `ai-calls-router` is a
local reverse proxy that detects them and routes only those turns to a cheaper
model, transparently, while the turns that actually require reasoning go
straight to Anthropic untouched.

It is a standalone proxy, not a Claude Code plugin: Claude Code connects to it
through the standard `ANTHROPIC_BASE_URL` environment variable.

## How it works

```mermaid
flowchart TD
    CC["Claude Code<br/>ANTHROPIC_BASE_URL=http://127.0.0.1:8747"] --> SRV["acr daemon<br/>(Starlette + uvicorn)"]
    SRV --> DEC{"POST /v1/messages<br/>decide"}
    DEC -- "no pending tool results /<br/>premium or unknown tool /<br/>any error" --> PASS["Passthrough: stream to<br/>api.anthropic.com,<br/>headers untouched"]
    DEC -- "tool_result blocks mapped<br/>to a cheap tier" --> ROUTE["litellm.acompletion<br/>tier model + tier key only"]
    ROUTE -- "response calls a<br/>premium tool" --> PASS
    ROUTE -- ok --> ACCT["savings.jsonl ledger<br/>(true routed model)"]
    ACCT --> MASK["mask response model to<br/>client-requested id"]
    MASK --> OUT["synthesized Anthropic SSE<br/>or plain JSON"]
```

A request that carries `tool_result` blocks is processing the output of a tool.
The proxy resolves which tool produced each result, maps it to a serving tier
(`tools:` in the config), and routes the turn to that tier's cheap model. Every
other request -- a fresh user turn, a plain assistant reply, anything mapped to
`premium`, anything it cannot classify -- is streamed straight through to
Anthropic.

## Guarantees

These hold on every turn:

1. The response Claude Code receives always claims the model it asked for, so
   session restore and model display stay correct. Routing is invisible.
2. Your Anthropic OAuth token / `x-api-key` is never forwarded to a routed
   provider. Routed calls carry only that tier's own key. Key values are never
   logged.
3. Any failure on the routing path -- provider error, malformed config, bad
   response -- falls back to a normal Anthropic passthrough. Routing never
   breaks a turn.
4. The savings ledger records the real routed model; only the client-facing
   response is masked.
5. Cost numbers are never fabricated. A model LiteLLM cannot price is left out
   of the ledger rather than guessed at.

## Install

Requires Python 3.11 or newer.

```bash
pip install ai-calls-router
```

For local development:

```bash
make install   # creates .venv, installs the package with dev extras
make test      # run the suite
make coverage  # suite with coverage, fails under 98%
```

## Quick start

1. Create a config interactively and provide your cheap provider's API key:

   ```bash
   acr init
   export DEEPSEEK_API_KEY=sk-...        # or the key_env your config names
   ```

   `acr init` writes `~/.ai-calls-router/config.yaml`. See
   [`config.example.yaml`](config.example.yaml) for the full annotated schema.

2. Launch Claude Code through the proxy:

   ```bash
   acr code -- -p "explain this repo"
   ```

   `acr code` boots the daemon if it is not already running, sets
   `ANTHROPIC_BASE_URL` for the child process, and runs `claude` with any
   arguments you pass after `--`.

3. Check what you saved:

   ```bash
   acr savings
   ```

## Commands

| Command | Purpose |
| --- | --- |
| `acr init` | Generate `config.yaml` interactively. |
| `acr start` | Start the proxy daemon in the background. |
| `acr stop` | Stop the daemon. |
| `acr status` | Report whether the daemon is running, with its pid and URL. |
| `acr code [-- ARGS]` | Boot the daemon if needed and launch `claude` through the proxy. |
| `acr savings` | Print the aggregated routing savings report. |
| `acr serve` | Run the proxy in the foreground (used by the daemon). |
| `acr version` | Print the version. |

## Configuration

The config lives at `~/.ai-calls-router/config.yaml` (override with
`$ACR_CONFIG`). It is reloaded by mtime on each request, so edits apply without
a restart. The key sections:

- `server`: bind host, port (default 8747), and the premium upstream.
- `premium`: reserved for future premium providers; v1 accepts only
  `provider: anthropic`.
- `settings`: tier precedence, compression, and the premium-tool escalation
  guard.
- `tiers`: each cheap tier's LiteLLM model, key environment variable, token
  cap, and optional price overrides.
- `tools`: the tool-to-tier mapping (exact match, then trailing-`*` glob;
  unmapped tools route to premium).

Tier prices are optional. The ledger prices a routed model from LiteLLM's own
pricing table first; supply `input_cost_per_1m` / `output_cost_per_1m` only for
models LiteLLM does not already know. Unpriced models are omitted from the
ledger rather than estimated.

### Compression

Routed turns can be compressed before they are sent to the cheap model to keep
token costs down: the most recent messages are preserved intact and older
`tool_result` text is truncated to a character budget. If the `rtk` CLI is on
your `PATH` and `compression.use_rtk` is `auto`, it is used as the backend;
otherwise a built-in compressor runs. `rtk` is never required.

## State and environment

| Path / variable | Meaning |
| --- | --- |
| `~/.ai-calls-router/config.yaml` | Configuration (`$ACR_CONFIG` overrides). |
| `~/.ai-calls-router/savings.jsonl` | Savings ledger (`$ACR_SAVINGS_LEDGER` overrides). |
| `~/.ai-calls-router/acr.pid` | Daemon pidfile. |
| `~/.ai-calls-router/acr.log` | Daemon log. |
| `$ACR_HOME` | Overrides the state directory root. |

## Limitations

- Routed turns are buffered, not streamed: the escalation guard needs the
  complete response before it can be served, so time-to-first-token on a routed
  turn equals its full completion latency. Decision turns stream normally
  through passthrough.
- v1 supports the Anthropic passthrough only for the premium path.
- The desktop Claude app is out of scope for v1; `acr code` covers the terminal.

## Contributing

See `CONTRIBUTING.md` for development setup and contribution guidelines,
`CODE_OF_CONDUCT.md` for community standards, and `SECURITY.md` for reporting
vulnerabilities.

## License

MIT.
