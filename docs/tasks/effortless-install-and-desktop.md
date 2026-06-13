# Task: Effortless install and `acr desktop` helper

This document is a self-contained implementation spec. An engineer or LLM with no
prior context should be able to complete the task from this file alone.

## Goal

Make adopting `ai-calls-router` as frictionless as possible for two surfaces:

1. Claude Code CLI - already works through `acr code`; this task makes install and
   first-run onboarding effortless and documents the one real footgun (the
   `ANTHROPIC_BASE_URL` precedence trap).
2. Claude Code in the desktop app (and any surface that reads persistent Claude
   settings) - add an `acr desktop` command that points it at the proxy by editing
   the persistent settings file, with a clean revert path.

## Hard constraints (must follow)

Same project rules as the rest of the repo:

- Markdown: no emojis.
- Python 3.11-3.13, full type hints (built-in generics), Google-style docstrings,
  2-4 sentence module header.
- `ruff` for lint/format; Makefile targets where present.
- TDD: write spec-derived adversarial tests first; coverage gate 98% (suite holds
  100% today). Mock only external boundaries (filesystem via tmp dirs, subprocess,
  clock) - never mock internal logic.
- One significant public class/module per file; layered architecture (see below).
- Never silently swallow errors; validate at boundaries; never log or print secrets.
- Be critical and verify: parts of the desktop mechanism are unconfirmed. Do not
  ship a command that pretends to work. Gate the exact target file on the research
  step in Phase 0 and document what was verified, with evidence.

## Current state of the relevant code (cite before changing)

- CLI: `src/ai_calls_router/cli.py`
  - `build_parser()` (lines 51-72) registers subcommands via `add_subparsers`.
    Add `desktop` here.
  - `_cmd_code()` (lines 122-131) already does the CLI routing:
    ```python
    env = {**os.environ, "ANTHROPIC_BASE_URL": _listen_url()}
    result = subprocess.run(["claude", *args.claude_args], env=env)
    ```
  - `_listen_url()` (lines 75-82) returns `http://{host}:{port}` from config.
  - Handlers are dispatched through the `_HANDLERS` dict (lines 160-169). Add the
    new handler there. Note the `_AcrParser.parse_args` special-cases the `code`
    subcommand's REMAINDER; `desktop` uses normal subparsers and needs no special
    handling.
- Config paths: `src/ai_calls_router/_lib/config.py`
  - `home_dir()` -> `$ACR_HOME` or `~/.ai-calls-router`.
  - `DEFAULT_HOST = "127.0.0.1"`, `DEFAULT_PORT = 8747`.
  - `server_settings(routes)` resolves host/port/upstream.
- Daemon: `src/ai_calls_router/ops/daemon.py` - `start()`, `status()`, `stop()`.
- Layered placement rule: a new `desktop` module belongs in the `ops` layer
  (`src/ai_calls_router/ops/desktop.py`), which already holds `daemon.py` and
  `wizard.py`. It may depend on `_lib` (config) and stdlib only. Keep the CLI thin:
  `cli.py` calls into `ops.desktop`, mirroring how `_cmd_init` calls `wizard`.

## The precedence trap (must be understood and documented)

Claude Code reads `ANTHROPIC_BASE_URL` from two places: the process environment and
the `env` block of `~/.claude/settings.json`. On the maintainer's machine this file
currently contains `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` (a different proxy,
Headroom). This creates two hazards:

1. A persistent `settings.json` value can silently win over (or lose to) the env var
   that `acr code` exports - the precedence between the two is not assumed, it must
   be verified live (Phase 0).
2. A naive "it works" check is a false positive: traffic may be going to the wrong
   proxy. The only reliable confirmation is checking which proxy's log actually
   received the request.

Consequences for this task:
- `acr code` (env-var injection) and `acr desktop` (settings-file edit) can conflict
  if both are used. The docs must explain which wins and recommend one approach per
  workflow.
- Every verification step must confirm the receiving proxy by inspecting
  `~/.ai-calls-router/acr.log`, not just by observing that Claude "worked".

## Phase 0 - Research gate (do this first; do not skip)

Confirm, with evidence, before writing the desktop command:

1. Which persistent file each target surface reads for `ANTHROPIC_BASE_URL`:
   - Claude Code CLI: `~/.claude/settings.json` `env` block (believed correct;
     confirm on this machine since it already has the Headroom entry).
   - Claude Code in the desktop app / IDE extensions: confirm whether they share
     `~/.claude/settings.json` or use a separate settings store. Do not assume the
     consumer "Claude Desktop" `claude_desktop_config.json` (the MCP-server config at
     `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS) is
     relevant - that file configures MCP servers, not the API base URL. Verify before
     targeting it.
2. Precedence: with both a `settings.json` `env.ANTHROPIC_BASE_URL` and a process
   env var set to different ports, which one receives traffic? Run `acr start`, set
   one to 8747 and the other to a decoy, launch `claude -p "run a Bash echo"`, and
   read `~/.ai-calls-router/acr.log` to see which port was hit.
3. The exact JSON shape Claude expects in `settings.json` (`{ "env": { "ANTHROPIC_BASE_URL": "..." } }`)
   and whether unrelated keys in that file must be preserved (they must).

Output of Phase 0: a short findings note (commit it as a comment block in
`ops/desktop.py` or a section in the README) stating the confirmed target file path
per surface, the precedence rule, and the evidence (log lines). The design below is
written to be correct regardless of the answer by defaulting to
`~/.claude/settings.json` and exposing a `--config PATH` override.

## Sub-goal A - Effortless install and onboarding

No new code beyond docs unless Phase 0 reveals a gap. Deliverables:

1. README "Install" section: present the recommended isolated-install paths first,
   since this is a CLI tool, not a library:
   ```bash
   # Recommended: isolated install
   uv tool install ai-calls-router
   # or
   pipx install ai-calls-router

   # Or into an environment
   pip install ai-calls-router
   ```
   Keep `pip install` but lead with `uv tool` / `pipx`. Fix the Python requirement
   line to "3.11 or newer" (also tracked in the governance task).
2. A "First run" subsection with the minimal happy path:
   ```bash
   acr init                      # write ~/.ai-calls-router/config.yaml
   export DEEPSEEK_API_KEY=sk-... # or the key_env your config names
   acr code -- -p "explain this repo"
   ```
3. A "Persistent setup vs per-session" subsection contrasting:
   - `acr code` - per-session, injects `ANTHROPIC_BASE_URL` for the child only.
     Nothing persists; nothing to revert.
   - `acr desktop on` - persistent, edits Claude's settings so every Claude Code
     session (terminal, IDE, desktop) routes through the proxy until `acr desktop off`.
   - Explicit warning about the precedence trap: if `~/.claude/settings.json` already
     sets `ANTHROPIC_BASE_URL` (e.g. another proxy), say how `acr desktop` handles it
     (it backs up and overwrites; `off` restores) and that mixing `acr code` with a
     persistent setting is redundant and can confuse which proxy is active. Tell users
     to confirm via `~/.ai-calls-router/acr.log`.
4. Troubleshooting entry: "Claude seems to ignore the proxy" -> check for a
   conflicting `ANTHROPIC_BASE_URL` in `~/.claude/settings.json`, check `acr status`,
   and confirm traffic in `acr.log`.

## Sub-goal B - `acr desktop` command

A subcommand that turns persistent proxy routing on and off by editing the Claude
settings JSON, safely and reversibly.

### CLI surface

```
acr desktop on        # route Claude (persistently) through the proxy
acr desktop off       # restore the previous setting
acr desktop status    # report whether routing is currently on, and to what URL
```

Options:
- `--config PATH` - override the settings file (defaults to the Phase 0 target,
  i.e. `~/.claude/settings.json`). Required so the command works for whatever store
  Phase 0 confirms and so tests can point at a tmp file.

Implement the `on`/`off`/`status` selector as a nested value on the parsed args
(e.g. a positional `action` with choices), dispatched inside `_cmd_desktop` in
`cli.py`. Keep all file logic in `ops/desktop.py`.

### Behavior - `on`

1. Resolve the proxy URL from config (`_listen_url()` logic: `http://host:port`).
2. Resolve the settings path (`--config` or default). Create the parent dir if
   missing.
3. Read the existing JSON if present. If the file exists but is not valid JSON,
   fail with a clear error (exit 1) and do not touch it.
4. Back up the current state exactly once per "on" so `off` can restore it:
   - Record the prior value of `env.ANTHROPIC_BASE_URL`: either the string, or a
     sentinel meaning "key was absent". Store this in a sidecar managed by acr,
     e.g. `~/.ai-calls-router/desktop_backup.json`, NOT inside the user's settings
     file. The sidecar records `{ "config_path": "...", "previous": <str|null> }`.
   - If a backup already exists (routing already on), do not overwrite the original
     backup; just update the live value (idempotent `on`).
5. Set `settings["env"]["ANTHROPIC_BASE_URL"]` to the proxy URL. Preserve every
   other key in the file and in the `env` block. Write back with stable formatting
   (`json.dumps(..., indent=2)` + trailing newline).
6. Print a confirmation including the URL and a reminder that `acr` must be running
   (`acr start` / `acr status`).

### Behavior - `off`

1. Load the sidecar backup. If none exists, report "desktop routing is not enabled
   by acr" and exit 0 (nothing to do) - but if the live file still points at the
   proxy URL with no backup, report that and leave the file untouched (do not guess).
2. Restore: if `previous` was a string, set `env.ANTHROPIC_BASE_URL` back to it; if
   it was the absent-sentinel, delete the `ANTHROPIC_BASE_URL` key (and delete the
   now-empty `env` block only if acr added it and it is empty).
3. Remove the sidecar backup.
4. Preserve all other keys. Write back with the same formatting.

### Behavior - `status`

Report, without modifying anything: whether a backup sidecar exists (acr-managed
routing on/off), the current `env.ANTHROPIC_BASE_URL` value in the settings file (or
"unset"), the configured proxy URL, and whether they match. Exit 0 always; this is
informational.

### Safety and edge cases (cover each with a test)

- Settings file does not exist -> `on` creates it with `{ "env": { "ANTHROPIC_BASE_URL": url } }`.
- Settings file exists with unrelated keys (`permissions`, `model`, other `env`
  vars) -> all preserved after `on` and after `off`.
- Settings file has a different existing `ANTHROPIC_BASE_URL` -> `on` backs up the
  old value; `off` restores exactly that value (the precedence-trap case).
- `env` key absent vs `env` present without `ANTHROPIC_BASE_URL` vs present with it
  -> `off` restores the precise prior shape (absent stays absent).
- Malformed JSON in the settings file -> `on`/`off` fail loudly, file untouched.
- `on` then `on` again (idempotent) -> original backup preserved, live value updated.
- `off` with no backup -> safe no-op with a clear message.
- `--config` pointing at a tmp path -> everything operates on that file (this is how
  tests run; no global state touched).
- Never print the contents of other `env` vars (they may be secrets); only print the
  `ANTHROPIC_BASE_URL` value, which is not a secret.

### Immutability note

Per the project style, do not mutate the loaded settings dict in place across the
read/modify/write if it can be avoided cheaply; build the updated structure and write
it. A deep copy before edit keeps the original available for comparison/rollback.

## TDD test plan (write first)

Create `tests/unit/test_desktop.py` and, if a CLI-level path is added,
extend `tests/e2e/test_cli.py`. All tests use a tmp settings file via `--config` or
by monkeypatching the default path; none touch the real `~/.claude/`.

Derive cases from the contract above, not from the implementation. At minimum:

- `test_on_creates_settings_file_when_absent`
- `test_on_sets_base_url_to_configured_proxy`
- `test_on_preserves_unrelated_top_level_and_env_keys`
- `test_on_backs_up_existing_base_url_value`
- `test_on_is_idempotent_and_keeps_original_backup`
- `test_off_restores_previous_string_value`
- `test_off_removes_key_when_it_was_originally_absent`
- `test_off_with_no_backup_is_safe_noop`
- `test_malformed_settings_json_fails_without_writing`
- `test_status_reports_on_off_and_mismatch_without_mutating`
- `test_off_removes_acr_added_empty_env_block_only_when_empty`

For each, assert on the exact resulting JSON structure and the sidecar state, and
assert the file is unchanged on the failure paths (read bytes before and after).

## Acceptance criteria

- `acr desktop on|off|status` work against a `--config` tmp file in tests and against
  the real default path in a manual run.
- Round-trip safety: `on` then `off` returns the settings file byte-for-byte to its
  semantically prior state (key restored or removed; other keys identical).
- Phase 0 findings are documented with evidence (which file, which precedence, log
  proof of routing).
- README onboarding covers isolated install, first run, persistent-vs-per-session,
  and the precedence-trap troubleshooting.
- `make coverage` stays >= 98% with the new module fully covered; `make lint` and
  `ruff format --check` clean.
- No secrets printed or logged.

## Manual smoke test (the only trustworthy end-to-end check)

```bash
acr start
acr desktop on
# launch Claude Code in the target surface, run a turn with a Bash tool result
grep "tier=" ~/.ai-calls-router/acr.log    # confirm a routed turn hit THIS proxy
acr savings                                 # confirm the ledger recorded it
acr desktop off
# relaunch; confirm traffic no longer hits the acr proxy log
```

If `acr.log` shows no routed turns, routing is not actually active regardless of what
the UI shows - investigate the precedence trap before declaring success.

## Risks and open questions

- Whether the desktop app reads `~/.claude/settings.json` `env` at all is unconfirmed
  (Phase 0). If it does not, `acr desktop` still correctly configures the CLI and IDE
  surfaces; document the limitation honestly rather than claiming desktop coverage it
  does not have.
- Precedence between process env (`acr code`) and `settings.json` is unverified;
  resolve it in Phase 0 and document which approach to use when.
- Editing a user's `~/.claude/settings.json` is a destructive-ish action on a file the
  tool did not create. The backup sidecar and the loud-fail-on-malformed-JSON rule are
  what make it safe; do not weaken them.

## Recommended study/build order

`_lib/config.py` (paths) -> `cli.py` (`_cmd_code`, parser) -> Phase 0 research ->
write `tests/unit/test_desktop.py` -> implement `ops/desktop.py` -> wire `cli.py`
`desktop` subcommand -> README onboarding -> manual smoke test.
