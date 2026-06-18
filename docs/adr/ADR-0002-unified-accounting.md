# ADR-0002: Unified Accounting Ledger

## Status

Accepted.

## Context

ADR-0001 establishes that routed Codex and routed Anthropic/DeepSeek calls share the routed recording tail while preserving the Codex byte-deterministic bypass. That shared tail needs one durable accounting source so live dashboard counters, savings reports, and restart bootstrap do not drift.

The durable source is the JSONL savings ledger. `accounting/savings.py:265-270` is the write site: it creates the parent directory, appends one JSON object per routed savings record, and sets the ledger file mode to `0o600`. `accounting/ledger.py:33-58` reads that JSONL file tolerantly and preserves valid entries in file order, while `accounting/ledger.py:104-114` aggregates those entries into totals and per-routed-model buckets.

## Decision

Use the append-only JSONL ledger as the source of truth for routed accounting. Keep in-memory metrics as a serving/dashboard view, but derive historical startup state from `ledger.aggregate` instead of maintaining a second aggregation implementation. `accounting/metrics.py:335-357` loads valid ledger records, delegates historical totals to `ledger.aggregate`, and rebuilds recent rows through the shared `_request_entry` shape at `accounting/metrics.py:45-75`.

Keep the ledger owner-readable and owner-writable only. The file can contain model names, user-agent labels, tool names, token counts, and derived costs; `0o600` prevents accidental local disclosure without adding a secrets system or changing the append-only format.

## Consequences

Restart replay is deterministic: replaying the ledger through `ledger.aggregate` reproduces historical routed totals, and process-local counters only add deltas recorded after bootstrap. `accounting/metrics.py:361-405` returns copied dashboard rows, so snapshot consumers cannot mutate live recent-request state.

Accounting remains passive. Routing decisions do not read accounting state, and direct provider paths still serve already-chosen tiers rather than deciding routing policy. Future accounting changes must preserve the append-only ledger format or include an explicit migration plan.
