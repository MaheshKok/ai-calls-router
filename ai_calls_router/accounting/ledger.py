"""Savings ledger reading, aggregation, and report formatting.

Ported from Headroom's tool-router router_savings.py with provider-neutral
wording. load_entries reads the JSONL ledger tolerantly (corrupt lines are
skipped, never fatal), aggregate rolls entries up into totals plus a
per-routed-model breakdown, and format_report renders the human summary
behind the acr savings command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

from ai_calls_router._lib import config
from ai_calls_router._lib.types import JsonObject, JsonValue

_INT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_NONNEGATIVE_INT_FIELDS = ("shrink_chars_before", "shrink_chars_after")
_FLOAT_FIELDS = ("routed_usd", "premium_usd", "saved_usd")

LedgerEntry: TypeAlias = JsonObject
Bucket: TypeAlias = dict[str, int | float]
Summary: TypeAlias = dict[str, Bucket | dict[str, Bucket]]


def _json_int(value: JsonValue) -> int:
    """Coerce a ledger JSON value to int, defaulting malformed values to zero."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _json_float(value: JsonValue) -> float:
    """Coerce a ledger JSON value to float, defaulting malformed values to zero."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def load_entries(ledger: Path | None = None) -> list[LedgerEntry]:
    """Read all valid JSONL entries from the savings ledger.

    Blank lines, unparseable lines, and non-dict JSON values are skipped.

    Args:
        ledger: Ledger path override; defaults to config.ledger_path().

    Returns:
        Ledger entries in file order; empty list when the file is missing.
    """
    ledger = ledger if ledger is not None else config.ledger_path()
    if not ledger.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = cast("JsonValue", json.loads(stripped))
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _accumulate(bucket: Bucket, entry: LedgerEntry) -> None:
    """Add one entry's counters into an aggregation bucket in place.

    Args:
        bucket: Mutable totals dict for one model or the grand total.
        entry: A single ledger entry (missing/null fields count as zero).
    """
    bucket["requests"] += 1
    for key in _INT_FIELDS:
        bucket[key] += _json_int(entry.get(key, 0))
    for key in _NONNEGATIVE_INT_FIELDS:
        bucket[key] += max(_json_int(entry.get(key, 0)), 0)
    for key in _FLOAT_FIELDS:
        bucket[key] += _json_float(entry.get(key, 0.0))


def _empty_bucket() -> Bucket:
    """Create a zeroed aggregation bucket.

    Returns:
        Dict with all counter fields initialized to zero.
    """
    bucket: Bucket = {"requests": 0}
    for key in _INT_FIELDS:
        bucket[key] = 0
    for key in _NONNEGATIVE_INT_FIELDS:
        bucket[key] = 0
    for key in _FLOAT_FIELDS:
        bucket[key] = 0.0
    return bucket


def aggregate(entries: list[LedgerEntry]) -> Summary:
    """Roll ledger entries up into totals and a per-routed-model breakdown.

    Args:
        entries: Ledger entries as returned by load_entries.

    Returns:
        {"totals": {...counters, "savings_pct"}, "by_model": {model: counters}}
        where savings_pct is saved/premium*100 rounded to one decimal, or 0.0
        when no premium cost was accumulated.
    """
    totals = _empty_bucket()
    by_model: dict[str, Bucket] = {}
    for entry in entries:
        model = str(entry.get("routed_model", "?"))
        if model not in by_model:
            by_model[model] = _empty_bucket()
        _accumulate(totals, entry)
        _accumulate(by_model[model], entry)
    premium = totals["premium_usd"]
    totals["savings_pct"] = round(totals["saved_usd"] / premium * 100, 1) if premium > 0 else 0.0
    return {"totals": totals, "by_model": by_model}


def format_report(summary: Summary) -> str:
    """Render an aggregated savings summary as a human-readable report.

    Args:
        summary: Output of aggregate().

    Returns:
        Multi-line report text for the acr savings command.
    """
    totals = cast("Bucket", summary["totals"])
    if totals["requests"] == 0:
        return "No routed calls recorded yet (ledger empty)."
    lines = [
        "Routing savings",
        f"  Routed requests:        {totals['requests']}",
        f"  Input tokens:           {totals['input_tokens']:,}",
        f"  Output tokens:          {totals['output_tokens']:,}",
        f"  Cache hit tokens:       {totals['cache_read_input_tokens']:,}",
        f"  Cache miss tokens:      {totals['cache_creation_input_tokens']:,}",
        f"  Cost on routed models:  ${totals['routed_usd']:.4f}",
        f"  Cost if premium:        ${totals['premium_usd']:.4f}",
        f"  Saved:                  ${totals['saved_usd']:.2f} ({totals['savings_pct']}%)",
        "",
        "By routed model:",
    ]
    by_model = cast("dict[str, Bucket]", summary["by_model"])
    for model in sorted(by_model):
        bucket = by_model[model]
        lines.append(f"  {model}: {bucket['requests']} requests, saved ${bucket['saved_usd']:.2f}")
    return "\n".join(lines)
