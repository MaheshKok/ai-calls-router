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
from typing import Any

from ai_calls_router._lib import config

_INT_FIELDS = ("input_tokens", "output_tokens")
_FLOAT_FIELDS = ("routed_usd", "premium_usd", "saved_usd")


def load_entries(ledger: Path | None = None) -> list[dict[str, Any]]:
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
    entries: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _accumulate(bucket: dict[str, Any], entry: dict[str, Any]) -> None:
    """Add one entry's counters into an aggregation bucket in place.

    Args:
        bucket: Mutable totals dict for one model or the grand total.
        entry: A single ledger entry (missing/null fields count as zero).
    """
    bucket["requests"] += 1
    for key in _INT_FIELDS:
        bucket[key] += int(entry.get(key, 0) or 0)
    for key in _FLOAT_FIELDS:
        bucket[key] += float(entry.get(key, 0.0) or 0.0)


def _empty_bucket() -> dict[str, Any]:
    """Create a zeroed aggregation bucket.

    Returns:
        Dict with all counter fields initialized to zero.
    """
    bucket: dict[str, Any] = {"requests": 0}
    for key in _INT_FIELDS:
        bucket[key] = 0
    for key in _FLOAT_FIELDS:
        bucket[key] = 0.0
    return bucket


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll ledger entries up into totals and a per-routed-model breakdown.

    Args:
        entries: Ledger entries as returned by load_entries.

    Returns:
        {"totals": {...counters, "savings_pct"}, "by_model": {model: counters}}
        where savings_pct is saved/premium*100 rounded to one decimal, or 0.0
        when no premium cost was accumulated.
    """
    totals = _empty_bucket()
    by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model = str(entry.get("routed_model", "?"))
        if model not in by_model:
            by_model[model] = _empty_bucket()
        _accumulate(totals, entry)
        _accumulate(by_model[model], entry)
    premium = totals["premium_usd"]
    totals["savings_pct"] = round(totals["saved_usd"] / premium * 100, 1) if premium > 0 else 0.0
    return {"totals": totals, "by_model": by_model}


def format_report(summary: dict[str, Any]) -> str:
    """Render an aggregated savings summary as a human-readable report.

    Args:
        summary: Output of aggregate().

    Returns:
        Multi-line report text for the acr savings command.
    """
    totals = summary["totals"]
    if totals["requests"] == 0:
        return "No routed calls recorded yet (ledger empty)."
    lines = [
        "Routing savings",
        f"  Routed requests:        {totals['requests']}",
        f"  Input tokens:           {totals['input_tokens']:,}",
        f"  Output tokens:          {totals['output_tokens']:,}",
        f"  Cost on routed models:  ${totals['routed_usd']:.4f}",
        f"  Cost if premium:        ${totals['premium_usd']:.4f}",
        f"  Saved:                  ${totals['saved_usd']:.2f} ({totals['savings_pct']}%)",
        "",
        "By routed model:",
    ]
    for model in sorted(summary["by_model"]):
        bucket = summary["by_model"][model]
        lines.append(
            f"  {model}: {bucket['requests']} requests, saved ${bucket['saved_usd']:.2f}"
        )
    return "\n".join(lines)
