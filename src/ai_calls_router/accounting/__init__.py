"""Accounting layer: the savings ledger. savings records per-call routing
savings (cache-aware, never fabricated) to the JSONL ledger, and ledger reads
and aggregates those entries for `acr savings`. Depends only on the _lib
foundation.
"""
