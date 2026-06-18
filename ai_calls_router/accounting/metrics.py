"""In-process metrics counters and the /metrics endpoint for the proxy.

The Metrics singleton is a thread-safe registry of counters incremented by the
server path.  Every handler feeds it and  GET /metrics  renders the current
snapshot as JSON for dashboard consumption.  This is the only shared mutable
state the proxy carries (aside from the mtime config cache).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, TypeAlias, cast

from ai_calls_router._lib import config, jsonnum
from ai_calls_router.accounting import ledger as savings_ledger
from ai_calls_router.accounting.shrink_stats import ShrinkStats

if TYPE_CHECKING:
    from pathlib import Path

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

DbValue: TypeAlias = None | int | float | str | bytes
DbRow: TypeAlias = tuple[DbValue, ...]

logger = logging.getLogger("acr.metrics")

_REQUEST_EVENT_CREATE = """
CREATE TABLE IF NOT EXISTS request_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status INTEGER NOT NULL,
    tier TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL,
    premium_model TEXT NOT NULL,
    provider TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    agent TEXT NOT NULL,
    session_id TEXT NOT NULL,
    decision_reason TEXT NOT NULL,
    client_ip TEXT NOT NULL,
    tool_names TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_creation_tokens INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    shrink_chars_before INTEGER NOT NULL,
    shrink_chars_after INTEGER NOT NULL
)
"""
_REQUEST_EVENT_INSERT = """
INSERT INTO request_events (
    ts, request_id, method, path, status, tier, route, model, premium_model,
    provider, user_agent, agent, session_id, decision_reason, client_ip,
    tool_names, input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, duration_ms, shrink_chars_before, shrink_chars_after
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_REQUEST_EVENT_RECENT = """
SELECT
    ts, request_id, method, path, status, tier, route, model, premium_model,
    provider, user_agent, agent, session_id, decision_reason, client_ip,
    tool_names, input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, duration_ms, shrink_chars_before, shrink_chars_after
FROM request_events
ORDER BY ts DESC, id DESC
LIMIT ?
"""

# ── singleton ──────────────────────────────────────────────────────────────


def _parse_tool_names(value: JsonValue) -> list[str]:
    """Normalize persisted tool-name metadata from the savings ledger."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        if value.strip().startswith("["):
            try:
                parsed = cast("JsonValue", json.loads(value))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _is_savings_record(rec: savings_ledger.LedgerEntry) -> bool:
    """Return True when a ledger record has the savings fields metrics need."""
    premium_model = str(rec.get("premium_model") or "")
    routed_model = str(rec.get("routed_model") or "")
    return bool(premium_model and routed_model)


def _request_entry(
    *,
    ts: int,
    method: str,
    path: str,
    status: int,
    tier: str,
    route: str,
    model: str,
    user_agent: str,
    client_ip: str,
    tool_names: list[str],
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    duration_ms: int,
    premium_model: str = "",
    agent: str = "",
    session_id: str = "",
    provider: str = "",
    decision_reason: str = "",
    request_id: str = "",
    shrink_chars_before: int = 0,
    shrink_chars_after: int = 0,
) -> JsonObject:
    """Build one dashboard recent-request row."""
    return cast(
        "JsonObject",
        {
            "ts": ts,
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "tier": tier,
            "route": route,
            "model": model,
            "premium_model": premium_model,
            "provider": provider or identify_provider(model),
            "user_agent": user_agent[:200] if user_agent else "",
            "agent": agent,
            "session_id": session_id,
            "decision_reason": decision_reason,
            "client_ip": client_ip,
            "tool_names": tool_names,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "duration_ms": duration_ms,
            "shrink_chars_before": max(int(shrink_chars_before), 0),
            "shrink_chars_after": max(int(shrink_chars_after), 0),
        },
    )


def _entry_text(entry: JsonObject, key: str) -> str:
    """Return one entry field as DB-safe text."""
    value = entry.get(key)
    return value if isinstance(value, str) else str(value or "")


def _entry_int(entry: JsonObject, key: str) -> int:
    """Return one entry field as DB-safe integer."""
    return jsonnum.int_value(entry.get(key, 0))


def _entry_tool_names_json(entry: JsonObject) -> str:
    """Return tool names as compact JSON text."""
    return json.dumps(_parse_tool_names(entry.get("tool_names", [])), separators=(",", ":"))


def _db_row_text(row: DbRow, index: int) -> str:
    """Return a SQLite row value as text."""
    value = row[index]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value or "")


def _db_row_int(row: DbRow, index: int) -> int:
    """Return a SQLite row value as integer."""
    value = row[index]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return jsonnum.int_value(value)
    return 0


def _entry_db_values(entry: JsonObject) -> tuple[DbValue, ...]:
    """Return DB insert values for one request-entry row."""
    return (
        _entry_int(entry, "ts"),
        _entry_text(entry, "request_id"),
        _entry_text(entry, "method"),
        _entry_text(entry, "path"),
        _entry_int(entry, "status"),
        _entry_text(entry, "tier"),
        _entry_text(entry, "route"),
        _entry_text(entry, "model"),
        _entry_text(entry, "premium_model"),
        _entry_text(entry, "provider"),
        _entry_text(entry, "user_agent"),
        _entry_text(entry, "agent"),
        _entry_text(entry, "session_id"),
        _entry_text(entry, "decision_reason"),
        _entry_text(entry, "client_ip"),
        _entry_tool_names_json(entry),
        _entry_int(entry, "input_tokens"),
        _entry_int(entry, "output_tokens"),
        _entry_int(entry, "cache_read_tokens"),
        _entry_int(entry, "cache_creation_tokens"),
        _entry_int(entry, "duration_ms"),
        _entry_int(entry, "shrink_chars_before"),
        _entry_int(entry, "shrink_chars_after"),
    )


def _entry_from_db_row(row: DbRow) -> JsonObject:
    """Build one dashboard recent-request row from SQLite history."""
    return _request_entry(
        ts=_db_row_int(row, 0),
        request_id=_db_row_text(row, 1),
        method=_db_row_text(row, 2),
        path=_db_row_text(row, 3),
        status=_db_row_int(row, 4),
        tier=_db_row_text(row, 5),
        route=_db_row_text(row, 6),
        model=_db_row_text(row, 7),
        premium_model=_db_row_text(row, 8),
        provider=_db_row_text(row, 9),
        user_agent=_db_row_text(row, 10),
        agent=_db_row_text(row, 11),
        session_id=_db_row_text(row, 12),
        decision_reason=_db_row_text(row, 13),
        client_ip=_db_row_text(row, 14),
        tool_names=_parse_tool_names(_db_row_text(row, 15)),
        input_tokens=_db_row_int(row, 16),
        output_tokens=_db_row_int(row, 17),
        cache_read=_db_row_int(row, 18),
        cache_creation=_db_row_int(row, 19),
        duration_ms=_db_row_int(row, 20),
        shrink_chars_before=_db_row_int(row, 21),
        shrink_chars_after=_db_row_int(row, 22),
    )


def _recent_entry_from_ledger_record(rec: savings_ledger.LedgerEntry) -> JsonObject:
    """Build one dashboard recent-request row from a savings ledger record."""
    routed_model = str(rec.get("routed_model") or "")
    return _request_entry(
        ts=jsonnum.int_value(rec.get("ts", 0)),
        method="POST",
        path="/v1/messages",
        status=200,
        tier=str(rec.get("tier_name") or ""),
        route="routed",
        model=routed_model,
        premium_model=str(rec.get("premium_model") or ""),
        provider=str(rec.get("provider") or identify_provider(routed_model)),
        user_agent=str(rec.get("user_agent") or ""),
        agent=str(rec.get("agent") or ""),
        session_id=str(rec.get("session_id") or ""),
        decision_reason=str(rec.get("decision_reason") or "routed"),
        client_ip="",
        tool_names=_parse_tool_names(rec.get("tool_names", "")),
        input_tokens=jsonnum.int_value(rec.get("input_tokens", 0)),
        output_tokens=jsonnum.int_value(rec.get("output_tokens", 0)),
        cache_read=jsonnum.int_value(rec.get("cache_read_input_tokens", 0)),
        cache_creation=jsonnum.int_value(rec.get("cache_creation_input_tokens", 0)),
        duration_ms=0,
        shrink_chars_before=jsonnum.int_value(rec.get("shrink_chars_before", 0)),
        shrink_chars_after=jsonnum.int_value(rec.get("shrink_chars_after", 0)),
    )


def _ensure_request_db(db_path: Path) -> None:
    """Create the request-history table if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(_REQUEST_EVENT_CREATE)
        db.execute("CREATE INDEX IF NOT EXISTS idx_request_events_ts ON request_events(ts)")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_request_events_request_id ON request_events(request_id)"
        )


def _insert_request_event(db_path: Path, entry: JsonObject) -> None:
    """Persist one request-history row."""
    _ensure_request_db(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(_REQUEST_EVENT_INSERT, _entry_db_values(entry))


def _update_request_event(
    db_path: Path,
    *,
    request_id: str,
    status: int,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    duration_ms: int,
) -> None:
    """Update latest matching persisted request-history row."""
    _ensure_request_db(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            UPDATE request_events
            SET status = ?,
                input_tokens = ?,
                output_tokens = ?,
                cache_read_tokens = ?,
                cache_creation_tokens = ?,
                duration_ms = ?
            WHERE id = (
                SELECT id FROM request_events
                WHERE request_id = ?
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (
                status,
                input_tokens,
                output_tokens,
                cache_read,
                cache_creation,
                duration_ms,
                request_id,
            ),
        )


def _load_request_events(db_path: Path, max_recent: int) -> list[JsonObject]:
    """Load newest request-history rows from SQLite."""
    if not db_path.exists():
        return []
    _ensure_request_db(db_path)
    with sqlite3.connect(db_path) as db:
        rows = cast("list[DbRow]", db.execute(_REQUEST_EVENT_RECENT, (max_recent,)).fetchall())
    return [_entry_from_db_row(row) for row in rows]


class _Metrics:
    """Thread-safe counters for the proxy."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._db_path = db_path
        # Counters
        self._total_requests = 0
        self._routed_requests = 0
        self._passthrough_requests = 0
        self._error_requests = 0
        self._escalated_requests = 0
        self._fallback_requests = 0
        # Token totals (sampled from success / routed responses)
        self._premium_input_tokens = 0
        self._premium_output_tokens = 0
        self._premium_cache_read_tokens = 0
        self._premium_cache_creation_tokens = 0
        self._routed_input_tokens = 0
        self._routed_output_tokens = 0
        self._routed_cache_read_tokens = 0
        self._routed_cache_creation_tokens = 0
        # Cost totals (fed from accounting after recording)
        self._routed_usd = 0.0
        self._premium_usd = 0.0
        self._saved_usd = 0.0
        # Cumulative tool_result compression chars (reduce + compress passes)
        self._shrink_chars_before = 0
        self._shrink_chars_after = 0
        # Latest per-request metadata for the recent-activity table
        self._last_requests: list[JsonObject] = []

    # ── counters ──────────────────────────────────────────────────────

    def incr_total(self) -> None:
        with self._lock:
            self._total_requests += 1

    def incr_routed(self) -> None:
        with self._lock:
            self._routed_requests += 1

    def incr_passthrough(self) -> None:
        with self._lock:
            self._passthrough_requests += 1

    def incr_error(self) -> None:
        with self._lock:
            self._error_requests += 1

    def incr_escalated(self) -> None:
        with self._lock:
            self._escalated_requests += 1

    def incr_fallback(self) -> None:
        with self._lock:
            self._fallback_requests += 1

    # ── token accumulation ─────────────────────────────────────────────

    def add_routed_tokens(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
    ) -> None:
        with self._lock:
            self._routed_input_tokens += input_tokens
            self._routed_output_tokens += output_tokens
            self._routed_cache_read_tokens += cache_read
            self._routed_cache_creation_tokens += cache_creation

    def add_premium_tokens(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
    ) -> None:
        with self._lock:
            self._premium_input_tokens += input_tokens
            self._premium_output_tokens += output_tokens
            self._premium_cache_read_tokens += cache_read
            self._premium_cache_creation_tokens += cache_creation

    def add_savings(
        self,
        *,
        routed_usd: float,
        premium_usd: float,
        saved_usd: float,
    ) -> None:
        with self._lock:
            self._routed_usd += routed_usd
            self._premium_usd += premium_usd
            self._saved_usd += saved_usd

    def add_shrink(self, *, chars_before: int, chars_after: int) -> None:
        """Accumulate one routed turn's tool_result compression measurement.

        Args:
            chars_before: tool_result characters before the shrink pass.
            chars_after: tool_result characters after the shrink pass.
        """
        with self._lock:
            self._shrink_chars_before += max(int(chars_before), 0)
            self._shrink_chars_after += max(int(chars_after), 0)

    # ── recent requests ────────────────────────────────────────────────

    def record_request(
        self,
        *,
        method: str,
        path: str,
        status: int,
        tier: str,
        route: str,
        model: str,
        user_agent: str,
        client_ip: str,
        tool_names: list[str],
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
        duration: float,
        premium_model: str = "",
        agent: str = "",
        session_id: str = "",
        provider: str = "",
        decision_reason: str = "",
        request_id: str = "",
        shrink_chars_before: int = 0,
        shrink_chars_after: int = 0,
    ) -> None:
        entry = _request_entry(
            ts=int(time.time()),
            method=method,
            path=path,
            status=status,
            tier=tier,
            route=route,
            model=model,
            premium_model=premium_model,
            provider=provider,
            user_agent=user_agent,
            agent=agent,
            session_id=session_id,
            decision_reason=decision_reason,
            client_ip=client_ip,
            tool_names=tool_names,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_creation=cache_creation,
            duration_ms=int(duration * 1000),
            request_id=request_id,
            shrink_chars_before=shrink_chars_before,
            shrink_chars_after=shrink_chars_after,
        )
        with self._lock:
            self._last_requests.insert(0, entry)
            if len(self._last_requests) > 100:
                self._last_requests = self._last_requests[:100]
        self._save_request_entry(entry)

    def update_request_usage(
        self,
        *,
        request_id: str,
        status: int,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
        duration: float,
    ) -> None:
        """Update an existing recent request row after a streamed response completes."""
        if not request_id:
            return
        with self._lock:
            for entry in self._last_requests:
                if entry.get("request_id") != request_id:
                    continue
                entry["status"] = status
                entry["input_tokens"] = input_tokens
                entry["output_tokens"] = output_tokens
                entry["cache_read_tokens"] = cache_read
                entry["cache_creation_tokens"] = cache_creation
                entry["duration_ms"] = int(duration * 1000)
                break
        self._update_saved_request_entry(
            request_id=request_id,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read=cache_read,
            cache_creation=cache_creation,
            duration_ms=int(duration * 1000),
        )

    def bootstrap(self, *, ledger_path: Path | None, max_recent: int = 100) -> None:
        """Restore historical routed metrics from a savings JSONL ledger.

        The savings ledger records completed routed requests. The ledger
        aggregate is the historical source of truth; in-process counters only
        add process-local deltas after bootstrap.
        """
        db_recent = self._load_saved_request_entries(max_recent)
        if ledger_path is None or not ledger_path.exists():
            if db_recent:
                with self._lock:
                    self._last_requests = db_recent + self._last_requests
                    self._last_requests = self._last_requests[:100]
            return
        try:
            entries = [
                rec for rec in savings_ledger.load_entries(ledger_path) if _is_savings_record(rec)
            ]
        except OSError:
            return
        summary = savings_ledger.aggregate(entries)
        totals = cast("savings_ledger.Bucket", summary["totals"])
        recent = db_recent or [_recent_entry_from_ledger_record(rec) for rec in entries]
        recent.sort(key=lambda item: jsonnum.int_value(item.get("ts", 0)), reverse=True)
        with self._lock:
            self._routed_requests += int(totals["requests"])
            self._routed_input_tokens += int(totals["input_tokens"])
            self._routed_output_tokens += int(totals["output_tokens"])
            self._routed_cache_read_tokens += int(totals["cache_read_input_tokens"])
            self._routed_cache_creation_tokens += int(totals["cache_creation_input_tokens"])
            self._routed_usd += float(totals["routed_usd"])
            self._premium_usd += float(totals["premium_usd"])
            self._saved_usd += float(totals["saved_usd"])
            self._shrink_chars_before += int(totals["shrink_chars_before"])
            self._shrink_chars_after += int(totals["shrink_chars_after"])
            self._last_requests = recent[:max_recent] + self._last_requests
            self._last_requests = self._last_requests[:100]

    def _save_request_entry(self, entry: JsonObject) -> None:
        """Persist one request-history row without raising into serving."""
        if self._db_path is None:
            return
        try:
            _insert_request_event(self._db_path, entry)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("acr: metrics history write failed (%s)", exc, exc_info=True)

    def _update_saved_request_entry(
        self,
        *,
        request_id: str,
        status: int,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
        duration_ms: int,
    ) -> None:
        """Persist one request-history usage update without raising into serving."""
        if self._db_path is None:
            return
        try:
            _update_request_event(
                self._db_path,
                request_id=request_id,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_creation=cache_creation,
                duration_ms=duration_ms,
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("acr: metrics history update failed (%s)", exc, exc_info=True)

    def _load_saved_request_entries(self, max_recent: int) -> list[JsonObject]:
        """Load persisted request-history rows without raising into serving."""
        if self._db_path is None:
            return []
        try:
            return _load_request_events(self._db_path, max_recent)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("acr: metrics history load failed (%s)", exc, exc_info=True)
            return []

    # ── snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> JsonObject:
        with self._lock:
            compression = ShrinkStats(
                path="cumulative",
                chars_before=self._shrink_chars_before,
                chars_after=self._shrink_chars_after,
            )
            return cast(
                "JsonObject",
                {
                    "uptime_seconds": round(time.time() - self._started_at, 1),
                    "requests": {
                        "total": self._routed_requests + self._passthrough_requests,
                        "routed": self._routed_requests,
                        "passthrough": self._passthrough_requests,
                        "errors": self._error_requests,
                        "escalated": self._escalated_requests,
                        "fallback": self._fallback_requests,
                    },
                    "routed_tokens": {
                        "input": self._routed_input_tokens,
                        "output": self._routed_output_tokens,
                        "cache_read": self._routed_cache_read_tokens,
                        "cache_creation": self._routed_cache_creation_tokens,
                    },
                    "premium_tokens": {
                        "input": self._premium_input_tokens,
                        "output": self._premium_output_tokens,
                        "cache_read": self._premium_cache_read_tokens,
                        "cache_creation": self._premium_cache_creation_tokens,
                    },
                    "costs": {
                        "routed_usd": round(self._routed_usd, 8),
                        "premium_usd": round(self._premium_usd, 8),
                        "saved_usd": round(self._saved_usd, 8),
                    },
                    "compression": {
                        "chars_before": compression.chars_before,
                        "chars_after": compression.chars_after,
                        "chars_saved": compression.chars_saved,
                        "ratio": round(compression.ratio, 4),
                        "est_tokens_saved": compression.est_tokens_saved(),
                    },
                    "last_requests": copy.deepcopy(self._last_requests[:50]),
                },
            )


Metrics = _Metrics

_metrics_singleton: _Metrics | None = None
_metrics_init_lock = threading.Lock()

# ── agent/session helpers (module-level) ──────────────────────────────────


def identify_agent(user_agent: str | None) -> str:
    """Map a raw User-Agent header into a dashboard-friendly agent label."""
    raw = (user_agent or "").strip()
    if not raw:
        return "unknown"
    return _find_agent_match(raw.lower()) or raw[:80]


# Each entry: (keywords, label, require_all). require_all=True -> ALL keywords
# must be in UA; False -> ANY keyword is enough.
_AGENT_RULES: list[tuple[list[str], str, bool]] = [
    (["claude-code", "claude code", "claude-cli"], "claude-code-cli", False),
    (["claude-desktop"], "claude-desktop", False),
    (["claude/", "electron"], "claude-desktop", True),
    (["anthropic-", "anthropic/"], "api", False),
]


def _find_agent_match(ua: str) -> str | None:
    """Return the first matching agent label, or None."""
    for keywords, label, all_required in _AGENT_RULES:
        if all_required:
            if all(k in ua for k in keywords):
                return label
        elif any(k in ua for k in keywords):
            return label
    return None


def identify_provider(model: str | None) -> str:
    """Map a model identifier to a dashboard-friendly provider label.

    Returns one of: anthropic, openai, deepseek, google, aws, azure, meta,
    mistral, cohere, groq, fireworks, perplexity, together, or unknown.
    """
    if not model:
        return "unknown"
    m = model.strip().lower()
    prefix_matches = (
        ("bedrock/", "aws"),
        ("azure/", "azure"),
        ("groq/", "groq"),
        ("fireworks/", "fireworks"),
        ("together/", "together"),
        ("anthropic/", "anthropic"),
    )
    for prefix, provider in prefix_matches:
        if m.startswith(prefix):
            return provider

    family_matches = (
        (("claude",), "anthropic"),
        (("gpt", "chatgpt"), "openai"),
        (("deepseek",), "deepseek"),
        (("gemini",), "google"),
        (("llama", "meta-llama"), "meta"),
        (("mistral",), "mistral"),
        (("sonar",), "perplexity"),
    )
    if m.startswith(("o1", "o3")):
        return "openai"
    if "command" in m and ("r-plus" in m or "r." in m):
        return "cohere"
    return next(
        (
            provider
            for needles, provider in family_matches
            if any(needle in m for needle in needles)
        ),
        "unknown",
    )


def _looks_like_session_opener(messages: JsonArray) -> bool:
    first = messages[0]
    if not isinstance(first, dict):
        return False
    if first.get("role") != "user":
        return False
    return not _message_has_tool_result(first)


def _message_has_tool_result(message: JsonObject) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content)


def _stable_message_projection(message: JsonObject) -> JsonObject:
    content = message.get("content")
    if isinstance(content, list):
        projected = [_stable_content_part(part) for part in content]
    else:
        projected = content
    return {"role": message.get("role"), "content": projected}


def _stable_content_part(part: JsonValue) -> JsonValue:
    if not isinstance(part, dict):
        return part
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    return {"type": part.get("type")}


def session_fingerprint(messages: JsonValue) -> str | None:
    """Return a stable session fingerprint for opener-style message payloads.

    Claude clients do not currently send a documented session-id header to the
    Anthropic Messages API. For dashboard grouping, use the first user turn as a
    best-effort session opener and hash its shape/content. Tool-result turns are
    deliberately ignored so second+ turns do not create synthetic sessions.
    """
    if not isinstance(messages, list) or not messages:
        return None
    if not _looks_like_session_opener(messages):
        return None
    first = cast("JsonObject", messages[0])
    material = repr(_stable_message_projection(first))
    return hashlib.blake2s(material.encode("utf-8"), digest_size=8).hexdigest()


def get_metrics() -> _Metrics:
    """Return the singleton (created on first call)."""
    global _metrics_singleton  # noqa: PLW0603
    if _metrics_singleton is None:
        with _metrics_init_lock:
            if _metrics_singleton is None:
                _metrics_singleton = _Metrics(db_path=config.metrics_db_path())
    return _metrics_singleton
