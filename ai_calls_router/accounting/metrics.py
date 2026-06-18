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
import threading
import time
from typing import TYPE_CHECKING, cast

from ai_calls_router.accounting.shrink_stats import ShrinkStats

if TYPE_CHECKING:
    from pathlib import Path

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

# ── singleton ──────────────────────────────────────────────────────────────


def _parse_tool_names(value: JsonValue) -> list[str]:
    """Normalize persisted tool-name metadata from the savings ledger."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _json_int(value: JsonValue, default: int = 0) -> int:
    """Coerce persisted JSON values to int for dashboard totals."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _json_float(value: JsonValue, default: float = 0.0) -> float:
    """Coerce persisted JSON values to float for dashboard totals."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _parse_savings_line(line: str) -> JsonObject | None:
    """Parse one JSONL line into a valid savings record, or None on error."""
    try:
        rec = cast("JsonValue", json.loads(line))
    except json.JSONDecodeError:
        return None
    if not isinstance(rec, dict):
        return None
    premium_model = str(rec.get("premium_model") or "")
    routed_model = str(rec.get("routed_model") or "")
    if not premium_model or not routed_model:
        return None
    return rec


class _Metrics:
    """Thread-safe counters for the proxy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
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
        entry = cast(
            "JsonObject",
            {
                "ts": int(time.time()),
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
                "duration_ms": int(duration * 1000),
                "shrink_chars_before": max(int(shrink_chars_before), 0),
                "shrink_chars_after": max(int(shrink_chars_after), 0),
            },
        )
        with self._lock:
            self._last_requests.insert(0, entry)
            if len(self._last_requests) > 100:
                self._last_requests = self._last_requests[:100]

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
                return

    class _BootstrapAccumulator:
        """Mutable state bag for one bootstrap replay pass."""

        def __init__(self) -> None:
            self.routed_count = 0
            self.routed_input = 0
            self.routed_output = 0
            self.routed_cache_read = 0
            self.routed_cache_creation = 0
            self.routed_usd = 0.0
            self.premium_usd = 0.0
            self.saved_usd = 0.0
            self.shrink_chars_before = 0
            self.shrink_chars_after = 0
            self.recent: list[JsonObject] = []

        def process_one(self, rec: JsonObject) -> None:
            self.routed_count += 1
            routed_model = str(rec.get("routed_model") or "")
            premium_model = str(rec.get("premium_model") or "")
            input_tokens = _json_int(rec.get("input_tokens", 0))
            output_tokens = _json_int(rec.get("output_tokens", 0))
            cache_read = _json_int(rec.get("cache_read_input_tokens", 0))
            cache_creation = _json_int(rec.get("cache_creation_input_tokens", 0))

            self.routed_input += input_tokens
            self.routed_output += output_tokens
            self.routed_cache_read += cache_read
            self.routed_cache_creation += cache_creation
            self.routed_usd += _json_float(rec.get("routed_usd", 0.0))
            self.premium_usd += _json_float(rec.get("premium_usd", 0.0))
            self.saved_usd += _json_float(rec.get("saved_usd", 0.0))
            self.shrink_chars_before += max(_json_int(rec.get("shrink_chars_before", 0)), 0)
            self.shrink_chars_after += max(_json_int(rec.get("shrink_chars_after", 0)), 0)

            self.recent.append(
                self._build_recent_entry(
                    rec,
                    routed_model=routed_model,
                    premium_model=premium_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read=cache_read,
                    cache_creation=cache_creation,
                )
            )

        @staticmethod
        def _build_recent_entry(
            rec: JsonObject,
            *,
            routed_model: str,
            premium_model: str,
            input_tokens: int,
            output_tokens: int,
            cache_read: int,
            cache_creation: int,
        ) -> JsonObject:
            """Build one dashboard recent-request row from a ledger record.

            Args:
                rec: One parsed savings-ledger record.
                routed_model: Pre-coerced routed model name.
                premium_model: Pre-coerced premium model name.
                input_tokens: Pre-coerced input token count.
                output_tokens: Pre-coerced output token count.
                cache_read: Pre-coerced cache-read token count.
                cache_creation: Pre-coerced cache-creation token count.

            Returns:
                A recent-request dict matching the live dashboard row shape.
            """
            return cast(
                "JsonObject",
                {
                    "ts": _json_int(rec.get("ts", 0)),
                    "method": "POST",
                    "path": "/v1/messages",
                    "status": 200,
                    "tier": str(rec.get("tier_name") or ""),
                    "route": "routed",
                    "model": routed_model,
                    "premium_model": premium_model,
                    "provider": str(rec.get("provider") or identify_provider(routed_model)),
                    "user_agent": str(rec.get("user_agent") or "")[:200],
                    "agent": str(rec.get("agent") or ""),
                    "session_id": str(rec.get("session_id") or ""),
                    "decision_reason": str(rec.get("decision_reason") or "routed"),
                    "client_ip": "",
                    "tool_names": _parse_tool_names(rec.get("tool_names", "")),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read,
                    "cache_creation_tokens": cache_creation,
                    "duration_ms": 0,
                    "shrink_chars_before": max(_json_int(rec.get("shrink_chars_before", 0)), 0),
                    "shrink_chars_after": max(_json_int(rec.get("shrink_chars_after", 0)), 0),
                },
            )

    def bootstrap(self, *, ledger_path: Path | None, max_recent: int = 100) -> None:
        """Restore historical routed metrics from a savings JSONL ledger.

        The savings ledger records completed routed requests. Replaying it gives
        the dashboard historical routed-token/cost/session context after proxy
        restarts. Request totals remain live-process counters and are not
        synthesized from the ledger.
        """
        if ledger_path is None or not ledger_path.exists():
            return
        acc = self._BootstrapAccumulator()
        try:
            with ledger_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rec = _parse_savings_line(line)
                    if rec is None:
                        continue
                    acc.process_one(rec)
        except OSError:
            return
        acc.recent.sort(key=lambda item: _json_int(item.get("ts", 0)), reverse=True)
        with self._lock:
            self._routed_requests += acc.routed_count
            self._routed_input_tokens += acc.routed_input
            self._routed_output_tokens += acc.routed_output
            self._routed_cache_read_tokens += acc.routed_cache_read
            self._routed_cache_creation_tokens += acc.routed_cache_creation
            self._routed_usd += acc.routed_usd
            self._premium_usd += acc.premium_usd
            self._saved_usd += acc.saved_usd
            self._shrink_chars_before += acc.shrink_chars_before
            self._shrink_chars_after += acc.shrink_chars_after
            self._last_requests = acc.recent[:max_recent] + self._last_requests
            self._last_requests = self._last_requests[:100]

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
                _metrics_singleton = _Metrics()
    return _metrics_singleton
