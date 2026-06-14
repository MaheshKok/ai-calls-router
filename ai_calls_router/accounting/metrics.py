"""In-process metrics counters and the /metrics endpoint for the proxy.

The Metrics singleton is a thread-safe registry of counters incremented by the
server path.  Every handler feeds it and  GET /metrics  renders the current
snapshot as JSON for dashboard consumption.  This is the only shared mutable
state the proxy carries (aside from the mtime config cache).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# ── singleton ──────────────────────────────────────────────────────────────


def _parse_tool_names(value: Any) -> list[str]:
    """Normalize persisted tool-name metadata from the savings ledger."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _parse_savings_line(line: str) -> dict[str, Any] | None:
    """Parse one JSONL line into a valid savings record, or None on error."""
    try:
        rec = json.loads(line)
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
        # Latest per-request metadata for the recent-activity table
        self._last_requests: list[dict[str, Any]] = []

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
    ) -> None:
        entry: dict[str, Any] = {
            "ts": int(time.time()),
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
            "client_ip": client_ip,
            "tool_names": tool_names,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
            "duration_ms": int(duration * 1000),
        }
        with self._lock:
            self._last_requests.insert(0, entry)
            if len(self._last_requests) > 100:
                self._last_requests = self._last_requests[:100]

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
            self.recent: list[dict[str, Any]] = []

        def process_one(self, rec: dict[str, Any]) -> None:
            self.routed_count += 1
            routed_model = str(rec.get("routed_model") or "")
            premium_model = str(rec.get("premium_model") or "")
            input_tokens = int(rec.get("input_tokens") or 0)
            output_tokens = int(rec.get("output_tokens") or 0)
            cache_read = int(rec.get("cache_read_input_tokens") or 0)
            cache_creation = int(rec.get("cache_creation_input_tokens") or 0)

            self.routed_input += input_tokens
            self.routed_output += output_tokens
            self.routed_cache_read += cache_read
            self.routed_cache_creation += cache_creation
            self.routed_usd += float(rec.get("routed_usd") or 0.0)
            self.premium_usd += float(rec.get("premium_usd") or 0.0)
            self.saved_usd += float(rec.get("saved_usd") or 0.0)

            self.recent.append(
                {
                    "ts": int(rec.get("ts") or 0),
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
                    "client_ip": "",
                    "tool_names": _parse_tool_names(rec.get("tool_names")),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read,
                    "cache_creation_tokens": cache_creation,
                    "duration_ms": 0,
                }
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
        acc.recent.sort(key=lambda item: int(item.get("ts") or 0), reverse=True)
        with self._lock:
            self._routed_requests += acc.routed_count
            self._routed_input_tokens += acc.routed_input
            self._routed_output_tokens += acc.routed_output
            self._routed_cache_read_tokens += acc.routed_cache_read
            self._routed_cache_creation_tokens += acc.routed_cache_creation
            self._routed_usd += acc.routed_usd
            self._premium_usd += acc.premium_usd
            self._saved_usd += acc.saved_usd
            self._last_requests = acc.recent[:max_recent] + self._last_requests
            self._last_requests = self._last_requests[:100]

    # ── snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "requests": {
                    "total": self._total_requests,
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
                "last_requests": self._last_requests[:50],
            }


_METRICS: _Metrics | None = None
_METRICS_INIT_LOCK = threading.Lock()

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


def _looks_like_session_opener(messages: list[Any]) -> bool:
    first = messages[0]
    if not isinstance(first, dict):
        return False
    if first.get("role") != "user":
        return False
    return not _message_has_tool_result(first)


def _message_has_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content)


def _stable_message_projection(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        projected = [_stable_content_part(part) for part in content]
    else:
        projected = content
    return {"role": message.get("role"), "content": projected}


def _stable_content_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    return {"type": part.get("type")}


def session_fingerprint(messages: Any) -> str | None:
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
    first = messages[0]
    material = repr(_stable_message_projection(first))
    return hashlib.blake2s(material.encode("utf-8"), digest_size=8).hexdigest()


def get_metrics() -> _Metrics:
    """Return the singleton (created on first call)."""
    global _METRICS  # noqa: PLW0603
    if _METRICS is None:
        with _METRICS_INIT_LOCK:
            if _METRICS is None:
                _METRICS = _Metrics()
    return _METRICS
