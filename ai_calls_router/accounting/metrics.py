"""In-process metrics counters and the /metrics endpoint for the proxy.

The Metrics singleton is a thread-safe registry of counters incremented by the
server path.  Every handler feeds it and  GET /metrics  renders the current
snapshot as JSON for dashboard consumption.  This is the only shared mutable
state the proxy carries (aside from the mtime config cache).
"""

from __future__ import annotations

import threading
import time
from typing import Any

# ── singleton ──────────────────────────────────────────────────────────────


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
            "user_agent": user_agent[:200] if user_agent else "",
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


def get_metrics() -> _Metrics:
    """Return the singleton (created on first call)."""
    global _METRICS  # noqa: PLW0603
    if _METRICS is None:
        with _METRICS_INIT_LOCK:
            if _METRICS is None:
                _METRICS = _Metrics()
    return _METRICS
