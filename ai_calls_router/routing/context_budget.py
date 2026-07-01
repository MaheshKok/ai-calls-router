"""Per-session context-size guard for routed Anthropic turns.

The routed tier models (Sonnet 5, Haiku 4.5) cap context at ~200K tokens; the
premium model (Opus, 1M) does not. Once a conversation grows past a tier's
window, routing that turn 400s upstream and fails open to premium -- a wasted
round-trip that repeats on every subsequent turn of a long session. This module
remembers the conversation size after each completed turn (from the usage the
API already returns -- no tokenizer) so the router can force premium up front
once a session is known to overflow the tier, paying the wasted hop at most once
per session instead of every turn.

The store is a bounded in-memory LRU keyed by the session fingerprint. It is a
latency/cost optimizer only: fail-open remains the correctness backstop, so a
missing, stale, or absent entry never breaks a turn -- it just misses one
optimization. State is process-local and lost on restart (a restarted daemon
re-learns each overflowing session on its next turn).
"""

from __future__ import annotations

from collections import OrderedDict

# Room reserved below the tier window for this turn's output plus estimate drift
# (the new user turn is not yet counted, and cross-model tokenizers differ a few
# percent). Conservative by design: tripping early costs a routable large turn;
# tripping late costs a wasted routed hop that fail-open already absorbs.
_SAFETY_MARGIN_TOKENS = 4096

# ponytail: LRU cap bounds memory (~a few KB); no TTL -- eviction reclaims idle
# sessions, and a session evicted mid-run just re-learns on its next turn. Raise
# only if a host runs thousands of concurrent long sessions.
_MAX_SESSIONS = 1024

# session fingerprint -> last observed (input_tokens + output_tokens). ponytail:
# module-global dict is safe without a lock -- record/would_overflow are fully
# synchronous (no await), so on the single asyncio event-loop thread they run to
# completion without interleaving. Add a lock only if a threaded caller appears.
_last_context_tokens: OrderedDict[str, int] = OrderedDict()


def record_context_size(session: str | None, input_tokens: int, output_tokens: int) -> None:
    """Remember the conversation size after a completed turn.

    The next turn's input is at least this turn's prompt plus its response, so
    ``input_tokens + output_tokens`` is the best zero-cost predictor of the next
    turn's input size. No-op for a missing session key or a non-positive total
    (an unparsed/errored turn carries no usable size).

    Args:
        session: Stable per-conversation fingerprint, or None to skip.
        input_tokens: Prompt tokens the completed turn consumed.
        output_tokens: Completion tokens the completed turn produced.
    """
    if not session:
        return
    total = max(int(input_tokens), 0) + max(int(output_tokens), 0)
    if total <= 0:
        return
    _last_context_tokens[session] = total
    _last_context_tokens.move_to_end(session)
    while len(_last_context_tokens) > _MAX_SESSIONS:
        _last_context_tokens.popitem(last=False)


def would_overflow(session: str | None, *, context_window: int, output_reserve: int) -> bool:
    """Return whether this session's next turn is projected to exceed the window.

    Projects the next turn's input as the last observed ``input + output`` and
    trips when that leaves no room for the turn's output within the tier's
    context window (minus a fixed safety margin). Returns False -- route normally
    -- for any unknown session or non-positive window, so a first or oversize
    turn still gets its fail-open routed attempt.

    Args:
        session: Stable per-conversation fingerprint, or None to skip the guard.
        context_window: Total token window (input+output) of the tier model.
        output_reserve: Tokens to reserve for this turn's output.

    Returns:
        True when the session should be forced to premium passthrough.
    """
    if not session or context_window <= 0:
        return False
    last_total = _last_context_tokens.get(session)
    if last_total is None:
        return False
    _last_context_tokens.move_to_end(session)
    ceiling = context_window - max(int(output_reserve), 0) - _SAFETY_MARGIN_TOKENS
    return last_total >= ceiling
