"""Per-socket conversation store for the stateful Codex WebSocket router.

Codex talks to the ChatGPT Codex backend over a stateful WebSocket: each
``response.create`` turn is ``store: false`` yet chains to the prior turn via
``previous_response_id``, and its ``input`` carries only the new delta items
(typically a ``function_call_output``). The matching ``function_call`` and the
encrypted ``reasoning`` live server-side, referenced by that id. To route a turn
to a different (stateless) model, the router must reconstruct the FULL input by
accumulating every prior delta input and every prior streamed output item.

This module owns that state. ``CodexSession`` virtualizes ``previous_response_id``
by issuing its own response ids and mapping each to the full ordered item list as
of that response. Reconstructing the next turn is then history-plus-delta, which
restores the ``function_call``/``function_call_output`` pairing the stateless
backend requires (a missing pair is the original 400 "No tool output found for
function call"). One instance lives per WebSocket connection; state dies with it.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray

logger = logging.getLogger("acr.codex_session")


class CodexSession:
    """Reconstructs full stateless input for one Codex WebSocket conversation.

    The store maps each router-issued response id to the full ordered item list
    (all prior inputs and outputs) as of that response. Item lists are never
    mutated in place: every stored or returned list is a fresh copy, so callers
    may freely mutate results without corrupting session state.
    """

    def __init__(self) -> None:
        """Create an empty session with no recorded responses."""
        self._history: dict[str, JsonArray] = {}

    def reconstruct_input(self, previous_response_id: str | None, delta: JsonArray) -> JsonArray:
        """Return the full stateless input for a turn: prior history then delta.

        Args:
            previous_response_id: The router response id this turn continues, or
                None for the first turn of the conversation.
            delta: The new input items the client sent for this turn.

        Returns:
            A fresh list of the history recorded for ``previous_response_id``
            followed by ``delta``. An unknown (or None) id contributes no
            history, so the result is a copy of ``delta`` alone; an unknown id is
            logged because it means prior pairing context is unavailable.
        """
        if previous_response_id is None:
            return [*delta]
        history = self._history.get(previous_response_id)
        if history is None:
            logger.warning(
                "acr: codex session has no history for previous_response_id=%r; "
                "reconstructing from delta only",
                previous_response_id,
            )
            return [*delta]
        return [*history, *delta]

    def record_response(
        self, *, full_input: JsonArray, output: JsonArray, response_id: str | None = None
    ) -> str:
        """Store a completed turn and return the response id it was stored under.

        The snapshot is ``full_input`` followed by ``output``: the conversation
        state a continuation turn will build on. The returned id is what the
        client echoes back as ``previous_response_id``.

        Args:
            full_input: The full input that produced this response (history plus
                this turn's delta), as returned by ``reconstruct_input``.
            output: The streamed output items of this response (messages,
                function calls, reasoning, ...).
            response_id: The id to store under. When None (a router-served turn)
                a unique router id is generated. When given (a passthrough turn
                observed on the wire), the upstream's real response id is used so
                the client's next ``previous_response_id`` resolves to this turn.

        Returns:
            The response id the snapshot was stored under.
        """
        key = response_id if response_id is not None else f"resp_acr_{uuid.uuid4().hex}"
        self._history[key] = [*full_input, *output]
        return key

    def knows(self, response_id: str) -> bool:
        """Return whether a response id was issued by this session.

        Args:
            response_id: The id to check, typically an incoming
                ``previous_response_id``.

        Returns:
            True when this session recorded a response under ``response_id``.
        """
        return response_id in self._history
