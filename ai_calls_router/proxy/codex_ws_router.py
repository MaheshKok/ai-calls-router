"""Hybrid sticky-route relay for Codex ChatGPT-auth Responses WebSockets.

The default Codex WebSocket transport stays a transparent passthrough (see
``websocket_passthrough``); this module powers the opt-in routing path gated by
``ACR_CODEX_WS_ROUTING``. It relays a Codex WebSocket to the ChatGPT Codex
backend while observing every turn into a :class:`CodexSession`. Once a turn
becomes routable to a cheaper tier, the router reconstructs the full stateless
input, serves the turn over the direct HTTP Codex path, and synthesizes the
Responses stream events back as WebSocket frames.

Because a router-issued response id is unknown to the upstream WebSocket, a
session that has routed once stays on the HTTP path for the rest of its life
(sticky route): later turns -- including premium-tier turns -- are served over
HTTP as well. The relay is turn-sequential, matching Codex's request/response
cadence.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from starlette.websockets import WebSocketDisconnect

from ai_calls_router._lib import logging_setup
from ai_calls_router.proxy.codex_session import CodexSession
from ai_calls_router.routing import codex_direct
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import engine as routed_call
from ai_calls_router.routing.adapters.openai_responses import OpenAIResponsesAdapter
from ai_calls_router.routing.synthesis_responses import synthesize_response_object_frames

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    import httpx

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue
    from ai_calls_router.accounting import shrink_stats

logger = logging.getLogger("acr.codex_ws_router")

_TERMINAL_RESPONSE_TYPES = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
_ADAPTER = OpenAIResponsesAdapter()


class _ClientSocket(Protocol):
    """The client WebSocket surface the relay uses (Starlette-compatible)."""

    async def receive_text(self) -> str: ...
    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...


class _Upstream(Protocol):
    """The upstream WebSocket surface the relay uses (websockets-compatible)."""

    async def send(self, message: str | bytes) -> None: ...
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


class _Recorder(Protocol):
    """The passthrough metrics recorder surface the relay drives."""

    def note_request(self, raw_msg: str) -> None: ...
    def note_response(self, raw_msg: str) -> None: ...


class RoutedTurnError(RuntimeError):
    """Raised when a routed turn cannot be served and the session must close.

    A sticky-routed session cannot fall back to the upstream WebSocket (its
    response ids are virtual), so an unrecoverable routed turn closes the socket
    rather than corrupt the conversation.
    """


@dataclass(frozen=True)
class WsTurnDecision:
    """Routing decision for one Codex WebSocket turn."""

    routable: bool
    tier: str
    tier_cfg: JsonObject | None
    credential: routing.TierCredential | None
    names: list[str]


def parse_response_create(raw_msg: str) -> JsonObject | None:
    """Return the flat Responses body from a ``response.create`` frame, else None.

    Mirrors ``websocket_passthrough.response_create_to_http_body`` but yields the
    raw flat body (model, input, tools, previous_response_id, instructions)
    without injecting ``stream``; kept local to avoid an import cycle.

    Args:
        raw_msg: A client WebSocket text frame.

    Returns:
        The flat request body, or None for any non-``response.create`` frame.
    """
    try:
        parsed = cast("JsonValue", json.loads(raw_msg))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("type") != "response.create":
        return None
    response = parsed.get("response")
    if isinstance(response, dict):
        return dict(response)
    return {key: value for key, value in parsed.items() if key != "type"}


def decide_ws_turn(full_input: JsonArray, *, routes: JsonObject, group: str) -> WsTurnDecision:
    """Decide whether a turn's reconstructed input routes to a cheaper tier.

    The pending tool names are resolved from the FULL reconstructed input (a
    WebSocket delta carries ``function_call_output`` without the tool name, so
    the name comes from the ``function_call`` recorded earlier in history).

    Args:
        full_input: The turn's reconstructed full input (history plus delta).
        routes: Parsed, assembled routes config.
        group: Agent group for tool-to-tier lookup (``codex``).

    Returns:
        A decision whose ``routable`` is True only when a non-premium tier has
        both a config entry and a usable credential.
    """
    names = _ADAPTER.extract_pending_tools({"input": list(full_input)})
    tier = routing.tier_for_tools(names, routes, group=group)
    if tier == "premium":
        return WsTurnDecision(False, "premium", None, None, names)
    tier_cfg = _tier_cfg(routes, tier)
    if tier_cfg is None:
        return WsTurnDecision(False, tier, None, None, names)
    credential = routing.resolve_tier_credential(tier_cfg, _settings(routes))
    if credential is None:
        return WsTurnDecision(False, tier, tier_cfg, None, names)
    return WsTurnDecision(True, tier, tier_cfg, credential, names)


async def run_hybrid_relay(
    client: _ClientSocket,
    upstream: _Upstream,
    *,
    recorder: _Recorder,
    chatgpt_headers: list[tuple[str, str]] | None,
    routes_loader: Callable[[], JsonObject],
    group: str = "codex",
    http_client: httpx.AsyncClient | None = None,
    user_agent: str = "",
    agent: str = "",
) -> None:
    """Relay one Codex WebSocket, routing routable turns to cheaper tiers.

    Turn-sequential: each client ``response.create`` is fully served (either
    passed through to the upstream backend or routed over HTTP and synthesized
    back) before the next is read. The session sticks to the HTTP path once it
    routes for the first time.

    Args:
        client: Accepted client WebSocket.
        upstream: Open upstream connection to the ChatGPT Codex backend.
        recorder: Passthrough metrics recorder for non-routed turns.
        chatgpt_headers: Forwardable ChatGPT OAuth headers for routed HTTP calls.
        routes_loader: Callable returning the assembled routes config.
        group: Agent group for tier resolution.
        http_client: Optional shared HTTP client (tests inject a mock transport).
        user_agent: Client User-Agent, for routed metrics.
        agent: Identified agent label, for routed metrics.
    """
    session = CodexSession()
    routed = False
    while True:
        try:
            raw = await client.receive_text()
        except WebSocketDisconnect:
            return
        body = parse_response_create(raw)
        if body is None:
            await upstream.send(raw)
            continue
        full_input = _reconstruct(session, body)
        decision = decide_ws_turn(full_input, routes=routes_loader(), group=group)
        if not routed and not decision.routable:
            recorder.note_request(raw)
            await upstream.send(raw)
            await _observe_passthrough_turn(upstream, client, recorder, session, full_input)
            continue
        routed = True
        await _serve_routed_turn(
            client,
            session,
            full_input,
            body,
            decision,
            chatgpt_headers=chatgpt_headers,
            http_client=http_client,
            user_agent=user_agent,
            agent=agent,
        )


def _reconstruct(session: CodexSession, body: JsonObject) -> JsonArray:
    """Return the full stateless input for a turn from session history plus delta."""
    delta = body.get("input")
    delta_items: JsonArray = delta if isinstance(delta, list) else []
    previous = body.get("previous_response_id")
    previous_id = previous if isinstance(previous, str) else None
    return session.reconstruct_input(previous_id, delta_items)


async def _observe_passthrough_turn(
    upstream: _Upstream,
    client: _ClientSocket,
    recorder: _Recorder,
    session: CodexSession,
    full_input: JsonArray,
) -> None:
    """Relay one passthrough response and record it under the upstream's real id.

    Forwards every upstream frame to the client unchanged while accumulating the
    streamed ``response.output_item.done`` items, then records the turn under the
    upstream's real response id so a later routable turn can reconstruct from it.
    """
    output_items: JsonArray = []
    real_id: str | None = None
    async for message in upstream:
        if not isinstance(message, str):
            await client.send_bytes(message)
            continue
        recorder.note_response(message)
        await client.send_text(message)
        event = _safe_loads(message)
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif event_type in _TERMINAL_RESPONSE_TYPES:
            real_id = _response_id(event)
            break
    if real_id is not None:
        session.record_response(full_input=full_input, output=output_items, response_id=real_id)


async def _serve_routed_turn(
    client: _ClientSocket,
    session: CodexSession,
    full_input: JsonArray,
    body: JsonObject,
    decision: WsTurnDecision,
    *,
    chatgpt_headers: list[tuple[str, str]] | None,
    http_client: httpx.AsyncClient | None,
    user_agent: str,
    agent: str,
) -> None:
    """Serve one turn over HTTP, synthesize WS frames, and record it for chaining."""
    requested_model = str(body.get("model") or "")
    routed_body = {**body, "input": list(full_input)}
    started = time.monotonic()
    served = await _dispatch_turn(
        routed_body, decision, requested_model, chatgpt_headers, http_client
    )
    if served is None:
        raise RoutedTurnError(decision.tier)
    response_body, usage_values, shrink, tier_name, routed_model = served
    duration = time.monotonic() - started
    output = response_body.get("output")
    output_items: JsonArray = output if isinstance(output, list) else []
    recorded_id = session.record_response(
        full_input=full_input, output=output_items, response_id=_response_id(response_body)
    )
    frames_response = {**response_body, "id": recorded_id}
    for frame in synthesize_response_object_frames(frames_response):
        await client.send_text(json.dumps(frame, ensure_ascii=False))
    await _record_routed_metrics(
        decision=decision,
        tier_name=tier_name,
        routed_model=routed_model,
        requested_model=requested_model,
        usage_values=usage_values,
        shrink=shrink,
        duration=duration,
        user_agent=user_agent,
        agent=agent,
    )


async def _dispatch_turn(
    routed_body: JsonObject,
    decision: WsTurnDecision,
    requested_model: str,
    chatgpt_headers: list[tuple[str, str]] | None,
    http_client: httpx.AsyncClient | None,
) -> tuple[JsonObject, tuple[int, int, int, int], shrink_stats.ShrinkStats, str, str] | None:
    """Serve a turn on its cheap tier, falling back to premium over HTTP.

    Once a session is sticky-routed it cannot return to the upstream WebSocket,
    so a premium-tier turn (or a failed cheap call) is served against the
    client's requested model over the same OAuth HTTP path.

    Returns:
        ``(response_body, usage, shrink, tier_name, routed_model)`` or None when
        even the premium attempt fails.
    """
    if decision.routable and decision.tier_cfg is not None and decision.credential is not None:
        cheap = await codex_direct.responses_call(
            body=routed_body,
            tier_cfg=decision.tier_cfg,
            credential=decision.credential.value,
            auth_mode=decision.credential.auth_mode,
            chatgpt_headers=chatgpt_headers,
            client=http_client,
        )
        if cheap is not None:
            model = codex_direct.native_model_id(decision.tier_cfg)
            return (*cheap, decision.tier, model)
        logger.warning("acr: codex ws routed tier=%s failed; trying premium HTTP", decision.tier)
    if not requested_model:
        return None
    premium_cfg: JsonObject = {"model": f"codex/{requested_model}", "provider": "codex"}
    premium = await codex_direct.responses_call(
        body=routed_body,
        tier_cfg=premium_cfg,
        credential="oauth",
        auth_mode="oauth",
        chatgpt_headers=chatgpt_headers,
        client=http_client,
    )
    if premium is None:
        return None
    return (*premium, "premium", requested_model)


async def _record_routed_metrics(
    *,
    decision: WsTurnDecision,
    tier_name: str,
    routed_model: str,
    requested_model: str,
    usage_values: tuple[int, int, int, int],
    shrink: shrink_stats.ShrinkStats,
    duration: float,
    user_agent: str,
    agent: str,
) -> None:
    """Record one routed WebSocket turn into shared accounting."""
    usage = routed_call.RouteUsage(
        input_tokens=usage_values[0],
        output_tokens=usage_values[1],
        cache_read_tokens=usage_values[2],
        cache_creation_tokens=usage_values[3],
    )
    await routed_call.record_route_outcome(
        routed_call.RouteOutcome(
            premium_model=requested_model,
            routed_model=routed_model,
            tier_name=tier_name,
            tier_cfg=decision.tier_cfg or {},
            tool_names=decision.names,
            usage=usage,
            request_path="/v1/responses",
            route="direct",
            user_agent=user_agent,
            agent=agent,
            session_id="",
            elapsed=duration,
            shrink=shrink,
            method="WS",
            request_id=logging_setup.current_request_id(),
        )
    )


def _safe_loads(raw_msg: str) -> JsonValue:
    """Return parsed JSON or None for a frame that is not valid JSON."""
    try:
        return cast("JsonValue", json.loads(raw_msg))
    except json.JSONDecodeError:
        return None


def _response_id(event: JsonObject) -> str | None:
    """Return the response id from a frame's response envelope, if present."""
    response = event.get("response")
    if isinstance(response, dict):
        candidate = response.get("id")
        return candidate if isinstance(candidate, str) and candidate else None
    candidate = event.get("id")
    return candidate if isinstance(candidate, str) and candidate else None


def _tier_cfg(routes: JsonObject, tier: str) -> JsonObject | None:
    """Return the config entry for a tier, or None when missing or malformed."""
    tiers = routes.get("tiers")
    if not isinstance(tiers, dict):
        return None
    tier_cfg = tiers.get(tier)
    return tier_cfg if isinstance(tier_cfg, dict) else None


def _settings(routes: JsonObject) -> JsonObject:
    """Return the settings block, defaulting to empty."""
    settings = routes.get("settings")
    return settings if isinstance(settings, dict) else {}
