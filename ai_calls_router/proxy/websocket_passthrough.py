"""WebSocket passthrough for Codex ChatGPT-auth Responses traffic.

By default this module relays Codex ChatGPT-auth Responses WebSockets
transparently to `wss://chatgpt.com/backend-api/codex/responses`: every client
frame is forwarded to the backend unchanged and every backend frame is forwarded
back unchanged. Frames are never intercepted or re-routed, so the stateful
conversation (server side `previous_response_id` and function-call pairing) stays
intact. A passive `_UsageRecorder` sniffs the relayed `response.completed` frames
so the dashboard records real token usage, model, tools, and duration without
altering traffic.

When `ACR_CODEX_WS_ROUTING` is enabled and a routes loader is supplied, the
connection is instead handed to `codex_ws_router`, which observes the stream and
routes routable turns to cheaper tiers (see that module). The passthrough path
below is unchanged and remains the default.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, cast

from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from ai_calls_router._lib import config, logging_setup
from ai_calls_router.accounting import metrics
from ai_calls_router.proxy import codex_ws_router

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from websockets.typing import Subprotocol

    from ai_calls_router._lib.types import JsonObject, JsonValue

CHATGPT_CODEX_RESPONSES_WS = "wss://chatgpt.com/backend-api/codex/responses"
logger = logging.getLogger("acr.websocket")
_CHATGPT_ACCOUNT_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"
_TERMINAL_RESPONSE_TYPES = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
_FUNCTION_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})
_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "upgrade",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-protocol",
    }
)


async def forward_codex_chatgpt(
    websocket: WebSocket,
    *,
    sub_path: str = "",
    routes_loader: Callable[[], JsonObject] | None = None,
) -> None:
    """Relay a Codex ChatGPT-auth WebSocket to the ChatGPT Codex backend.

    By default the relay is transparent: client frames reach the backend
    unchanged and backend frames reach the client unchanged, so Codex's stateful
    `previous_response_id` continuation and function-call pairing are never
    disrupted. A `_UsageRecorder` observes the relayed frames to feed dashboard
    metrics; it never modifies what is sent. When `ACR_CODEX_WS_ROUTING` is
    enabled and `routes_loader` is supplied, the connection is handed to
    `codex_ws_router` for opt-in cheaper-tier routing instead.

    Args:
        websocket: Accepted Starlette WebSocket.
        sub_path: Optional `/responses/...` suffix to preserve.
        routes_loader: Assembled-routes loader required to enable routing.
    """
    headers = codex_chatgpt_headers(websocket.headers)
    if headers is None:
        await websocket.close(code=1008, reason="unsupported codex websocket auth")
        return

    subprotocols = _client_subprotocols(websocket.headers)
    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)
    target = _chatgpt_ws_url(sub_path, websocket.url.query)
    user_agent = websocket.headers.get("user-agent")
    recorder = _UsageRecorder(user_agent or "")
    if routes_loader is not None and config.codex_ws_routing_enabled():
        await _run_routed_relay(
            websocket,
            target,
            headers=headers,
            subprotocols=subprotocols,
            recorder=recorder,
            routes_loader=routes_loader,
            user_agent=user_agent,
        )
        return
    first_msg = await _receive_first_text(websocket)
    if first_msg is None:
        return
    recorder.note_request(first_msg)
    try:
        async with connect(
            target,
            additional_headers=headers,
            user_agent_header=user_agent,
            subprotocols=subprotocols or None,
        ) as upstream:
            await upstream.send(first_msg)
            await _relay_both(websocket, upstream, recorder=recorder)
    except Exception as exc:
        logger.warning("acr: codex websocket passthrough failed (%s)", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="upstream websocket failed")


async def _run_routed_relay(
    websocket: WebSocket,
    target: str,
    *,
    headers: list[tuple[str, str]],
    subprotocols: list[Subprotocol],
    recorder: _UsageRecorder,
    routes_loader: Callable[[], JsonObject],
    user_agent: str | None,
) -> None:
    """Open the upstream and hand the connection to the hybrid routing relay."""
    try:
        async with connect(
            target,
            additional_headers=headers,
            user_agent_header=user_agent,
            subprotocols=subprotocols or None,
        ) as upstream:
            await codex_ws_router.run_hybrid_relay(
                websocket,
                upstream,
                recorder=recorder,
                chatgpt_headers=headers,
                routes_loader=routes_loader,
                user_agent=user_agent or "",
                agent=metrics.identify_agent(user_agent or ""),
            )
    except Exception as exc:
        logger.warning("acr: codex websocket routed relay failed (%s)", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="codex routed relay failed")


def codex_chatgpt_headers(headers: Mapping[str, str]) -> list[tuple[str, str]] | None:
    """Build upstream headers and detect ChatGPT OAuth.

    Args:
        headers: Client WebSocket request headers.

    Returns:
        Forwardable headers, or `None` when the request is not ChatGPT-auth.
    """
    account_id = headers.get("chatgpt-account-id") or _account_id_from_auth(headers)
    if not account_id:
        return None
    result = _forwardable_headers(headers)
    if not _has_header(result, "chatgpt-account-id"):
        result.append(("ChatGPT-Account-ID", account_id))
    if not _has_header(result, "originator"):
        result.append(("originator", "codex_cli_rs"))
    return result


def response_create_to_http_body(raw_msg: str) -> JsonObject | None:
    """Normalize a Codex WS `response.create` frame into an HTTP Responses body.

    Args:
        raw_msg: A WebSocket text frame from Codex.

    Returns:
        A copied HTTP body with `stream: true`, or `None` for non-create frames.
    """
    try:
        parsed = cast("JsonValue", json.loads(raw_msg))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("type") != "response.create":
        return None
    response = parsed.get("response")
    if isinstance(response, dict):
        body = dict(response)
    else:
        body = {key: value for key, value in parsed.items() if key != "type"}
    if body.get("type") in {"response.create", "response"}:
        body.pop("type", None)
    return {**body, "stream": True}


class _UsageRecorder:
    """Passively records dashboard metrics from relayed Codex WS frames.

    A Codex WebSocket carries many request/response cycles. Each client
    `response.create` frame refreshes the pending request context (model, tool
    names, start time); each terminal backend frame (`response.completed`,
    `response.failed`, `response.incomplete`) emits one metrics row with the
    real token usage parsed from that frame. The recorder never mutates frames.
    """

    def __init__(self, user_agent: str) -> None:
        """Bind the recorder to one WebSocket connection's client identity.

        Args:
            user_agent: Client User-Agent header, used to identify the agent.
        """
        self._user_agent = user_agent
        self._agent = metrics.identify_agent(user_agent)
        self._model = ""
        self._tool_names: list[str] = []
        self._started: float | None = None

    def note_request(self, raw_msg: str) -> None:
        """Capture request context from a client `response.create` frame.

        Args:
            raw_msg: A client WebSocket text frame; non-create frames are ignored.
        """
        body = response_create_to_http_body(raw_msg)
        if body is None:
            return
        model = body.get("model")
        self._model = model if isinstance(model, str) else ""
        self._tool_names = _request_tool_names(body)
        self._started = time.monotonic()

    def note_response(self, raw_msg: str) -> None:
        """Record one metrics row when a backend frame ends a response.

        Args:
            raw_msg: A backend WebSocket text frame; non-terminal frames are
                ignored.
        """
        try:
            parsed = cast("JsonValue", json.loads(raw_msg))
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict):
            return
        msg_type = parsed.get("type")
        if msg_type not in _TERMINAL_RESPONSE_TYPES:
            return
        response = parsed.get("response")
        self._record(str(msg_type), response if isinstance(response, dict) else {})

    def _record(self, msg_type: str, response: JsonObject) -> None:
        """Persist one WebSocket request row from a terminal response frame."""
        input_tokens, output_tokens, cache_read, cache_creation = _usage_from_response(response)
        model = response.get("model")
        resolved_model = model if isinstance(model, str) and model else self._model
        tool_names = _response_tool_names(response) or self._tool_names
        duration = time.monotonic() - self._started if self._started is not None else 0.0
        status = 200 if msg_type == "response.completed" else 400
        try:
            m = metrics.get_metrics()
            m.incr_total()
            m.incr_passthrough()
            m.record_request(
                method="WS",
                path="/v1/responses",
                status=status,
                tier="premium",
                route="passthrough",
                model=resolved_model,
                user_agent=self._user_agent,
                client_ip="",
                tool_names=tool_names,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read=cache_read,
                cache_creation=cache_creation,
                duration=duration,
                premium_model=resolved_model,
                agent=self._agent,
                session_id="",
                decision_reason="codex_passthrough",
                request_id=logging_setup.current_request_id(),
            )
        except Exception as exc:
            logger.warning("acr: websocket metrics record failed (%s); continuing", exc)
        self._started = None


def _usage_from_response(response: JsonObject) -> tuple[int, int, int, int]:
    """Return Responses usage as cache-miss input, output, cache-read, cache-create."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    cache_read = 0
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        cache_read = _non_negative_int(details.get("cached_tokens"))
    total_input = _non_negative_int(usage.get("input_tokens"))
    output_tokens = _non_negative_int(usage.get("output_tokens"))
    return max(total_input - cache_read, 0), output_tokens, cache_read, 0


def _non_negative_int(value: JsonValue) -> int:
    """Return a non-negative int for a usage field, treating bad input as zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _request_tool_names(body: JsonObject) -> list[str]:
    """Return function-call tool names referenced in a request body's input."""
    items = body.get("input")
    if not isinstance(items, list):
        return []
    return _tool_names_from_items(items)


def _response_tool_names(response: JsonObject) -> list[str]:
    """Return function-call tool names the model emitted in a response output."""
    items = response.get("output")
    if not isinstance(items, list):
        return []
    return _tool_names_from_items(items)


def _tool_names_from_items(items: list[JsonValue]) -> list[str]:
    """Return ordered, de-duplicated function-call names from response items."""
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in _FUNCTION_CALL_TYPES:
            continue
        name = item.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


async def _relay_both(
    websocket: WebSocket,
    upstream: ClientConnection,
    *,
    recorder: _UsageRecorder,
) -> None:
    client_task = asyncio.create_task(_client_to_upstream(websocket, upstream, recorder))
    upstream_task = asyncio.create_task(_upstream_to_client(websocket, upstream, recorder))
    _done, pending = await asyncio.wait(
        {client_task, upstream_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    results = await asyncio.gather(client_task, upstream_task, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


async def _client_to_upstream(
    websocket: WebSocket,
    upstream: ClientConnection,
    recorder: _UsageRecorder,
) -> None:
    try:
        async for message in websocket.iter_text():
            recorder.note_request(message)
            await upstream.send(message)
    except (ConnectionClosed, RuntimeError, WebSocketDisconnect):
        return
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()


async def _upstream_to_client(
    websocket: WebSocket, upstream: ClientConnection, recorder: _UsageRecorder
) -> None:
    try:
        async for message in upstream:
            if isinstance(message, str):
                recorder.note_response(message)
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)
    except (ConnectionClosed, RuntimeError, WebSocketDisconnect):
        return


def _forwardable_headers(headers: Mapping[str, str]) -> list[tuple[str, str]]:
    return [(key, value) for key, value in headers.items() if key.lower() not in _HOP_HEADERS]


async def _receive_first_text(websocket: WebSocket) -> str | None:
    try:
        return await websocket.receive_text()
    except WebSocketDisconnect:
        return None


def _client_subprotocols(headers: Mapping[str, str]) -> list[Subprotocol]:
    raw_protocol = headers.get("sec-websocket-protocol", "")
    return [
        cast("Subprotocol", protocol.strip())
        for protocol in raw_protocol.split(",")
        if protocol.strip()
    ]


def _has_header(headers: Iterable[tuple[str, str]], name: str) -> bool:
    expected = name.lower()
    return any(key.lower() == expected for key, _ in headers)


def _account_id_from_auth(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = _decode_jwt_payload(parts[1])
    if not isinstance(payload, dict):
        return None
    account_id = payload.get(_CHATGPT_ACCOUNT_CLAIM)
    return account_id if isinstance(account_id, str) and account_id else None


def _decode_jwt_payload(segment: str) -> JsonValue:
    padded = segment + "=" * (-len(segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return cast("JsonValue", json.loads(decoded))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


def _chatgpt_ws_url(sub_path: str, query: str) -> str:
    clean_path = sub_path.strip("/")
    url = CHATGPT_CODEX_RESPONSES_WS
    if clean_path:
        url = f"{url}/{clean_path}"
    if query:
        url = f"{url}?{query}"
    return url
