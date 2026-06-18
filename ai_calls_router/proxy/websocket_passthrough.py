"""WebSocket passthrough for Codex ChatGPT-auth Responses traffic.

This module intentionally handles only the live Codex OAuth path first:
`/v1/responses` from Codex to `wss://chatgpt.com/backend-api/codex/responses`.
HTTP routing and non-streaming conversions stay in the existing adapter stack.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import TYPE_CHECKING, cast

from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

    from websockets.typing import Subprotocol

    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

CHATGPT_CODEX_RESPONSES_WS = "wss://chatgpt.com/backend-api/codex/responses"
logger = logging.getLogger("acr.websocket")
_CHATGPT_ACCOUNT_CLAIM = "https://api.openai.com/auth.chatgpt_account_id"
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
    route_first_frame: Callable[[str, Mapping[str, str]], Awaitable[list[str] | None]]
    | None = None,
) -> None:
    """Relay a Codex ChatGPT-auth WebSocket to the ChatGPT Codex backend.

    Args:
        websocket: Accepted Starlette WebSocket.
        sub_path: Optional `/responses/...` suffix to preserve.
        route_first_frame: Optional callback that returns WebSocket messages
            when acr can serve a `response.create` frame itself.
    """
    headers = codex_chatgpt_headers(websocket.headers)
    if headers is None:
        await websocket.close(code=1008, reason="unsupported codex websocket auth")
        return

    subprotocols = _client_subprotocols(websocket.headers)
    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)
    target = _chatgpt_ws_url(sub_path, websocket.url.query)
    user_agent = websocket.headers.get("user-agent")
    call_items: dict[str, JsonObject] = {}
    first_msg = await _receive_first_text(websocket)
    if first_msg is None:
        return
    if await _try_send_routed_response(
        websocket, first_msg, route_first_frame, websocket.headers, call_items
    ):
        return
    _remember_call_items(first_msg, call_items)
    try:
        async with connect(
            target,
            additional_headers=headers,
            user_agent_header=user_agent,
            subprotocols=subprotocols or None,
        ) as upstream:
            await upstream.send(first_msg)
            await _relay_both(
                websocket,
                upstream,
                route_frame=route_first_frame,
                headers=websocket.headers,
                call_items=call_items,
            )
    except Exception as exc:
        logger.warning("acr: codex websocket passthrough failed (%s)", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="upstream websocket failed")


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
        raw_msg: First WebSocket text frame from Codex.

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


def sse_to_ws_messages(sse_bytes: bytes) -> list[str]:
    """Convert Responses SSE bytes into Codex WebSocket JSON messages.

    Args:
        sse_bytes: Synthesized Responses SSE bytes.

    Returns:
        The `data:` payloads, excluding the SSE-only `[DONE]` sentinel.
    """
    messages: list[str] = []
    for block in sse_bytes.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data and data != "[DONE]":
                messages.append(data)
    return messages


async def _relay_both(
    websocket: WebSocket,
    upstream: ClientConnection,
    *,
    route_frame: Callable[[str, Mapping[str, str]], Awaitable[list[str] | None]] | None,
    headers: Mapping[str, str],
    call_items: dict[str, JsonObject],
) -> None:
    client_task = asyncio.create_task(
        _client_to_upstream(websocket, upstream, route_frame, headers, call_items)
    )
    upstream_task = asyncio.create_task(_upstream_to_client(websocket, upstream, call_items))
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
    route_frame: Callable[[str, Mapping[str, str]], Awaitable[list[str] | None]] | None,
    headers: Mapping[str, str],
    call_items: dict[str, JsonObject],
) -> None:
    try:
        async for message in websocket.iter_text():
            if await _try_send_routed_response(
                websocket, message, route_frame, headers, call_items
            ):
                continue
            _remember_call_items(message, call_items)
            await upstream.send(message)
    except (ConnectionClosed, RuntimeError, WebSocketDisconnect):
        return
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()


async def _upstream_to_client(
    websocket: WebSocket, upstream: ClientConnection, call_items: dict[str, JsonObject]
) -> None:
    try:
        async for message in upstream:
            if isinstance(message, str):
                _remember_call_items(message, call_items)
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)
    except (ConnectionClosed, RuntimeError, WebSocketDisconnect):
        return


async def _try_send_routed_response(
    websocket: WebSocket,
    raw_msg: str,
    route_frame: Callable[[str, Mapping[str, str]], Awaitable[list[str] | None]] | None,
    headers: Mapping[str, str],
    call_items: dict[str, JsonObject],
) -> bool:
    if route_frame is None:
        return False
    routed_messages = await route_frame(_with_cached_call_items(raw_msg, call_items), headers)
    if routed_messages is None:
        return False
    for message in routed_messages:
        _remember_call_items(message, call_items)
        await websocket.send_text(message)
    return True


def _with_cached_call_items(raw_msg: str, call_items: Mapping[str, JsonObject]) -> str:
    if not call_items:
        return raw_msg
    body = response_create_to_http_body(raw_msg)
    if body is None:
        return raw_msg
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return raw_msg
    expanded = _inject_missing_call_items(cast("JsonArray", input_items), call_items)
    if expanded == input_items:
        return raw_msg
    routed_body = {**body, "input": expanded}
    return json.dumps(
        {"type": "response.create", "response": routed_body},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _inject_missing_call_items(items: JsonArray, call_items: Mapping[str, JsonObject]) -> JsonArray:
    known = _known_call_ids(items)
    expanded: JsonArray = []
    for item in items:
        if isinstance(item, dict):
            call_id = item.get("call_id")
            if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
                cached = call_items.get(str(call_id))
                if cached is not None and str(call_id) not in known:
                    expanded.append(cached)
                    known.add(str(call_id))
        expanded.append(item)
    return expanded


def _known_call_ids(items: JsonArray) -> set[str]:
    known: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("type") in {"function_call", "custom_tool_call"}:
            call_id = item.get("call_id")
            if call_id:
                known.add(str(call_id))
    return known


def _remember_call_items(raw_msg: str, call_items: dict[str, JsonObject]) -> None:
    try:
        parsed = cast("JsonValue", json.loads(raw_msg))
    except json.JSONDecodeError:
        return
    _remember_call_items_from_value(parsed, call_items)


def _remember_call_items_from_value(value: JsonValue, call_items: dict[str, JsonObject]) -> None:
    if isinstance(value, dict):
        call_item = _call_item(value)
        if call_item is not None:
            call_items[str(call_item["call_id"])] = call_item
        for child in value.values():
            _remember_call_items_from_value(child, call_items)
    elif isinstance(value, list):
        for child in value:
            _remember_call_items_from_value(child, call_items)


def _call_item(value: JsonObject) -> JsonObject | None:
    if value.get("type") not in {"function_call", "custom_tool_call"}:
        return None
    call_id = value.get("call_id")
    name = value.get("name")
    if not call_id or not name:
        return None
    return dict(value)


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
