"""Streaming reverse proxy to the selected premium upstream.

forward relays a client request to the upstream with headers preserved
(minus hop-by-hop ones; accept-encoding is forced to identity so the body
relays byte-identical), streams the response back without ever parsing it,
and always closes the upstream response -- including on client disconnect.
An unreachable upstream yields a 502 with an Anthropic-shaped error body.
"""

from __future__ import annotations

import codecs
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, cast

import httpx
from starlette.responses import Response, StreamingResponse

from ai_calls_router._lib import jsonnum
from ai_calls_router.proxy import websocket_passthrough

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue

logger = logging.getLogger("acr.passthrough")


# Client-supplied values that must not be forwarded: httpx computes correct
# host/content-length itself, and accept-encoding is forced to identity so
# the relayed bytes match the relayed headers.
HOP_BY_HOP_REQUEST_HEADERS = frozenset({"host", "content-length", "connection", "accept-encoding"})

# Upstream framing headers that no longer describe the re-chunked relay.
FRAMING_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection"}
)

UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
CHATGPT_CODEX_UPSTREAM = "https://chatgpt.com/backend-api/codex"

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

UsageSnapshot = dict[str, int]
ResponseComplete = Callable[[int, UsageSnapshot, float], None]


def filter_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop hop-by-hop headers from a client request.

    Args:
        headers: Raw client request headers.

    Returns:
        Headers safe to forward upstream; auth headers pass verbatim.
    """
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS and not key.lower().startswith("x-acr-")
    }


def filter_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop framing headers from an upstream response.

    Args:
        headers: Raw upstream response headers.

    Returns:
        Headers safe to relay to the client of the streamed response.
    """
    return {
        key: value for key, value in headers.items() if key.lower() not in FRAMING_RESPONSE_HEADERS
    }


def _is_chatgpt_codex_upstream(upstream: str) -> bool:
    return upstream.rstrip("/") == CHATGPT_CODEX_UPSTREAM


def _upstream_path(upstream: str, path: str) -> str:
    if _is_chatgpt_codex_upstream(upstream) and path == "/v1/responses":
        return "/responses"
    return path


def _request_headers_for_upstream(upstream: str, headers: Mapping[str, str]) -> dict[str, str]:
    if not _is_chatgpt_codex_upstream(upstream):
        return filter_request_headers(headers)
    codex_headers = websocket_passthrough.codex_chatgpt_headers(headers)
    if codex_headers is None:
        return filter_request_headers(headers)
    return {
        key: value
        for key, value in codex_headers
        if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS and not key.lower().startswith("x-acr-")
    }


class _UsageCapture:
    """Capture Anthropic usage metadata while preserving relayed bytes."""

    def __init__(self, content_type: str) -> None:
        self._content_type = content_type.lower()
        self._usage: UsageSnapshot = dict.fromkeys(_USAGE_KEYS, 0)
        self._is_sse = "text/event-stream" in self._content_type
        self._is_json = "json" in self._content_type
        self._json_chunks: list[bytes] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._sse_buffer = ""

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._is_sse:
            self._feed_sse(chunk)
        elif self._is_json:
            self._json_chunks.append(chunk)

    def finish(self) -> UsageSnapshot:
        if self._is_sse:
            self._feed_sse(b"", final=True)
            self._parse_sse_event(self._sse_buffer)
            self._sse_buffer = ""
        elif self._json_chunks:
            self._parse_json_payload(b"".join(self._json_chunks))
        return dict(self._usage)

    def _feed_sse(self, chunk: bytes, *, final: bool = False) -> None:
        text = self._decoder.decode(chunk, final=final)
        if not text:
            return
        self._sse_buffer = (self._sse_buffer + text).replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in self._sse_buffer:
            event, self._sse_buffer = self._sse_buffer.split("\n\n", 1)
            self._parse_sse_event(event)

    def _parse_sse_event(self, event: str) -> None:
        data_lines: list[str] = []
        for line in event.split("\n"):
            if not line.startswith("data:"):
                continue
            data = line[5:]
            data_lines.append(data.removeprefix(" "))
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            return
        try:
            payload = cast("JsonValue", json.loads(data_text))
        except json.JSONDecodeError:
            return
        self._apply_usage_payload(payload)

    def _parse_json_payload(self, body: bytes) -> None:
        try:
            payload = cast("JsonValue", json.loads(body))
        except json.JSONDecodeError:
            return
        self._apply_usage_payload(payload)

    def _apply_usage_payload(self, payload: JsonValue) -> None:
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._apply_usage(usage)
        message = payload.get("message")
        if isinstance(message, dict):
            message_usage = message.get("usage")
            if isinstance(message_usage, dict):
                self._apply_usage(message_usage)

    def _apply_usage(self, usage: JsonObject) -> None:
        for key in _USAGE_KEYS:
            if key in usage:
                self._usage[key] = jsonnum.int_value(usage.get(key, 0), minimum=0)


def _finish_capture(
    *,
    upstream_response: httpx.Response,
    capture: _UsageCapture,
    started: float,
    on_complete: ResponseComplete | None,
) -> None:
    if on_complete is None:
        return
    try:
        on_complete(
            int(getattr(upstream_response, "status_code", 0) or 0),
            capture.finish(),
            time.monotonic() - started,
        )
    except Exception as exc:
        logger.warning("acr: passthrough usage capture failed: %s", exc, exc_info=True)


async def _relay(
    upstream_response: httpx.Response,
    *,
    on_complete: ResponseComplete | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw upstream chunks, closing the upstream when done.

    The finally clause runs both on normal completion and when the client
    disconnects (the server closes this generator), so the upstream
    connection is never leaked. A transport may have buffered the body
    already (httpx MockTransport does); that buffer is served as-is since
    aiter_raw refuses consumed streams.

    Args:
        upstream_response: An httpx response opened with stream=True.
        on_complete: Optional callback invoked after a full response relay with
            status, usage counters, and elapsed seconds.

    Yields:
        Raw response bytes exactly as received.
    """
    capture = _UsageCapture(str(getattr(upstream_response, "headers", {}).get("content-type", "")))
    started = time.monotonic()
    try:
        if getattr(upstream_response, "is_stream_consumed", False):
            chunk = upstream_response.content
            capture.feed(chunk)
            yield chunk
        else:
            async for chunk in upstream_response.aiter_raw():
                capture.feed(chunk)
                yield chunk
        _finish_capture(
            upstream_response=upstream_response,
            capture=capture,
            started=started,
            on_complete=on_complete,
        )
    finally:
        await upstream_response.aclose()


def _bad_gateway(exc: Exception) -> Response:
    """Build a 502 response in the Anthropic error envelope.

    Args:
        exc: The transport error that prevented reaching the upstream.

    Returns:
        A JSON 502 response the Anthropic SDK can parse.
    """
    del exc
    body = {"type": "error", "error": {"type": "api_error", "message": "acr upstream unreachable"}}
    return Response(json.dumps(body), status_code=502, media_type="application/json")


async def forward(
    *,
    client: httpx.AsyncClient,
    upstream: str,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    query: str = "",
    on_complete: ResponseComplete | None = None,
) -> Response:
    """Relay one client request to the premium upstream, streaming.

    Args:
        client: Shared httpx client (connection pooling).
        upstream: Upstream base URL without trailing slash.
        method: HTTP method of the client request.
        path: Request path. Forwarded unchanged except ChatGPT Codex Responses
            passthrough, where `/v1/responses` maps to `/responses`.
        headers: Raw client request headers.
        body: Raw client request body bytes.
        query: Raw query string without the leading "?".
        on_complete: Optional callback invoked after a full upstream response
            relay with status, usage counters, and elapsed seconds.

    Returns:
        A StreamingResponse relaying the upstream answer, or a 502 Response
        when the upstream cannot be reached.
    """
    url = f"{upstream}{_upstream_path(upstream, path)}"
    if query:
        url = f"{url}?{query}"
    request_headers = _request_headers_for_upstream(upstream, headers)
    request_headers["accept-encoding"] = "identity"

    request = client.build_request(method, url, headers=request_headers, content=body)
    try:
        upstream_response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("acr: upstream %s %s failed: %s", method, path, exc, exc_info=True)
        return _bad_gateway(exc)

    return StreamingResponse(
        _relay(upstream_response, on_complete=on_complete),
        status_code=upstream_response.status_code,
        headers=filter_response_headers(upstream_response.headers),
    )
