"""Streaming reverse proxy to the premium Anthropic upstream.

forward relays a client request to the upstream with headers preserved
(minus hop-by-hop ones; accept-encoding is forced to identity so the body
relays byte-identical), streams the response back without ever parsing it,
and always closes the upstream response -- including on client disconnect.
An unreachable upstream yields a 502 with an Anthropic-shaped error body.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

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
        if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS
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


async def _relay(upstream_response: Any) -> AsyncIterator[bytes]:
    """Yield raw upstream chunks, closing the upstream when done.

    The finally clause runs both on normal completion and when the client
    disconnects (the server closes this generator), so the upstream
    connection is never leaked. A transport may have buffered the body
    already (httpx MockTransport does); that buffer is served as-is since
    aiter_raw refuses consumed streams.

    Args:
        upstream_response: An httpx response opened with stream=True.

    Yields:
        Raw response bytes exactly as received.
    """
    try:
        if getattr(upstream_response, "is_stream_consumed", False):
            yield upstream_response.content
        else:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
    finally:
        await upstream_response.aclose()


def _bad_gateway(exc: Exception) -> Response:
    """Build a 502 response in the Anthropic error envelope.

    Args:
        exc: The transport error that prevented reaching the upstream.

    Returns:
        A JSON 502 response the Anthropic SDK can parse.
    """
    body = {
        "type": "error",
        "error": {"type": "api_error", "message": f"acr upstream unreachable: {exc}"},
    }
    return Response(json.dumps(body), status_code=502, media_type="application/json")


async def forward(
    client: httpx.AsyncClient,
    upstream: str,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    query: str = "",
) -> Response:
    """Relay one client request to the premium upstream, streaming.

    Args:
        client: Shared httpx client (connection pooling).
        upstream: Upstream base URL without trailing slash.
        method: HTTP method of the client request.
        path: Request path (forwarded unchanged).
        headers: Raw client request headers.
        body: Raw client request body bytes.
        query: Raw query string without the leading "?".

    Returns:
        A StreamingResponse relaying the upstream answer, or a 502 Response
        when the upstream cannot be reached.
    """
    url = f"{upstream}{path}"
    if query:
        url = f"{url}?{query}"
    request_headers = filter_request_headers(headers)
    request_headers["accept-encoding"] = "identity"

    request = client.build_request(method, url, headers=request_headers, content=body)
    try:
        upstream_response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("acr: upstream %s %s failed: %s", method, path, exc)
        return _bad_gateway(exc)

    return StreamingResponse(
        _relay(upstream_response),
        status_code=upstream_response.status_code,
        headers=filter_response_headers(upstream_response.headers),
    )
