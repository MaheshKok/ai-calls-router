"""Spec-derived tests for ai_calls_router.passthrough.

Contract under test: forward relays a client request to the premium upstream
with all headers preserved except hop-by-hop ones (host, content-length,
connection, accept-encoding -- replaced by identity so bytes relay 1:1),
streams the upstream body back unchanged with the upstream status and
headers (minus framing headers), returns a 502 Anthropic-shaped error when
the upstream is unreachable, and always closes the upstream response --
including when the client disconnects mid-stream.
"""

from __future__ import annotations

import httpx
from starlette.responses import Response

from ai_calls_router.proxy import passthrough as pt

UPSTREAM = "https://upstream.example"


class _Upstream:
    """Mock upstream recording requests and serving a canned response."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b'{"ok": true}',
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._status_code = status_code
        self._content = content
        self._headers = headers or {}
        self._error = error

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return httpx.Response(self._status_code, content=self._content, headers=self._headers)


def _client(upstream: _Upstream) -> httpx.AsyncClient:
    """Build an httpx client wired to the mock upstream."""
    return httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))


async def _drain(response: Response) -> bytes:
    """Consume a StreamingResponse body iterator into bytes."""
    chunks = [chunk async for chunk in response.body_iterator]
    return b"".join(chunk.encode("utf-8") if isinstance(chunk, str) else chunk for chunk in chunks)


CLIENT_HEADERS = {
    "host": "localhost:8747",
    "content-length": "999",
    "connection": "keep-alive",
    "accept-encoding": "gzip, br",
    "x-api-key": "client-secret",
    "authorization": "Bearer oauth-token",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


class TestFilterRequestHeaders:
    def test_hop_by_hop_headers_dropped(self) -> None:
        filtered = pt.filter_request_headers(CLIENT_HEADERS)
        for name in ("host", "content-length", "connection", "accept-encoding"):
            assert name not in filtered

    def test_hop_by_hop_dropped_case_insensitively(self) -> None:
        filtered = pt.filter_request_headers({"Host": "x", "Content-Length": "5"})
        assert not any(key.lower() in ("host", "content-length") for key in filtered)

    def test_auth_and_api_headers_preserved_verbatim(self) -> None:
        filtered = pt.filter_request_headers(CLIENT_HEADERS)
        assert filtered["x-api-key"] == "client-secret"
        assert filtered["authorization"] == "Bearer oauth-token"
        assert filtered["anthropic-version"] == "2023-06-01"
        assert filtered["content-type"] == "application/json"

    def test_acr_control_headers_dropped(self) -> None:
        filtered = pt.filter_request_headers({"x-acr-agent": "hermes", "authorization": "Bearer c"})
        assert "x-acr-agent" not in {key.lower(): value for key, value in filtered.items()}
        assert filtered["authorization"] == "Bearer c"


class TestForward:
    async def test_forwards_method_path_query_and_body(self) -> None:
        upstream = _Upstream()
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b'{"model": "m"}',
                query="beta=true",
            )
            await _drain(response)
        request = upstream.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/v1/messages"
        assert request.url.query == b"beta=true"
        assert request.content == b'{"model": "m"}'

    async def test_client_auth_headers_reach_upstream(self) -> None:
        upstream = _Upstream()
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
            await _drain(response)
        request = upstream.requests[0]
        assert request.headers["x-api-key"] == "client-secret"
        assert request.headers["authorization"] == "Bearer oauth-token"

    async def test_client_hop_by_hop_values_replaced(self) -> None:
        upstream = _Upstream()
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
            await _drain(response)
        request = upstream.requests[0]
        assert request.headers["host"] == "upstream.example"
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["content-length"] == "2"

    async def test_upstream_status_relayed(self) -> None:
        upstream = _Upstream(status_code=429, content=b'{"type": "error"}')
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
            body = await _drain(response)
        assert response.status_code == 429
        assert body == b'{"type": "error"}'

    async def test_upstream_headers_relayed_minus_framing(self) -> None:
        upstream = _Upstream(
            headers={
                "content-type": "text/event-stream",
                "request-id": "req_123",
            }
        )
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
            await _drain(response)
        assert response.headers["content-type"] == "text/event-stream"
        assert response.headers["request-id"] == "req_123"

    async def test_sse_body_relayed_byte_identical(self) -> None:
        sse = b'event: message_start\ndata: {"type": "message_start"}\n\n'
        upstream = _Upstream(content=sse, headers={"content-type": "text/event-stream"})
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
            body = await _drain(response)
        assert body == sse

    async def test_sse_usage_reported_after_full_relay(self) -> None:
        sse = (
            b"event: message_start\n"
            b'data: {"message": {"usage": {"input_tokens": 100, '
            b'"cache_read_input_tokens": 20, "cache_creation_input_tokens": 5}}}\n\n'
            b"event: message_delta\n"
            b'data: {"usage": {"output_tokens": 12}}\n\n'
        )
        captured: list[tuple[int, dict[str, int], float]] = []
        upstream = _Upstream(content=sse, headers={"content-type": "text/event-stream"})
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
                on_complete=lambda status, usage, duration: captured.append(
                    (status, usage, duration)
                ),
            )
            body = await _drain(response)
        assert body == sse
        assert len(captured) == 1
        status, usage, duration = captured[0]
        assert status == 200
        assert duration >= 0
        assert usage == {
            "input_tokens": 100,
            "output_tokens": 12,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 5,
        }

    async def test_responses_sse_usage_reported_after_full_relay(self) -> None:
        sse = (
            b"event: response.completed\n"
            b'data: {"type": "response.completed", "response": {"usage": {'
            b'"input_tokens": 100, "output_tokens": 12, '
            b'"input_tokens_details": {"cached_tokens": 20}}}}\n\n'
        )
        captured: list[tuple[int, dict[str, int], float]] = []
        upstream = _Upstream(content=sse, headers={"content-type": "text/event-stream"})
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/responses",
                headers=CLIENT_HEADERS,
                body=b"{}",
                on_complete=lambda status, usage, duration: captured.append(
                    (status, usage, duration)
                ),
            )
            body = await _drain(response)
        assert body == sse
        assert len(captured) == 1
        status, usage, duration = captured[0]
        assert status == 200
        assert duration >= 0
        assert usage == {
            "input_tokens": 80,
            "output_tokens": 12,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 0,
        }

    async def test_json_usage_reported_after_full_relay(self) -> None:
        body = (
            b'{"content": [{"type": "text", "text": "ok"}], '
            b'"usage": {"input_tokens": 10, "output_tokens": 3}}'
        )
        captured: list[tuple[int, dict[str, int], float]] = []
        upstream = _Upstream(content=body, headers={"content-type": "application/json"})
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
                on_complete=lambda status, usage, duration: captured.append(
                    (status, usage, duration)
                ),
            )
            relayed = await _drain(response)
        assert relayed == body
        assert captured[0][0] == 200
        assert captured[0][1]["input_tokens"] == 10
        assert captured[0][1]["output_tokens"] == 3

    async def test_unreachable_upstream_returns_502_error(self) -> None:
        upstream = _Upstream(error=httpx.ConnectError("connection refused"))
        async with _client(upstream) as client:
            response = await pt.forward(
                client=client,
                upstream=UPSTREAM,
                method="POST",
                path="/v1/messages",
                headers=CLIENT_HEADERS,
                body=b"{}",
            )
        assert response.status_code == 502
        assert b'"type": "error"' in response.body
        assert b"connection refused" not in response.body


class _FakeUpstreamResponse:
    """Duck-typed httpx response tracking whether aclose was called."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.closed = False

    async def aiter_raw(self) -> object:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class TestRelayCleanup:
    async def test_upstream_closed_after_full_relay(self) -> None:
        fake = _FakeUpstreamResponse([b"a", b"b"])
        chunks = [chunk async for chunk in pt._relay(fake)]
        assert chunks == [b"a", b"b"]
        assert fake.closed is True

    async def test_upstream_closed_on_client_disconnect(self) -> None:
        fake = _FakeUpstreamResponse([b"a", b"b", b"c"])
        gen = pt._relay(fake)
        assert await anext(gen) == b"a"
        await gen.aclose()
        assert fake.closed is True

    async def test_on_complete_not_called_on_client_disconnect(self) -> None:
        called: list[tuple[int, dict[str, int], float]] = []
        fake = _FakeUpstreamResponse([b"a", b"b", b"c"])
        gen = pt._relay(
            fake,
            on_complete=lambda status, usage, duration: called.append((status, usage, duration)),
        )
        assert await anext(gen) == b"a"
        await gen.aclose()
        assert fake.closed is True
        assert called == []
