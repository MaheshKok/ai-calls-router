"""Starlette application wiring the proxy transport to the orchestrator.

This module owns transport concerns only: the route table, the shared upstream
HTTP client, the assembled-routes cache, the loopback-bind warning, and parsing
each incoming request into a transport-agnostic context. The serving decision --
adapter selection, group resolution, routed tier attempt, compression, recording,
and passthrough -- lives in :mod:`ai_calls_router.proxy.orchestrator`. POST
/v1/messages, /v1/chat/completions, and /v1/responses are decided by the
orchestrator; GET /health answers locally, GET /metrics and /dashboard expose
live telemetry, and every other path proxies unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import logging
import os
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ai_calls_router._lib import config, logging_setup
from ai_calls_router.accounting import metrics
from ai_calls_router.ops import bootstrap
from ai_calls_router.proxy import observability, orchestrator, passthrough, route_dispatch
from ai_calls_router.routing import decide as routing
from ai_calls_router.routing import provider_config
from ai_calls_router.routing.adapters.base import KNOWN_GROUPS

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject

logger = logging.getLogger("acr.server")

LOG_REVISION = "2026-06-15-premium-guard-v2"

codex_direct = route_dispatch.codex_direct
_try_codex_direct_route = route_dispatch.try_codex_direct_route

PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


@dataclass
class _RoutesCache:
    signature: tuple[tuple[str, int, int, int], ...] | None = None
    routes: JsonObject | None = None


_ROUTES_CACHE_LOCK = threading.Lock()
_ROUTES_CACHE = _RoutesCache()


def _is_loopback_host(host: str) -> bool:
    """Return whether a configured bind host is loopback-only."""
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _warn_if_public_bind(settings: config.ServerSettings) -> None:
    """Warn when unauthenticated local telemetry is bound beyond loopback."""
    if _is_loopback_host(settings.host):
        return
    logger.warning(
        "acr: server.host=%s is not loopback; /metrics and /dashboard are unauthenticated",
        settings.host,
    )


def _assembled_routes_signature() -> tuple[tuple[str, int, int, int], ...]:
    """Build the mtime signature for the global and provider config files."""
    paths = [config.config_path()]
    paths.extend(config.provider_config_path(group) for group in sorted(KNOWN_GROUPS))
    signature: list[tuple[str, int, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((str(path), 0, 0, 0))
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino))
    return tuple(signature)


def _assemble_routes_fail_open(
    base: JsonObject, provider_files: dict[str, JsonObject]
) -> JsonObject:
    """Assemble routes, dropping invalid provider payloads one at a time."""
    remaining = dict(provider_files)
    while remaining:
        try:
            return provider_config.assemble_routes(base, provider_files=remaining)
        except provider_config.ProviderConfigError as exc:
            logger.warning(
                "acr: provider config assembly failed (%s); skipping provider file",
                exc,
                exc_info=True,
            )
            if exc.group is not None and exc.group in remaining:
                remaining.pop(exc.group)
                continue
            remaining: dict[str, JsonObject] = {}
    return provider_config.assemble_routes(base, provider_files={})


def _load_assembled_routes() -> JsonObject:
    """Load the canonical routes dict assembled from global and provider YAML."""
    signature = _assembled_routes_signature()
    with _ROUTES_CACHE_LOCK:
        if _ROUTES_CACHE.signature == signature and _ROUTES_CACHE.routes is not None:
            return _ROUTES_CACHE.routes

    assembled = _assemble_routes_fail_open(
        routing.load_routes(),
        provider_config.load_provider_files(),
    )
    with _ROUTES_CACHE_LOCK:
        _ROUTES_CACHE.signature = signature
        _ROUTES_CACHE.routes = assembled
    return assembled


def _client_ip(request: Request) -> str:
    """Extract the client IP from the request."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _user_agent(request: Request) -> str:
    """Extract the User-Agent header."""
    return request.headers.get("user-agent", "")


def _build_context(request: Request, body_bytes: bytes) -> orchestrator.RequestContext:
    """Snapshot a Starlette request into a transport-agnostic context."""
    return orchestrator.RequestContext(
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        headers=request.headers,
        body_bytes=body_bytes,
        client=request.app.state.client,
        user_agent=_user_agent(request),
        client_ip=_client_ip(request),
    )


async def health(request: Request) -> JSONResponse:
    """Answer the daemon health probe without touching the upstream.

    Args:
        request: Incoming health-check request.

    Returns:
        A local 200 status response.
    """
    return JSONResponse({"status": "ok"})


async def _serve_routed(request: Request) -> Response:
    """Snapshot the request and serve it through the orchestrator.

    Args:
        request: Incoming adapter-backed request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    with logging_setup.request_context():
        ctx = _build_context(request, await request.body())
        return await orchestrator.handle(ctx, routes_loader=_load_assembled_routes)


async def messages(request: Request) -> Response:
    """Decide and serve one /v1/messages request.

    Args:
        request: Incoming Anthropic Messages API request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    return await _serve_routed(request)


async def chat_completions(request: Request) -> Response:
    """Decide and serve one /v1/chat/completions request.

    Args:
        request: Incoming OpenAI Chat Completions request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    return await _serve_routed(request)


async def responses(request: Request) -> Response:
    """Decide and serve one /v1/responses request.

    Args:
        request: Incoming OpenAI Responses request.

    Returns:
        The routed response or the streamed premium passthrough.
    """
    return await _serve_routed(request)


async def proxy(request: Request) -> Response:
    """Proxy any non-adapter endpoint to the resolved upstream.

    Args:
        request: Incoming client request for any other path.

    Returns:
        The streamed upstream response.
    """
    ctx = _build_context(request, await request.body())
    # Only /v1/models resolves identity here; other catch-all paths keep the premium default.
    group = orchestrator.models_passthrough_group(ctx, routes_loader=_load_assembled_routes)
    return await orchestrator.serve_passthrough(
        ctx, ctx.body_bytes, routes_loader=_load_assembled_routes, group=group
    )


@contextlib.asynccontextmanager
async def _lifespan(
    app: Starlette, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncGenerator[None, None]:
    """Own the shared upstream HTTP client for the app's lifetime.

    Args:
        app: The Starlette application being started.
        transport: Optional httpx transport override (tests inject mocks).

    Yields:
        None while the application serves requests.
    """
    try:
        bootstrap.ensure_provider_configs()
    except Exception as exc:
        logger.warning("acr: provider config bootstrap failed (%s); continuing", exc, exc_info=True)
    try:
        _warn_if_public_bind(config.server_settings(routing.load_routes()))
    except Exception as exc:
        logger.warning("acr: public bind warning check failed (%s); continuing", exc, exc_info=True)
    mtr = metrics.get_metrics()
    mtr.bootstrap(ledger_path=config.ledger_path(), max_recent=100)
    app.state.client = httpx.AsyncClient(transport=transport, timeout=passthrough.UPSTREAM_TIMEOUT)
    try:
        yield
    finally:
        await app.state.client.aclose()


def create_app(transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """Build the proxy application.

    Args:
        transport: Optional httpx transport override (tests inject mocks).

    Returns:
        A Starlette app serving /health, /metrics, /v1/messages, and a catch-all proxy.
    """
    logging_setup.setup_logging()
    logger.info(
        "startup revision=%s pid=%s cwd=%s log=%s",
        LOG_REVISION,
        os.getpid(),
        Path.cwd(),
        config.log_path(),
    )
    lifespan = functools.partial(_lifespan, transport=transport)
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/metrics", observability.metrics_endpoint, methods=["GET"]),
            Route("/dashboard", observability.dashboard, methods=["GET"]),
            Route("/v1/messages", messages, methods=["POST"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/responses", responses, methods=["POST"]),
            Route("/{path:path}", proxy, methods=PROXY_METHODS),
        ],
        lifespan=lifespan,
    )
