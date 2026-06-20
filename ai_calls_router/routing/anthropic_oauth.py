"""Direct routed calls for Anthropic subscription-OAuth tiers.

Claude Code subscription OAuth bills against the subscription's separate
per-model quota, not a pay-per-token API key, so an OAuth cheap tier cannot use
the LiteLLM/API-key engine path (that path forwards a tier key and never sees the
client's OAuth bearer). This module forwards a model-swapped Anthropic Messages
body to api.anthropic.com carrying the client's inbound OAuth headers verbatim,
so a routed Sonnet turn draws the subscription's Sonnet quota instead of Opus.
It buffers the JSON response and returns normalized usage so the routed
accounting and premium-escalation guard run unchanged. Every failure returns
None so serving falls back to premium passthrough.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import httpx

from ai_calls_router._lib import jsonnum
from ai_calls_router.accounting import shrink_stats
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)
from ai_calls_router.routing.engine import prepare_routed_body

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ai_calls_router._lib.types import JsonObject, JsonValue

logger = logging.getLogger("acr.anthropic_oauth")

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_PROVIDER_ERROR_BODY_CHARS = 500
_ANTHROPIC_MODEL_PREFIX = "anthropic/"
# Claude Code's opus[1m] sends anthropic-beta: context-1m-*; the subscription
# bills long-context routed turns as credit requests (HTTP 429), so drop it.
_LONG_CONTEXT_BETA_PREFIX = "context-1m"


def is_anthropic_oauth_tier(tier_cfg: JsonObject) -> bool:
    """Return whether a tier targets Anthropic subscription routing.

    The discriminator is the provider/model, not the auth mode: the caller has
    already resolved ``auth_mode == "oauth"``. A tier is Anthropic-OAuth when it
    is not a Codex tier and either declares ``provider: anthropic`` or carries an
    ``anthropic/`` model prefix.

    Args:
        tier_cfg: Tier entry from config.yaml.

    Returns:
        True when the tier should serve through api.anthropic.com, else False.
    """
    try:
        if is_codex_tier(tier_cfg):
            return False
        parsed = parse_tier_config(tier_cfg)
    except ConfigSchemaError:
        return False
    return parsed.provider == "anthropic" or parsed.model.startswith(_ANTHROPIC_MODEL_PREFIX)


def native_model_id(tier_cfg: JsonObject) -> str:
    """Return the provider-native model id for an Anthropic tier.

    Strips the ``anthropic/`` LiteLLM-style prefix so api.anthropic.com receives
    a bare id (``claude-sonnet-4-6``), leaving an already-bare id unchanged.

    Args:
        tier_cfg: Tier config marked as an Anthropic tier.

    Returns:
        The model id with any ``anthropic/`` prefix removed.
    """
    model = parse_tier_config(tier_cfg).model
    if model.startswith(_ANTHROPIC_MODEL_PREFIX):
        return model[len(_ANTHROPIC_MODEL_PREFIX) :]
    return model


def _response_usage(response_body: JsonObject) -> tuple[int, int, int, int]:
    """Return Anthropic usage as input, output, cache-read, cache-creation tokens."""
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    return (
        jsonnum.int_value(usage.get("input_tokens"), minimum=0),
        jsonnum.int_value(usage.get("output_tokens"), minimum=0),
        jsonnum.int_value(usage.get("cache_read_input_tokens"), minimum=0),
        jsonnum.int_value(usage.get("cache_creation_input_tokens"), minimum=0),
    )


def _payload_effort(payload: JsonObject) -> object:
    """Return the routed reasoning effort in a payload, or None when absent."""
    output_config = payload.get("output_config")
    return output_config.get("effort") if isinstance(output_config, dict) else None


def _without_long_context_beta(value: str) -> str:
    """Drop context-1m tokens from a comma-joined anthropic-beta header value."""
    kept = [
        token.strip()
        for token in value.split(",")
        if token.strip() and not token.strip().startswith(_LONG_CONTEXT_BETA_PREFIX)
    ]
    return ",".join(kept)


# Claude Code's opus[1m] also names the 1M variant in the body model itself
# (claude-opus-4-8[1m]); stripping only the header leaves the suffix to re-trigger
# the long-context 429, so the passthrough sanitizer drops both.
_LONG_CONTEXT_MODEL_SUFFIX = "[1m]"


def _strip_long_context_model(body_bytes: bytes) -> tuple[bytes, bool]:
    """Drop a trailing ``[1m]`` long-context suffix from the body model.

    Args:
        body_bytes: Raw Anthropic Messages request body.

    Returns:
        The (possibly re-serialized) body and whether the suffix was removed. The
        body is returned unchanged when it is not a JSON object, lacks a string
        model, or the model does not carry the suffix.
    """
    try:
        body = json.loads(body_bytes)
    except (ValueError, TypeError):
        return body_bytes, False
    if not isinstance(body, dict):
        return body_bytes, False
    model = body.get("model")
    if not isinstance(model, str) or not model.endswith(_LONG_CONTEXT_MODEL_SUFFIX):
        return body_bytes, False
    body["model"] = model[: -len(_LONG_CONTEXT_MODEL_SUFFIX)]
    return json.dumps(body, ensure_ascii=False).encode("utf-8"), True


def _drop_long_context_beta(header_items: list[tuple[str, str]]) -> dict[str, str]:
    """Return headers with context-1m stripped from any anthropic-beta value.

    The anthropic-beta header is dropped entirely when context-1m was its only
    token; every other header is preserved verbatim.
    """
    out: dict[str, str] = {}
    for key, value in header_items:
        if key.lower() != "anthropic-beta":
            out[key] = value
            continue
        filtered = _without_long_context_beta(value)
        if filtered:
            out[key] = filtered
    return out


def strip_long_context_passthrough(
    body_bytes: bytes, headers: Mapping[str, str]
) -> tuple[bytes, dict[str, str]] | None:
    """Remove the 1M long-context opt-in from a premium-passthrough Anthropic turn.

    Claude Code's ``opus[1m]`` turns carry both an ``anthropic-beta: context-1m``
    token and a ``[1m]`` model suffix. An OAuth subscription without long-context
    credits rejects them with HTTP 429 (``Usage credits are required for long
    context requests``). The routed path strips the beta already; passthrough
    forwards the request verbatim, so large premium turns 429. This drops both
    markers so the subscription serves the turn within its standard window. A turn
    that genuinely exceeds the standard window then fails on length instead, which
    the subscription cannot serve either way.

    Args:
        body_bytes: Raw Anthropic ``/v1/messages`` request body to relay.
        headers: Client request headers to forward upstream.

    Returns:
        The sanitized ``(body, headers)`` pair when either long-context marker was
        present, or ``None`` so the caller relays the request byte-identical.
    """
    header_items = list(headers.items())
    beta = next((value for key, value in header_items if key.lower() == "anthropic-beta"), None)
    has_long_context_beta = beta is not None and _without_long_context_beta(beta) != beta
    new_body, model_stripped = _strip_long_context_model(body_bytes)
    if not has_long_context_beta and not model_stripped:
        return None
    return new_body, _drop_long_context_beta(header_items)


# Only these client headers are forwarded to api.anthropic.com. An explicit
# allowlist keeps unrelated client headers (cookies, x-forwarded-*, any other
# auth) from leaking upstream; the routed OAuth turn needs only the subscription
# bearer, the API version, and the anthropic-beta flags. Content-Type is set
# from the re-serialized body, so it is never taken from the client.
_FORWARDABLE_OAUTH_HEADERS = frozenset({"authorization", "anthropic-version", "anthropic-beta"})


def _forward_headers(oauth_headers: Mapping[str, str]) -> dict[str, str]:
    """Return the allowlisted upstream headers for a routed OAuth turn.

    Only the OAuth bearer, anthropic-version, and anthropic-beta tokens are
    forwarded; every other client header is dropped so nothing unrelated reaches
    api.anthropic.com. The 1M-context beta is stripped from anthropic-beta: a
    routed cheap turn must not opt into long context, which the subscription
    rejects for the routed model as a credit request (HTTP 429) instead of
    covering it. The JSON content type is forced for the re-serialized body.
    """
    headers: dict[str, str] = {}
    for key, value in oauth_headers.items():
        lowered = key.lower()
        if lowered not in _FORWARDABLE_OAUTH_HEADERS:
            continue
        if lowered == "anthropic-beta":
            filtered = _without_long_context_beta(value)
            if filtered:
                headers[key] = filtered
            continue
        headers[key] = value
    headers["Content-Type"] = "application/json"
    return headers


async def messages_call(
    *,
    body: JsonObject,
    tier_cfg: JsonObject,
    oauth_headers: Mapping[str, str],
    client: httpx.AsyncClient | None = None,
) -> tuple[JsonObject, tuple[int, int, int, int], shrink_stats.ShrinkStats] | None:
    """POST a model-swapped Anthropic Messages body to the subscription endpoint.

    The body is rewritten for the tier (model swapped to the native id, stream
    dropped, max_tokens clamped, thinking blocks stripped) and sent to
    api.anthropic.com with the client's inbound OAuth headers, so the turn bills
    the subscription's quota for the tier model. No compression runs on this
    path. Any non-200 status or transport error returns None so the caller falls
    back to premium passthrough.

    Args:
        body: Anthropic-format request body from the client.
        tier_cfg: Tier config marked as an Anthropic-OAuth tier.
        oauth_headers: Hop-by-hop-filtered client headers carrying the OAuth
            bearer to forward upstream.
        client: Optional shared HTTP client for tests or server reuse.

    Returns:
        Parsed Anthropic JSON body, normalized token usage, and a no-op shrink,
        or None when the caller must passthrough.
    """
    if not is_anthropic_oauth_tier(tier_cfg):
        return None
    shrink = shrink_stats.compute_shrink(path="none", before=body, after=body)
    payload = cast(
        "JsonObject",
        {**prepare_routed_body(body, tier_cfg), "model": native_model_id(tier_cfg)},
    )
    headers = _forward_headers(oauth_headers)
    logger.info(
        "acr: anthropic-oauth routed model=%s effort=%s",
        payload.get("model"),
        _payload_effort(payload),
    )
    timeout_value = tier_cfg.get("timeout", DEFAULT_TIMEOUT_SECONDS)
    timeout = (
        timeout_value
        if isinstance(timeout_value, int | float)
        and not isinstance(timeout_value, bool)
        and timeout_value > 0
        else DEFAULT_TIMEOUT_SECONDS
    )
    own_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.post(
            ANTHROPIC_MESSAGES_URL, json=payload, headers=headers, timeout=timeout
        )
    except httpx.HTTPError as exc:
        logger.warning("anthropic oauth call failed (%s); passing through", exc, exc_info=True)
        return None
    finally:
        if own_client:
            await http.aclose()
    if response.status_code != 200:
        logger.warning(
            "anthropic oauth call returned HTTP %d body=%s; passing through",
            response.status_code,
            response.text[:MAX_PROVIDER_ERROR_BODY_CHARS],
        )
        return None
    try:
        parsed = cast("JsonValue", response.json())
    except ValueError as exc:
        logger.warning("anthropic oauth call returned non-JSON body (%s); passing through", exc)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed, _response_usage(parsed), shrink
