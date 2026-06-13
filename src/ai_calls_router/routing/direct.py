"""Direct Anthropic-format calls to providers with native Anthropic endpoints.

When a tier model belongs to a provider that accepts Anthropic-format requests
natively, bypassing LiteLLM conversion and our built-in compression preserves
byte-identical prefixes for the provider's prefix cache (DeepSeek bills cache
hits ~50x cheaper than misses, beating our token compression on the repetitive
tool-result turns this proxy routes). This module owns the detection rule, the
endpoint table, and the raw httpx POST the routed_call engine invokes instead
of the LiteLLM path. Every failure returns None so routing falls back to
passthrough and never breaks a turn.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("acr.anthropic_direct")

DIRECT_ANTHROPIC_ENDPOINTS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/anthropic",
}
"""Provider prefix -> Anthropic-compatible base URL.

Keyed by the first '/'-separated segment of a LiteLLM model id (e.g.
``deepseek/deepseek-v4-flash`` -> ``deepseek``). Membership here is the entire
detection rule: a tier whose model prefix is a key takes the direct path. The
base URL is joined with ``/v1/messages`` for the POST.
"""

ANTHROPIC_VERSION = "2023-06-01"
DIRECT_CALL_TIMEOUT_SECONDS = 120.0


def provider_prefix(model: Any) -> str | None:
    """Return the provider segment of a ``provider/model`` id, else None.

    Args:
        model: A tier model id, expected to be a string.

    Returns:
        The text before the first '/', or None when ``model`` is not a string
        or carries no '/' (a bare id is never treated as a direct provider).
    """
    if not isinstance(model, str) or "/" not in model:
        return None
    return model.split("/", 1)[0]


def direct_endpoint(model: Any) -> str | None:
    """Return the direct Anthropic base URL for a tier model, if one applies.

    This is the detection gate routed_call branches on: a non-None result means
    the tier bypasses LiteLLM and compression.

    Args:
        model: A tier model id (e.g. ``deepseek/deepseek-v4-pro``).

    Returns:
        The provider's Anthropic base URL when its prefix is wired for direct
        routing, otherwise None (the tier uses the LiteLLM path).
    """
    prefix = provider_prefix(model)
    if prefix is None:
        return None
    return DIRECT_ANTHROPIC_ENDPOINTS.get(prefix)


def native_model_id(model: str) -> str:
    """Strip the LiteLLM provider prefix for the provider's native API.

    DeepSeek's ``/anthropic`` endpoint expects a bare id (``deepseek-v4-pro``),
    not the LiteLLM ``deepseek/deepseek-v4-pro`` form.

    Args:
        model: A tier model id.

    Returns:
        The id after the first '/', or the input unchanged when it has no
        provider prefix.
    """
    return model.split("/", 1)[1] if "/" in model else model


async def direct_call(
    *,
    body: dict[str, Any],
    tier_cfg: dict[str, Any],
    api_key: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = DIRECT_CALL_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """POST a prepared Anthropic-format body straight to the tier provider.

    The body is sent verbatim except for two required edits: the model id is
    reduced to its native form and extended thinking is disabled (DeepSeek
    rejects replayed Claude-signed thinking, and the cheap tiers do not think).
    The request carries only the tier key; the client's Anthropic credential
    never reaches here. Any non-200 status or transport error returns None so
    the caller falls back to passthrough.

    Args:
        body: Prepared Anthropic-format request body (model already swapped to
            the tier id, thinking blocks stripped, max_tokens clamped). Never
            mutated.
        tier_cfg: Tier configuration carrying the LiteLLM ``model`` key.
        api_key: Tier provider API key.
        client: Optional shared httpx client; a transient one is created and
            closed when omitted.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed Anthropic response body, or None on any failure.
    """
    model = tier_cfg.get("model")
    base_url = direct_endpoint(model)
    if base_url is None:
        logger.warning("no direct Anthropic endpoint mapped for model %r", model)
        return None

    url = base_url.rstrip("/") + "/v1/messages"
    payload: dict[str, Any] = {
        **body,
        "model": native_model_id(model),
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
    }

    own_client = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.warning(
                "direct call to %s returned HTTP %d; passing through",
                url,
                response.status_code,
            )
            return None
        return response.json()
    except Exception as exc:
        logger.warning("direct call to %s failed (%s); passing through", url, exc)
        return None
    finally:
        if own_client:
            await http.aclose()
