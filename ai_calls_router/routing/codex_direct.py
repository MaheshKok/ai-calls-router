"""Direct routed calls for Codex/OpenAI Responses tiers.

Codex subscription OAuth does not fit LiteLLM's API-key-only completion path,
so Codex tiers use the Responses API wire directly. The helper keeps that path
small, strips routed-only reasoning payloads, and returns None on provider
errors so serving can fall back to passthrough.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import TYPE_CHECKING, Literal, cast

import httpx

from ai_calls_router._lib import jsonnum
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue

logger = logging.getLogger("acr.codex_direct")

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
CHATGPT_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_PROVIDER_ERROR_BODY_CHARS = 2048
_SENSITIVE_FIELD_PATTERN = re.compile(
    r'("(?:access_token|refresh_token|id_token|api_key|authorization|session_key)"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)


def native_model_id(tier_cfg: JsonObject) -> str:
    """Return the provider-native model id for a Codex tier."""
    model = parse_tier_config(tier_cfg).model
    for prefix in ("codex/", "openai-codex/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _strip_reasoning_items(input_value: JsonValue) -> JsonValue:
    """Return Responses input without encrypted reasoning items."""
    if not isinstance(input_value, list):
        return copy.deepcopy(input_value)
    cleaned: JsonArray = []
    for item in input_value:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            continue
        cleaned.append(copy.deepcopy(item))
    return cleaned


def prepare_responses_body(body: JsonObject, tier_cfg: JsonObject) -> JsonObject:
    """Build a deterministic Responses request body for the routed Codex tier."""
    parsed = parse_tier_config(tier_cfg)
    routed = {
        key: copy.deepcopy(value)
        for key, value in body.items()
        if key not in {"stream", "previous_response_id"}
    }
    routed["model"] = native_model_id(tier_cfg)
    if "input" in body:
        routed["input"] = _strip_reasoning_items(body["input"])
    requested = body.get("max_output_tokens")
    if parsed.max_tokens is not None:
        over_limit = (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested > parsed.max_tokens
        )
        if over_limit:
            routed["max_output_tokens"] = parsed.max_tokens
    return routed


def _api_key_headers(api_key: str) -> dict[str, str]:
    """Return OpenAI API-key headers without client header leakage."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _chatgpt_oauth_headers(
    chatgpt_headers: list[tuple[str, str]] | None,
) -> dict[str, str] | None:
    """Return ChatGPT OAuth headers from already-resolved OAuth mode."""
    if chatgpt_headers is None:
        return None
    headers = dict(chatgpt_headers)
    headers["Content-Type"] = "application/json"
    return headers


def _request_target(
    *,
    credential: str,
    auth_mode: Literal["api_key", "oauth"],
    chatgpt_headers: list[tuple[str, str]] | None,
) -> tuple[str, dict[str, str]] | None:
    """Return URL and headers for an already-resolved auth mode."""
    if auth_mode == "oauth":
        headers = _chatgpt_oauth_headers(chatgpt_headers)
        return (CHATGPT_CODEX_RESPONSES_URL, headers) if headers is not None else None
    if auth_mode == "api_key":
        return OPENAI_RESPONSES_URL, _api_key_headers(credential)
    return None


def _response_usage(response_body: JsonObject) -> tuple[int, int, int, int]:
    """Return Responses usage as input, output, cache-read, cache-creation tokens."""
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0
    cache_read = 0
    details = usage.get("input_tokens_details")
    if isinstance(details, dict):
        cache_read = jsonnum.int_value(details.get("cached_tokens"), minimum=0)
    total_input = jsonnum.int_value(usage.get("input_tokens"), minimum=0)
    return (
        max(total_input - cache_read, 0),
        jsonnum.int_value(usage.get("output_tokens"), minimum=0),
        cache_read,
        0,
    )


def _provider_error_excerpt(response: httpx.Response) -> str:
    """Return a redacted, bounded provider error body excerpt."""
    text = response.text.strip()
    if not text:
        return "<empty>"
    redacted = _SENSITIVE_FIELD_PATTERN.sub(r'\1"[redacted]"', text)
    redacted = _BEARER_PATTERN.sub("Bearer [redacted]", redacted)
    if len(redacted) <= MAX_PROVIDER_ERROR_BODY_CHARS:
        return redacted
    return f"{redacted[:MAX_PROVIDER_ERROR_BODY_CHARS]}...<truncated>"


def _data_lines(block: str) -> str:
    """Return joined SSE data lines from one event block."""
    lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("data:"):
            lines.append(line.removeprefix("data:").strip())
    return "\n".join(lines).strip()


def _response_from_sse(text: str) -> JsonObject | None:
    """Return the final response object from a Responses SSE body."""
    for block in text.split("\n\n"):
        data = _data_lines(block)
        if not data or data == "[DONE]":
            continue
        parsed = cast("JsonValue", json.loads(data))
        if not isinstance(parsed, dict):
            continue
        response = parsed.get("response")
        if parsed.get("type") == "response.completed" and isinstance(response, dict):
            return response
        if parsed.get("object") == "response" and isinstance(parsed.get("output"), list):
            return parsed
    return None


async def responses_call(
    *,
    body: JsonObject,
    tier_cfg: JsonObject,
    credential: str,
    auth_mode: Literal["api_key", "oauth"] = "api_key",
    chatgpt_headers: list[tuple[str, str]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[JsonObject, tuple[int, int, int, int]] | None:
    """POST a sanitized Responses request to a Codex tier.

    Args:
        body: Original OpenAI Responses request body from the client.
        tier_cfg: Tier config marked as a Codex tier.
        credential: Resolved credential value.
        auth_mode: Resolved credential mode.
        chatgpt_headers: Forwardable ChatGPT OAuth headers for OAuth mode.
        client: Optional shared HTTP client for tests or server reuse.

    Returns:
        Parsed Responses JSON body and normalized token usage, or None when
        the caller must passthrough.
    """
    try:
        target = (
            _request_target(
                credential=credential,
                auth_mode=auth_mode,
                chatgpt_headers=chatgpt_headers,
            )
            if is_codex_tier(tier_cfg)
            else None
        )
        if target is None:
            return None
        url, headers = target
        payload = prepare_responses_body(body, tier_cfg)
        if auth_mode == "oauth":
            payload["stream"] = True
            payload.pop("max_output_tokens", None)
        timeout_value = tier_cfg.get("timeout", DEFAULT_TIMEOUT_SECONDS)
        timeout = (
            timeout_value if isinstance(timeout_value, int | float) else DEFAULT_TIMEOUT_SECONDS
        )
        own_client = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout)
        try:
            response = await http.post(url, json=payload, headers=headers, timeout=timeout)
        finally:
            if own_client:
                await http.aclose()
        if response.status_code != 200:
            logger.warning(
                "codex direct call to %s returned HTTP %d body=%s; passing through",
                url,
                response.status_code,
                _provider_error_excerpt(response),
            )
            return None
        if auth_mode == "oauth":
            parsed_sse = _response_from_sse(response.text)
            if parsed_sse is None:
                logger.warning(
                    "codex direct call to %s returned unparseable SSE body=%s; passing through",
                    url,
                    _provider_error_excerpt(response),
                )
            return (parsed_sse, _response_usage(parsed_sse)) if parsed_sse is not None else None
        parsed_response = cast("JsonValue", response.json())
        if not isinstance(parsed_response, dict):
            return None
        return parsed_response, _response_usage(parsed_response)
    except (ConfigSchemaError, httpx.HTTPError, ValueError) as exc:
        logger.warning("codex direct call failed (%s); passing through", exc, exc_info=True)
        return None
