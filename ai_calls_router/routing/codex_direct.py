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
from typing import TYPE_CHECKING, cast

import httpx

from ai_calls_router.routing.config_schema import (
    CODEX_OAUTH_SENTINEL,
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def response_escalates(response_body: JsonObject, premium_tools: Sequence[str]) -> list[str]:
    """Return premium tool names requested by a Codex Responses output."""
    premium = set(premium_tools)
    if not premium:
        return []
    output = response_body.get("output")
    if not isinstance(output, list):
        return []
    names: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        name = item.get("name")
        if isinstance(name, str) and name in premium and name not in names:
            names.append(name)
    return names


def _api_key_headers(api_key: str) -> dict[str, str]:
    """Return OpenAI API-key headers without client header leakage."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _oauth_headers(chatgpt_headers: Sequence[tuple[str, str]] | None) -> dict[str, str] | None:
    """Return ChatGPT OAuth headers when the client supplied Codex auth."""
    if chatgpt_headers is None:
        return None
    headers = dict(chatgpt_headers)
    headers["Content-Type"] = "application/json"
    return headers


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
    chatgpt_headers: Sequence[tuple[str, str]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> JsonObject | None:
    """POST a sanitized Responses request to a Codex tier.

    Args:
        body: Original OpenAI Responses request body from the client.
        tier_cfg: Tier config marked as a Codex tier.
        credential: OpenAI API key or ``"oauth"`` sentinel.
        chatgpt_headers: Forwardable ChatGPT OAuth headers for the sentinel path.
        client: Optional shared HTTP client for tests or server reuse.

    Returns:
        Parsed Responses JSON body, or None when the caller must passthrough.
    """
    try:
        if not is_codex_tier(tier_cfg):
            return None
        payload = prepare_responses_body(body, tier_cfg)
        if credential == CODEX_OAUTH_SENTINEL:
            payload["stream"] = True
            payload.pop("max_output_tokens", None)
            headers = _oauth_headers(chatgpt_headers)
            url = CHATGPT_CODEX_RESPONSES_URL
        else:
            headers = _api_key_headers(credential)
            url = OPENAI_RESPONSES_URL
        if headers is None:
            return None
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
        if credential == CODEX_OAUTH_SENTINEL:
            parsed_sse = _response_from_sse(response.text)
            if parsed_sse is None:
                logger.warning(
                    "codex direct call to %s returned unparseable SSE body=%s; passing through",
                    url,
                    _provider_error_excerpt(response),
                )
            return parsed_sse
        parsed_response = cast("JsonValue", response.json())
        return parsed_response if isinstance(parsed_response, dict) else None
    except (ConfigSchemaError, httpx.HTTPError, ValueError) as exc:
        logger.warning("codex direct call failed (%s); passing through", exc, exc_info=True)
        return None
