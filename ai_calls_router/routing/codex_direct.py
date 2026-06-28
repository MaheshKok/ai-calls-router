"""Direct routed calls for Codex/OpenAI Responses tiers.

Codex subscription OAuth does not fit LiteLLM's API-key-only completion path,
so Codex tiers use the Responses API wire directly. The ChatGPT Codex backend
is stateless -- it rejects ``previous_response_id`` ("Unsupported parameter")
-- so the path drops that field and replays full input, while preserving the
encrypted ``reasoning`` items the backend needs to pair a ``function_call`` with
its ``function_call_output``. It returns None on provider errors so serving can
fall back to passthrough.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import TYPE_CHECKING, Literal, cast

import httpx

from ai_calls_router._lib import jsonnum
from ai_calls_router.routing import forward_compression
from ai_calls_router.routing.config_schema import (
    ConfigSchemaError,
    is_codex_tier,
    parse_tier_config,
)

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue
    from ai_calls_router.accounting import shrink_stats

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
# Hosted tools the routed Codex models reject. gpt-5.3-codex-spark returns 400
# for image_generation, so routed tiers drop it; premium passthrough keeps it.
_UNSUPPORTED_ROUTED_TOOLS = frozenset({"image_generation"})


def native_model_id(tier_cfg: JsonObject) -> str:
    """Return the provider-native model id for a Codex tier."""
    model = parse_tier_config(tier_cfg).model
    for prefix in ("codex/", "openai-codex/", "openai/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _is_unsupported_routed_tool(tool: JsonValue) -> bool:
    """Return whether a tool entry is a hosted tool routed Codex tiers must drop."""
    return isinstance(tool, dict) and tool.get("type") in _UNSUPPORTED_ROUTED_TOOLS


def prepare_responses_body(body: JsonObject, tier_cfg: JsonObject) -> JsonObject:
    """Build a deterministic Responses request body for the routed Codex tier.

    ``stream`` and ``previous_response_id`` are dropped: the path sets ``stream``
    per auth mode, and the ChatGPT Codex backend rejects ``previous_response_id``
    ("Unsupported parameter") because it is stateless -- Codex replays full input
    history instead. Encrypted ``reasoning`` items are preserved: they precede
    their ``function_call`` and the backend needs them to pair the call with its
    ``function_call_output`` (dropping them triggers a 400 "No tool output found
    for function call"). The model id and ``max_output_tokens`` are normalized to
    the tier, and hosted tools the routed models reject (``image_generation``)
    are stripped so they do not 400 the request.

    Args:
        body: Original OpenAI Responses request body from the client.
        tier_cfg: Tier config marked as a Codex tier.

    Returns:
        A new Responses body with the routed model and clamped token ceiling.
    """
    parsed = parse_tier_config(tier_cfg)
    routed = {
        key: copy.deepcopy(value)
        for key, value in body.items()
        if key not in {"stream", "previous_response_id"}
    }
    routed["model"] = native_model_id(tier_cfg)
    requested = body.get("max_output_tokens")
    if parsed.max_tokens is not None:
        over_limit = (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested > parsed.max_tokens
        )
        if over_limit:
            routed["max_output_tokens"] = parsed.max_tokens
    tools = routed.get("tools")
    if isinstance(tools, list):
        routed["tools"] = [tool for tool in tools if not _is_unsupported_routed_tool(tool)]
    return routed


def unpaired_function_call_ids(body: JsonObject) -> list[str]:
    """Return ids of ``function_call`` items with no matching output in ``body``.

    The ChatGPT Codex backend is stateless: with ``previous_response_id`` dropped
    it validates the full input array, so every ``function_call`` /
    ``custom_tool_call`` must be answered by a ``function_call_output`` /
    ``custom_tool_call_output`` present in the same array. Codex sends delta input
    that relies on server-side state, so a routed continuation turn can carry a
    call whose output lives server-side -- that orphan triggers a 400 "No tool
    output found for function call". Callers decline routing (fall back to
    passthrough) when this is non-empty.

    Args:
        body: A prepared Responses request body.

    Returns:
        Sorted call ids of unanswered function calls; empty when input is safe.
    """
    items = body.get("input")
    if not isinstance(items, list):
        return []
    calls: set[str] = set()
    answered: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call"}:
            calls.add(call_id)
        elif item_type in {"function_call_output", "custom_tool_call_output"}:
            answered.add(call_id)
    return sorted(calls - answered)


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
    headers = {key: value for key, value in chatgpt_headers if key.lower() != "content-type"}
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


def _event_index(event: JsonObject, key: str) -> int | None:
    value = event.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _empty_text_part() -> JsonObject:
    annotations: list[JsonValue] = []
    return {"type": "output_text", "text": "", "annotations": annotations}


def _content_list(item: JsonObject) -> list[JsonValue]:
    content = item.get("content")
    if isinstance(content, list):
        return cast("list[JsonValue]", content)

    new_content: list[JsonValue] = []
    item["content"] = new_content
    return new_content


def _ensure_output_message(
    output: dict[int, JsonObject],
    output_index: int,
) -> JsonObject:
    item = output.get(output_index)
    if isinstance(item, dict) and item.get("type") == "message":
        return item
    content: list[JsonValue] = []
    new_item: JsonObject = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": content,
    }
    output[output_index] = new_item
    return new_item


def _ensure_text_part(
    output: dict[int, JsonObject],
    output_index: int,
    content_index: int,
) -> JsonObject:
    item = _ensure_output_message(output, output_index)
    content = _content_list(item)

    while len(content) <= content_index:
        content.append(_empty_text_part())

    part = content[content_index]
    if not isinstance(part, dict):
        part_obj = _empty_text_part()
        content[content_index] = part_obj
    else:
        part_obj = cast("JsonObject", part)

    part_obj.setdefault("type", "output_text")
    part_obj.setdefault("text", "")
    part_obj.setdefault("annotations", [])
    return part_obj


def _put_content_part(output: dict[int, JsonObject], event: JsonObject) -> None:
    output_index = _event_index(event, "output_index")
    content_index = _event_index(event, "content_index")
    part = event.get("part")
    if output_index is None or content_index is None or not isinstance(part, dict):
        return

    item = _ensure_output_message(output, output_index)
    content = _content_list(item)

    while len(content) <= content_index:
        content.append(_empty_text_part())
    content[content_index] = copy.deepcopy(cast("JsonObject", part))


def _put_output_item(output: dict[int, JsonObject], event: JsonObject) -> None:
    output_index = _event_index(event, "output_index")
    item = event.get("item")
    if output_index is not None and isinstance(item, dict):
        output[output_index] = copy.deepcopy(cast("JsonObject", item))


def _append_text_delta(output: dict[int, JsonObject], event: JsonObject) -> None:
    output_index = _event_index(event, "output_index")
    content_index = _event_index(event, "content_index")
    delta = event.get("delta")
    if output_index is None or content_index is None or not isinstance(delta, str):
        return

    part = _ensure_text_part(output, output_index, content_index)
    existing = part.get("text")
    part["text"] = f"{existing if isinstance(existing, str) else ''}{delta}"


def _put_text_done(output: dict[int, JsonObject], event: JsonObject) -> None:
    output_index = _event_index(event, "output_index")
    content_index = _event_index(event, "content_index")
    text_value = event.get("text")
    if output_index is not None and content_index is not None and isinstance(text_value, str):
        _ensure_text_part(output, output_index, content_index)["text"] = text_value


def _apply_sse_output_event(output: dict[int, JsonObject], event: JsonObject) -> None:
    event_type = event.get("type")
    if event_type == "response.output_item.added":
        _put_output_item(output, event)
    elif event_type in {"response.content_part.added", "response.content_part.done"}:
        _put_content_part(output, event)
    elif event_type == "response.output_text.delta":
        _append_text_delta(output, event)
    elif event_type == "response.output_text.done":
        _put_text_done(output, event)


def _sse_events(text: str) -> list[JsonObject]:
    events: list[JsonObject] = []
    for block in text.split("\n\n"):
        data = _data_lines(block)
        if not data or data == "[DONE]":
            continue
        parsed = cast("JsonValue", json.loads(data))
        if isinstance(parsed, dict):
            events.append(cast("JsonObject", parsed))
    return events


def _completed_response(event: JsonObject) -> JsonObject | None:
    if event.get("object") == "response" and isinstance(event.get("output"), list):
        return event
    response = event.get("response")
    if event.get("type") == "response.completed" and isinstance(response, dict):
        return cast("JsonObject", response)
    return None


def _response_with_rebuilt_output(
    final_response: JsonObject | None,
    output: dict[int, JsonObject],
) -> JsonObject | None:
    if final_response is None:
        return None

    final_output = final_response.get("output")
    if isinstance(final_output, list) and final_output:
        return final_response

    rebuilt: list[JsonValue] = []
    for output_index in sorted(output):
        item = output[output_index]
        if item.get("type") == "message":
            item["status"] = "completed"
        rebuilt.append(item)

    if not rebuilt:
        return None

    return {**final_response, "output": rebuilt}


def _response_from_sse(text: str) -> JsonObject | None:
    """Return the final response object from a Responses SSE body."""
    output: dict[int, JsonObject] = {}
    final_response: JsonObject | None = None

    for event in _sse_events(text):
        final_response = _completed_response(event)
        if final_response is not None:
            break
        _apply_sse_output_event(output, event)
    return _response_with_rebuilt_output(final_response, output)


async def responses_call(  # noqa: PLR0911 - fail-open routing declines to passthrough.
    *,
    body: JsonObject,
    tier_cfg: JsonObject,
    credential: str,
    auth_mode: Literal["api_key", "oauth"] = "api_key",
    chatgpt_headers: list[tuple[str, str]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[JsonObject, tuple[int, int, int, int], shrink_stats.ShrinkStats] | None:
    """POST a sanitized Responses request to a Codex tier.

    Args:
        body: Original OpenAI Responses request body from the client.
        tier_cfg: Tier config marked as a Codex tier.
        credential: Resolved credential value.
        auth_mode: Resolved credential mode.
        chatgpt_headers: Forwardable ChatGPT OAuth headers for OAuth mode.
        client: Optional shared HTTP client for tests or server reuse.

    Returns:
        Parsed Responses JSON body, normalized token usage, and shrink stats,
        or None when the caller must passthrough.
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
        orphans = unpaired_function_call_ids(payload)
        if orphans:
            logger.info(
                "codex direct declined: %d unanswered function_call(s) in input "
                "(stateless backend would 400); passing through",
                len(orphans),
            )
            return None
        payload, shrink = forward_compression.compress_responses(
            payload, enable_text_ml=bool(tier_cfg.get("text_ml_compression"))
        )
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
            return (
                (parsed_sse, _response_usage(parsed_sse), shrink)
                if parsed_sse is not None
                else None
            )
        parsed_response = cast("JsonValue", response.json())
        if not isinstance(parsed_response, dict):
            return None
        return parsed_response, _response_usage(parsed_response), shrink
    except (ConfigSchemaError, httpx.HTTPError, ValueError) as exc:
        logger.warning("codex direct call failed (%s); passing through", exc, exc_info=True)
        return None
