"""Golden-pair fidelity and spec tests for ai_calls_router.conversion.

The golden fixtures in fixtures/conversion_golden.json were produced by
running Headroom's ORIGINAL production converters (RoutedLiteLLMBackend)
under Python 3.10 via fixtures/generate_golden.py. The extracted functions
must produce identical output under Python 3.13. Spec tests cover the
contracts the goldens cannot: message id format, input immutability, and
the acompletion kwargs assembly (no client auth or unknown keys leak).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_calls_router import conversion

GOLDEN: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "conversion_golden.json").read_text(
        encoding="utf-8"
    )
)

MSG_ID_PATTERN = re.compile(r"^msg_[0-9a-f]{24}$")


def _cases(section: str) -> list[pytest.param]:
    """Build parametrize cases from a golden fixture section.

    Args:
        section: Top-level key in conversion_golden.json.

    Returns:
        pytest.param list of (input, expected) pairs, one per named case.
    """
    return [
        pytest.param(case["input"], case["expected"], id=name)
        for name, case in GOLDEN[section].items()
    ]


def _make_response(spec: dict[str, Any]) -> SimpleNamespace:
    """Build an attribute-style litellm response object from a JSON spec.

    Mirrors fixtures/generate_golden.py:make_response so both Pythons feed
    the converter structurally identical objects (kept duplicated because
    the generator imports Headroom and only runs under 3.10).

    Args:
        spec: JSON-serializable response description.

    Returns:
        SimpleNamespace mirroring litellm's ModelResponse attribute access.
    """
    choices = []
    for choice_spec in spec["choices"]:
        msg_spec = choice_spec["message"]
        tool_calls = msg_spec.get("tool_calls")
        if tool_calls is not None:
            tool_calls = [
                SimpleNamespace(
                    id=tc["id"],
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ]
        message = SimpleNamespace(content=msg_spec.get("content"), tool_calls=tool_calls)
        choices.append(
            SimpleNamespace(finish_reason=choice_spec["finish_reason"], message=message)
        )
    return SimpleNamespace(choices=choices, usage=SimpleNamespace(**spec.get("usage", {})))


class TestGoldenProvenance:
    def test_goldens_came_from_routed_backend(self) -> None:
        # The fixtures must capture the routed production behavior (with the
        # reasoning_content shim), not the base backend fallback.
        assert GOLDEN["generator"]["backend"] == "RoutedLiteLLMBackend"


class TestConvertMessagesGolden:
    @pytest.mark.parametrize(("body", "expected"), _cases("convert_messages"))
    def test_matches_original(self, body: list[dict[str, Any]], expected: Any) -> None:
        assert conversion.convert_messages_for_litellm(body) == expected


class TestConvertToolGolden:
    @pytest.mark.parametrize(("tool", "expected"), _cases("convert_tool"))
    def test_matches_original(self, tool: dict[str, Any], expected: Any) -> None:
        assert conversion.convert_anthropic_tool(tool) == expected


class TestConvertToolChoiceGolden:
    @pytest.mark.parametrize(("choice", "expected"), _cases("convert_tool_choice"))
    def test_matches_original(self, choice: Any, expected: Any) -> None:
        assert conversion.convert_tool_choice(choice) == expected


class TestParseToolArgumentsGolden:
    @pytest.mark.parametrize(("arguments", "expected"), _cases("parse_tool_arguments"))
    def test_matches_original(self, arguments: Any, expected: Any) -> None:
        assert conversion.parse_tool_arguments(arguments) == expected


class TestToAnthropicResponseGolden:
    @pytest.mark.parametrize(("spec", "expected"), _cases("to_anthropic_response"))
    def test_matches_original(self, spec: dict[str, Any], expected: Any) -> None:
        result = conversion.to_anthropic_response(_make_response(spec), "claude-sonnet-4-6")
        msg_id = result.pop("id")
        assert MSG_ID_PATTERN.match(msg_id)
        assert result == expected


class TestToAnthropicResponseSpec:
    def test_message_ids_are_unique_per_call(self) -> None:
        spec = GOLDEN["to_anthropic_response"]["text_stop"]["input"]
        first = conversion.to_anthropic_response(_make_response(spec), "m")
        second = conversion.to_anthropic_response(_make_response(spec), "m")
        assert first["id"] != second["id"]

    def test_model_field_echoes_requested_model(self) -> None:
        spec = GOLDEN["to_anthropic_response"]["text_stop"]["input"]
        result = conversion.to_anthropic_response(_make_response(spec), "any-model-id")
        assert result["model"] == "any-model-id"

    def test_message_without_tool_calls_attribute(self) -> None:
        # hasattr(message, "tool_calls") is False for raw objects without it.
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="plain")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )
        result = conversion.to_anthropic_response(response, "m")
        assert result["content"] == [{"type": "text", "text": "plain"}]
        assert result["stop_reason"] == "end_turn"


class TestConvertMessagesSpec:
    def test_input_is_not_mutated(self) -> None:
        body = GOLDEN["convert_messages"]["full_tool_round_trip"]["input"]
        snapshot = copy.deepcopy(body)
        conversion.convert_messages_for_litellm(body)
        assert body == snapshot

    def test_reasoning_content_not_added_to_user_messages(self) -> None:
        converted = conversion.convert_messages_for_litellm(
            [{"role": "user", "content": "hi"}]
        )
        assert "reasoning_content" not in converted[0]

    def test_reasoning_content_preserved_if_already_set(self) -> None:
        # setdefault contract: an existing value must never be overwritten.
        # String-content assistant messages pass through as new dicts, so this
        # exercises the backfill on the converted copy only.
        converted = conversion.convert_messages_for_litellm(
            [{"role": "assistant", "content": "thought"}]
        )
        assert converted[0]["reasoning_content"] == ""


class TestBackendResponse:
    def test_defaults(self) -> None:
        response = conversion.BackendResponse(body={"x": 1})
        assert response.status_code == 200
        assert response.headers == {}
        assert response.error is None

    def test_headers_not_shared_between_instances(self) -> None:
        a = conversion.BackendResponse(body={})
        b = conversion.BackendResponse(body={})
        a.headers["k"] = "v"
        assert b.headers == {}


class TestCompletionKwargs:
    def test_minimal_body(self) -> None:
        kwargs = conversion.completion_kwargs(
            {"model": "deepseek/x", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert kwargs == {
            "model": "deepseek/x",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_optional_params_copied_only_when_present(self) -> None:
        body = {
            "model": "m",
            "messages": [],
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END"],
        }
        kwargs = conversion.completion_kwargs(body)
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["stop"] == ["END"]
        assert "stop_sequences" not in kwargs

    def test_tools_and_tool_choice_converted(self) -> None:
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
        }
        kwargs = conversion.completion_kwargs(body)
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {"name": "Bash", "parameters": {"type": "object"}},
            }
        ]
        assert kwargs["tool_choice"] == "required"

    def test_system_string_inserted_first(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be brief",
        }
        kwargs = conversion.completion_kwargs(body)
        assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
        assert kwargs["messages"][1]["role"] == "user"

    def test_system_list_joined_with_spaces(self) -> None:
        body = {
            "model": "m",
            "messages": [],
            "system": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        }
        kwargs = conversion.completion_kwargs(body)
        assert kwargs["messages"][0] == {"role": "system", "content": "one two"}

    def test_api_key_included_only_when_provided(self) -> None:
        body = {"model": "m", "messages": []}
        assert "api_key" not in conversion.completion_kwargs(body)
        kwargs = conversion.completion_kwargs(body, api_key="sk-tier-key")
        assert kwargs["api_key"] == "sk-tier-key"

    def test_unknown_body_keys_never_leak(self) -> None:
        # Invariant 2 support: nothing from the client request reaches the
        # provider except the explicitly converted fields.
        body = {
            "model": "m",
            "messages": [],
            "stream": True,
            "metadata": {"user_id": "abc"},
            "anthropic_beta": ["x"],
        }
        kwargs = conversion.completion_kwargs(body)
        assert set(kwargs) == {"model", "messages"}

    def test_input_body_is_not_mutated(self) -> None:
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "s",
        }
        snapshot = copy.deepcopy(body)
        conversion.completion_kwargs(body)
        assert body == snapshot
