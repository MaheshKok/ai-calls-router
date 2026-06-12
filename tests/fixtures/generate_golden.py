"""Golden fixture generator for conversion fidelity tests.

Runs Headroom's ORIGINAL conversion functions (the proven production code)
under Python 3.10 on representative Anthropic Messages API inputs and dumps
their outputs to conversion_golden.json. tests/test_conversion.py then
asserts the extracted ai_calls_router.conversion functions produce identical
output under Python 3.13.

Run with: /usr/local/bin/python3.10 tests/fixtures/generate_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path.home() / ".headroom" / "extensions" / "tool-router"))

from headroom.backends.litellm import (  # noqa: E402
    LiteLLMBackend,
    _convert_anthropic_tool,
    _convert_tool_choice,
    _parse_tool_arguments,
)

try:
    from headroom_tool_router.llm import RoutedLiteLLMBackend as _RoutedBackend
except Exception:  # pragma: no cover - fallback if extension import breaks
    _RoutedBackend = None


# ---------------------------------------------------------------------------
# Fixture inputs (shared spec; test_conversion.py replays the same inputs)
# ---------------------------------------------------------------------------

MESSAGE_CASES: dict[str, list[dict[str, Any]]] = {
    "string_content_user": [{"role": "user", "content": "hello"}],
    "string_content_assistant": [{"role": "assistant", "content": "hi there"}],
    "missing_role_defaults_user": [{"content": "no role"}],
    "text_blocks_joined": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ],
        }
    ],
    "empty_content_list": [{"role": "user", "content": []}],
    "non_dict_blocks_ignored": [
        {"role": "user", "content": ["raw string", {"type": "text", "text": "kept"}]}
    ],
    "tool_use_with_text": [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"cmd": "ls"}},
            ],
        }
    ],
    "tool_use_without_text": [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_02", "name": "Read", "input": {"path": "/x"}}
            ],
        }
    ],
    "tool_use_missing_input": [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_03", "name": "Glob"}],
        }
    ],
    "tool_result_string_content": [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "file.txt"}
            ],
        }
    ],
    "tool_result_list_content": [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
        }
    ],
    "tool_result_text_alongside_discarded": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "this text is discarded"},
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "result"},
            ],
        }
    ],
    "tool_result_missing_content": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_09"}]}
    ],
    "multiple_tool_results": [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "first"},
                {"type": "tool_result", "tool_use_id": "toolu_02", "content": "second"},
            ],
        }
    ],
    "non_string_non_list_content_dropped": [
        {"role": "user", "content": None},
        {"role": "user", "content": "kept"},
    ],
    "full_tool_round_trip": [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running ls."},
                {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"cmd": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": "a.py\nb.py"}
            ],
        },
    ],
}

TOOL_CASES: dict[str, dict[str, Any]] = {
    "full_tool": {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
    },
    "no_description": {"name": "Read", "input_schema": {"type": "object"}},
    "no_input_schema": {"name": "Glob", "description": "Find files"},
    "empty_tool": {},
}

TOOL_CHOICE_CASES: dict[str, Any] = {
    "string_auto": "auto",
    "string_none": "none",
    "dict_auto": {"type": "auto"},
    "dict_any": {"type": "any"},
    "dict_tool": {"type": "tool", "name": "Bash"},
    "dict_tool_no_name": {"type": "tool"},
    "dict_unknown_type": {"type": "mystery"},
    "empty_dict": {},
    "non_str_non_dict": 42,
}

PARSE_ARGS_CASES: dict[str, Any] = {
    "valid_json_string": '{"cmd": "ls", "n": 1}',
    "invalid_json_string": "not json at all",
    "json_null_string": "null",
    "empty_string": "",
    "dict_passthrough": {"already": "parsed"},
    "none_passthrough": None,
}

# Each response spec is JSON; both the generator and the 3.13 test rebuild
# attribute-style objects from it before calling the converter.
RESPONSE_CASES: dict[str, dict[str, Any]] = {
    "text_stop": {
        "choices": [
            {"finish_reason": "stop", "message": {"content": "Done.", "tool_calls": None}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    },
    "tool_calls_finish": {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "Bash", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    },
    "text_and_tool_calls": {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Checking now.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "Read", "arguments": "broken{json"},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    },
    "length_finish": {
        "choices": [
            {"finish_reason": "length", "message": {"content": "truncat", "tool_calls": None}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    },
    "content_filter_finish": {
        "choices": [
            {"finish_reason": "content_filter", "message": {"content": "x", "tool_calls": None}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    },
    "unknown_finish_reason": {
        "choices": [
            {"finish_reason": "weird", "message": {"content": "y", "tool_calls": None}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    },
    "empty_content_no_text_block": {
        "choices": [{"finish_reason": "stop", "message": {"content": "", "tool_calls": None}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0},
    },
    "missing_usage_fields": {
        "choices": [
            {"finish_reason": "stop", "message": {"content": "z", "tool_calls": None}}
        ],
        "usage": {},
    },
}


def make_response(spec: dict[str, Any]) -> SimpleNamespace:
    """Build an attribute-style litellm response object from a JSON spec.

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


def main() -> None:
    """Generate conversion_golden.json from Headroom's original converters."""
    if _RoutedBackend is not None:
        backend = object.__new__(_RoutedBackend)
        backend_name = "RoutedLiteLLMBackend"
    else:
        backend = object.__new__(LiteLLMBackend)
        backend_name = "LiteLLMBackend"

    messages_golden = {}
    for name, msgs in MESSAGE_CASES.items():
        converted = backend._convert_messages_for_litellm(msgs)
        if _RoutedBackend is None:
            # Replicate the routed reasoning_content shim (llm.py:63-67)
            # verbatim when the extension itself is not importable.
            for msg in converted:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    msg.setdefault("reasoning_content", "")
        messages_golden[name] = {"input": msgs, "expected": converted}

    tools_golden = {
        name: {"input": tool, "expected": _convert_anthropic_tool(tool)}
        for name, tool in TOOL_CASES.items()
    }
    tool_choice_golden = {
        name: {"input": choice, "expected": _convert_tool_choice(choice)}
        for name, choice in TOOL_CHOICE_CASES.items()
    }
    parse_args_golden = {
        name: {"input": args, "expected": _parse_tool_arguments(args)}
        for name, args in PARSE_ARGS_CASES.items()
    }

    responses_golden = {}
    for name, spec in RESPONSE_CASES.items():
        result = backend._to_anthropic_response(make_response(spec), "claude-sonnet-4-6")
        # The message id is a fresh uuid per call; fidelity is asserted on
        # everything else, the id format is asserted separately in the tests.
        result.pop("id")
        responses_golden[name] = {"input": spec, "expected": result}

    golden = {
        "generator": {
            "python": sys.version.split()[0],
            "backend": backend_name,
        },
        "convert_messages": messages_golden,
        "convert_tool": tools_golden,
        "convert_tool_choice": tool_choice_golden,
        "parse_tool_arguments": parse_args_golden,
        "to_anthropic_response": responses_golden,
    }

    out = Path(__file__).parent / "conversion_golden.json"
    out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} (backend={backend_name}, python={sys.version.split()[0]})")


if __name__ == "__main__":
    main()
