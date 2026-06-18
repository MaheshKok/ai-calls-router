"""Boundary tests for shared JSON numeric coercers."""

from __future__ import annotations

import pytest

from ai_calls_router._lib import jsonnum
from ai_calls_router._lib.types import JsonValue


def test_jsonnum_coerce_boundaries() -> None:
    usage: dict[str, JsonValue] = {}
    very_large = 10**30
    cases: list[tuple[JsonValue, int]] = [
        (None, 0),
        (usage.get("input_tokens", 0), 0),
        ("0", 0),
        ("3", 3),
        (3.0, 3),
        (-4, -4),
        (very_large, very_large),
        ("x", 0),
        (True, 0),
        ({}, 0),
    ]

    for value, expected in cases:
        assert jsonnum.int_value(value) == expected


def test_int_value_minimum_clamps_negative() -> None:
    usage: dict[str, JsonValue] = {}
    assert jsonnum.int_value(usage.get("input_tokens", 0), minimum=0) == 0
    assert jsonnum.int_value(-4, minimum=0) == 0


def test_int_value_strict_bad_string_raises() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        jsonnum.int_value("x", strict=True)


def test_int_value_can_preserve_bool_as_int() -> None:
    assert jsonnum.int_value(True, bool_as_int=True) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.0),
        ("3.5", 3.5),
        (3, 3.0),
        ("x", 0.0),
        (False, 0.0),
    ],
)
def test_float_value_defaults(value: JsonValue, expected: float) -> None:
    assert jsonnum.float_value(value) == expected


def test_optional_float_value_matches_price_config_semantics() -> None:
    assert jsonnum.optional_float_value(3) == 3.0
    assert jsonnum.optional_float_value(3.5) == 3.5
    assert jsonnum.optional_float_value("3.5") is None
    assert jsonnum.optional_float_value(True) is None
