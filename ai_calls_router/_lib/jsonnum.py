"""Small coercers for numeric values read from JSON-like data.

Several serving and accounting paths need defensive int/float conversion with
slightly different malformed-value behavior. Keeping those knobs here prevents
each caller from carrying its own near-copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonValue


def int_value(
    value: JsonValue,
    *,
    default: int = 0,
    minimum: int | None = None,
    strict: bool = False,
    bool_as_int: bool = False,
) -> int:
    """Coerce a JSON scalar to int with caller-selected bounds and strictness."""
    if isinstance(value, bool):
        result = int(value) if bool_as_int else default
    elif isinstance(value, int | float | str):
        try:
            result = int(value)
        except (TypeError, ValueError):
            if strict:
                raise
            result = default
    else:
        result = default
    return max(result, minimum) if minimum is not None else result


def float_value(value: JsonValue, *, default: float = 0.0, allow_strings: bool = True) -> float:
    """Coerce a JSON scalar to float, defaulting malformed values."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float) or (allow_strings and isinstance(value, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def optional_float_value(value: JsonValue, *, allow_strings: bool = False) -> float | None:
    """Coerce a JSON scalar to float or return None when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) or (allow_strings and isinstance(value, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None
