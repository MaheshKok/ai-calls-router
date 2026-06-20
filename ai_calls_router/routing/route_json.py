"""Small JSON-shape coercion helpers shared across the routing layer.

The routing modules read loosely typed config and request payloads where a value
may legitimately be a mapping, a list, or something malformed. These helpers
narrow a ``JsonValue`` to the expected container shape, returning an empty
container on mismatch so callers stay fail-open without repeating isinstance
guards. They hold no policy and depend on nothing else in the routing package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonArray, JsonObject, JsonValue


def json_mapping(value: JsonValue) -> JsonObject:
    """Return a JSON object when value is mapping-shaped, else an empty mapping.

    Args:
        value: A parsed JSON value of unknown shape.

    Returns:
        The value when it is a ``dict``, otherwise a new empty mapping.
    """
    if isinstance(value, dict):
        return value
    return {}


def json_array(value: JsonValue) -> JsonArray:
    """Return a JSON array when value is list-shaped, else an empty list.

    Args:
        value: A parsed JSON value of unknown shape.

    Returns:
        The value when it is a ``list``, otherwise a new empty list.
    """
    if isinstance(value, list):
        return value
    return []
