"""Shared structural types for JSON-like request and config data.

The proxy mostly transforms parsed JSON/YAML payloads. These aliases keep that
boundary explicit without using Any or unparameterized containers.
"""

from __future__ import annotations

from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]
