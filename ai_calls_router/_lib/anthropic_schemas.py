"""Pydantic schemas for Anthropic Messages request envelopes.

Anthropic Messages is the router's canonical internal shape, so validation here
must not copy or normalize successful requests. It only rejects malformed
external envelopes early so the server can fail open to passthrough.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypeAliasType

JsonSchemaValue = TypeAliasType(
    "JsonSchemaValue",
    None | bool | int | float | str | list["JsonSchemaValue"] | dict[str, "JsonSchemaValue"],
)


class AnthropicSchemaError(ValueError):
    """Raised when an Anthropic Messages request envelope is malformed."""


class _AnthropicModel(BaseModel):
    """Base model preserving Anthropic beta and metadata fields."""

    model_config = ConfigDict(extra="allow", strict=True)


class AnthropicMessage(_AnthropicModel):
    """Schema for one Anthropic message."""

    role: Literal["user", "assistant"]
    content: JsonSchemaValue


class AnthropicMessagesRequest(_AnthropicModel):
    """Schema for an Anthropic Messages request envelope."""

    model: str = Field(min_length=1)
    messages: list[AnthropicMessage] = Field(min_length=1)
    system: JsonSchemaValue = None
    tools: list[dict[str, JsonSchemaValue]] | None = None
    tool_choice: JsonSchemaValue = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: JsonSchemaValue = None
    stream: bool | None = None


def validate_anthropic_messages_request(body: dict[str, JsonSchemaValue]) -> None:
    """Validate an Anthropic Messages request envelope."""
    try:
        AnthropicMessagesRequest.model_validate(body)
    except ValidationError as exc:
        raise AnthropicSchemaError(exc.errors()[0]["msg"]) from exc
