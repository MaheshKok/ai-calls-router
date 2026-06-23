"""Pydantic schemas for OpenAI-compatible client request envelopes.

Converters still preserve the original dictionaries so request-direction
serialization remains deterministic. These schemas validate the outer wire
shape before conversion reaches routing decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypeAliasType

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject

JsonSchemaValue = TypeAliasType(
    "JsonSchemaValue",
    None | bool | int | float | str | list["JsonSchemaValue"] | dict[str, "JsonSchemaValue"],
)
JsonSchemaObject = dict[str, JsonSchemaValue]


class OpenAISchemaError(ValueError):
    """Raised when an OpenAI-compatible request envelope is malformed."""


class _OpenAIModel(BaseModel):
    """Base model preserving forward-compatible OpenAI fields."""

    model_config = ConfigDict(extra="allow", strict=True)


class ChatMessage(_OpenAIModel):
    """Schema for one Chat Completions message."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: JsonSchemaValue = None
    tool_calls: list[JsonSchemaObject] | None = None
    tool_call_id: str | None = None


class ChatRequest(_OpenAIModel):
    """Schema for the Chat Completions request envelope."""

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[JsonSchemaObject] | None = None
    tool_choice: JsonSchemaValue = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    stop: JsonSchemaValue = None


class ResponsesRequest(_OpenAIModel):
    """Schema for the OpenAI Responses request envelope."""

    model: str = Field(min_length=1)
    input: str | list[JsonSchemaObject]
    instructions: str | None = None
    tools: list[JsonSchemaObject] | None = None
    tool_choice: JsonSchemaValue = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    stop: JsonSchemaValue = None
    stream: bool | None = None
    store: bool | None = None


def _error(exc: ValidationError) -> OpenAISchemaError:
    """Return a compact schema error preserving pydantic's first failure."""
    return OpenAISchemaError(exc.errors()[0]["msg"])


def validate_chat_request(body: JsonObject) -> ChatRequest:
    """Validate a Chat Completions request envelope."""
    try:
        return ChatRequest.model_validate(body)
    except ValidationError as exc:
        raise _error(exc) from exc


def validate_responses_request(body: JsonObject) -> ResponsesRequest:
    """Validate an OpenAI Responses request envelope."""
    try:
        return ResponsesRequest.model_validate(body)
    except ValidationError as exc:
        raise _error(exc) from exc
