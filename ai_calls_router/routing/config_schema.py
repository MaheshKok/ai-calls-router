"""Pydantic schemas for routing configuration payloads.

The decision core still consumes plain JSON-like dictionaries, but these
schemas validate config boundaries before those dictionaries drive provider
selection or credential handling. Validation errors are converted back to the
router's existing fail-open paths by callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from ai_calls_router._lib.types import JsonObject, JsonValue


class ConfigSchemaError(ValueError):
    """Raised when a routing config payload fails schema validation."""

    def __init__(self, message: str, *, group: str | None = None) -> None:
        """Initialize schema validation error metadata.

        Args:
            message: Human-readable validation failure.
            group: Provider group whose payload failed validation, when known.
        """
        super().__init__(message)
        self.group = group


class _SchemaModel(BaseModel):
    """Base model for config schemas that preserve future fields."""

    model_config = ConfigDict(extra="allow", strict=True)


class TierAuthConfig(_SchemaModel):
    """Schema for one tier auth declaration."""

    mode: Literal["api_key_env"]
    key_env: str | None = Field(default=None, min_length=1)


class TierConfig(_SchemaModel):
    """Schema for one cheap-tier entry."""

    model: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    key_env: str | None = Field(default=None, min_length=1)
    auth: TierAuthConfig | None = None
    max_tokens: int | None = Field(default=None, gt=0)


class ServerConfig(_SchemaModel):
    """Schema for server-level proxy settings."""

    host: str | None = Field(default=None, min_length=1)
    port: int | None = Field(default=None, gt=0)
    upstream: str | None = Field(default=None, min_length=1)


class SettingsConfig(_SchemaModel):
    """Schema for global routing settings."""

    env_file: str | None = Field(default=None, min_length=1)
    tier_precedence: list[str] | None = None
    compress_routed: bool | None = None
    premium_tools: list[str] | None = None


class AgentConfig(_SchemaModel):
    """Schema for one canonical agent config entry."""

    tools: dict[str, str] | None = None
    premium_tools: list[str] | None = None
    tiers: dict[str, TierConfig] | None = None
    upstream: str | None = Field(default=None, min_length=1)
    premium: dict[str, str] | None = None


class UserAgentRule(_SchemaModel):
    """Schema for one router.user_agent_map rule."""

    contains: str = Field(min_length=1)
    group: str = Field(min_length=1)


class RouterConfig(_SchemaModel):
    """Schema for agent identity router settings."""

    endpoint_defaults: dict[str, str] | None = None
    user_agent_map: list[UserAgentRule] | None = None
    fallback: str | None = None


class RoutesConfig(_SchemaModel):
    """Schema for the assembled global route config."""

    server: ServerConfig | None = None
    settings: SettingsConfig | None = None
    tiers: dict[str, TierConfig] | None = None
    router: RouterConfig | None = None
    agents: dict[str, AgentConfig] | None = None


class ProviderPayloadConfig(_SchemaModel):
    """Schema for one provider YAML payload."""

    group: Literal["claude_code", "hermes"]
    upstream: str = Field(min_length=1)
    auth: dict[str, str] | None = None
    wire: str = Field(min_length=1)
    endpoints: list[str]
    tools: dict[str, str] | None = None
    premium_tools: list[str] | None = None


def _schema_error(exc: ValidationError, *, group: str | None = None) -> ConfigSchemaError:
    """Convert pydantic errors to the router's local error type."""
    return ConfigSchemaError(exc.errors()[0]["msg"], group=group)


def _contains_forbidden_key_env(value: JsonValue, path: tuple[str, ...] = ()) -> bool:
    """Return whether a provider payload contains a non-auth key_env field."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if key == "key_env" and child_path[-2:] != ("auth", "key_env"):
                return True
            if _contains_forbidden_key_env(child, child_path):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_key_env(item, path) for item in value)
    return False


def validate_routes_payload(routes: JsonObject) -> None:
    """Validate a parsed global routes payload."""
    try:
        RoutesConfig.model_validate(routes)
    except ValidationError as exc:
        raise _schema_error(exc) from exc


def parse_tier_config(tier_cfg: JsonObject) -> TierConfig:
    """Validate and return one tier config model."""
    try:
        return TierConfig.model_validate(tier_cfg)
    except ValidationError as exc:
        raise _schema_error(exc) from exc


def validate_provider_payload(group: str, payload: JsonObject) -> ProviderPayloadConfig:
    """Validate one provider payload against the pydantic schema."""
    try:
        parsed = ProviderPayloadConfig.model_validate(payload)
    except ValidationError as exc:
        raise _schema_error(exc, group=group) from exc
    if _contains_forbidden_key_env(payload):
        raise ConfigSchemaError("provider config must not carry cheap key_env", group=group)
    return parsed
