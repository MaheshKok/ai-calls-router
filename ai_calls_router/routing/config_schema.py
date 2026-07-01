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

CODEX_OAUTH_SENTINEL = "oauth"


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

    mode: Literal["api_key_env", "oauth"]
    key_env: str | None = Field(default=None, min_length=1)


class TierConfig(_SchemaModel):
    """Schema for one cheap-tier entry."""

    model: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    key_env: str | None = Field(default=None, min_length=1)
    auth: TierAuthConfig | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    # Optional reasoning level forced on this tier's routed turns, one of the five
    # Anthropic levels. "xhigh" is accepted by Sonnet 5 and Opus but rejected by
    # older routed models (e.g. Sonnet 4.6, HTTP 400); pair it with
    # supports_xhigh_effort on tiers that name an xhigh-capable model. Premium
    # passthrough is never routed and keeps its own level, set in the client.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # Whether this tier's routed model accepts effort="xhigh". Default False. Set
    # True for xhigh-capable models (Sonnet 5, Opus) so a client-sent xhigh passes
    # through instead of being downgraded to high. A configured "effort" above
    # wins regardless; this only governs the safety downgrade of a client level.
    supports_xhigh_effort: bool = False
    # Opt this tier's routed turns into headroom's lossy ML plain-text compressor
    # (Kompress). Off by default: the lossless content compressors always run, so
    # installing the ML extra never changes behaviour until a tier sets this True.
    # Leave False for tiers serving coding agents where dropping prose tokens from
    # tool output could corrupt context.
    text_ml_compression: bool = False
    # Whether this tier's routed model accepts Anthropic's adaptive thinking.
    # Default True. Set False for the Claude 4.5 family (e.g. claude-haiku-4-5):
    # its subscription Messages endpoint returns HTTP 400 ("adaptive thinking is
    # not supported on this model") for both a top-level thinking={type:adaptive}
    # request and a non-empty output_config.effort (which it treats as adaptive),
    # failing the routed turn open to premium. When False, prepare_routed_body
    # disables the top-level adaptive thinking and strips effort before the POST.
    # No model registry carries this endpoint-specific fact (litellm's
    # supports_reasoning is True for Haiku), so the tier owns it. 4.6+ models
    # accept adaptive thinking and keep this True.
    supports_adaptive_thinking: bool = True
    # Total token context window (input+output) of this tier's routed model,
    # e.g. 200000 for Sonnet 5 / Haiku 4.5. When set, the router uses the prior
    # turn's observed usage to skip routing a turn projected to overflow the
    # window -- which would 400 ("prompt is too long") and fail open to premium.
    # Omit to disable the guard; routing then always attempts and relies on
    # fail-open. Premium (Opus, ~1M) is never routed, so it needs no window here.
    context_window: int | None = None


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
    anthropic_prompt_cache: bool | None = None
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


def is_codex_tier(tier_cfg: JsonObject) -> bool:
    """Return whether a tier targets Codex subscription routing."""
    parsed = parse_tier_config(tier_cfg)
    if parsed.provider in {"codex", "openai-codex"}:
        return True
    return parsed.model.startswith(("codex/", "openai-codex/"))


def validate_provider_payload(group: str, payload: JsonObject) -> ProviderPayloadConfig:
    """Validate one provider payload against the pydantic schema."""
    try:
        parsed = ProviderPayloadConfig.model_validate(payload)
    except ValidationError as exc:
        raise _schema_error(exc, group=group) from exc
    if _contains_forbidden_key_env(payload):
        raise ConfigSchemaError("provider config must not carry cheap key_env", group=group)
    return parsed
