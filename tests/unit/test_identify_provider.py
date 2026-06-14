"""Tests for the identify_provider classifier."""

from __future__ import annotations

from ai_calls_router.accounting.metrics import identify_provider


class TestIdentifyProvider:
    """identify_provider should map a model string to a provider label."""

    def test_anthropic(self) -> None:
        assert identify_provider("claude-sonnet-4-20250514") == "anthropic"
        assert identify_provider("claude-3-opus-20240229") == "anthropic"
        assert identify_provider("claude-3-haiku-20240307") == "anthropic"

    def test_openai(self) -> None:
        assert identify_provider("gpt-4o") == "openai"
        assert identify_provider("gpt-4-turbo") == "openai"
        assert identify_provider("o1-mini") == "openai"
        assert identify_provider("gpt-3.5-turbo") == "openai"
        assert identify_provider("chatgpt-4o-latest") == "openai"

    def test_deepseek(self) -> None:
        assert identify_provider("deepseek/deepseek-v4-flash") == "deepseek"
        assert identify_provider("deepseek-chat") == "deepseek"

    def test_google(self) -> None:
        assert identify_provider("gemini-1.5-pro") == "google"
        assert identify_provider("gemini-1.5-flash") == "google"

    def test_aws_via_bedrock(self) -> None:
        assert identify_provider("bedrock/anthropic.claude-sonnet-4") == "aws"
        assert identify_provider("bedrock/anthropic.claude-3-haiku") == "aws"

    def test_azure(self) -> None:
        assert identify_provider("azure/gpt-4") == "azure"

    def test_meta(self) -> None:
        assert identify_provider("llama-3.2-90b") == "meta"
        assert identify_provider("meta-llama/Llama-3.3-70b") == "meta"

    def test_mistral(self) -> None:
        assert identify_provider("mistral-large") == "mistral"
        assert identify_provider("mistral/mistral-small") == "mistral"

    def test_cohere(self) -> None:
        assert identify_provider("command-r-plus") == "cohere"

    def test_groq(self) -> None:
        assert identify_provider("groq/llama-3-70b") == "groq"

    def test_fireworks(self) -> None:
        assert identify_provider("fireworks/llama-v3") == "fireworks"

    def test_perplexity(self) -> None:
        assert identify_provider("sonar-pro") == "perplexity"

    def test_together(self) -> None:
        assert identify_provider("together/llama-3-70b") == "together"

    def test_unknown(self) -> None:
        assert identify_provider("") == "unknown"
        assert identify_provider("some-random-model") == "unknown"

    def test_none(self) -> None:
        assert identify_provider(None) == "unknown"
