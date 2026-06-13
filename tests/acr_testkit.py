"""Shared test harness for the ai-calls-router suite.

Centralizes the helpers that drive the serving pipeline so the unit,
integration, and end-to-end suites assert against one source of truth: the
litellm stand-in, the litellm ModelResponse factory, the mock premium upstream,
the app-client wiring, and the savings-ledger reader. It is importable from any
test subdirectory via pytest's ``pythonpath = ["tests"]`` setting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from ai_calls_router.proxy.server import create_app


class FakeLitellm:
    """litellm module stand-in: serves acompletion and captures its kwargs.

    Records every acompletion call so tests can assert on the kwargs (model,
    api_key, messages) the routed path assembles, and either returns a canned
    response or raises an injected error to exercise the fail-open path.
    """

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        """Initialize with a canned response or an error to raise.

        Args:
            response: ModelResponse stand-in returned from acompletion.
            error: Exception raised from acompletion instead of returning.
        """
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error

    async def acompletion(self, **kwargs: Any) -> Any:
        """Record the call kwargs, then return the response or raise the error.

        Args:
            **kwargs: The acompletion call arguments under test.

        Returns:
            The canned response when no error was injected.

        Raises:
            Exception: The injected error, when one was supplied.
        """
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def make_response(
    text: str | None = "done",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 1000,
    completion_tokens: int = 200,
) -> Any:
    """Build a litellm ModelResponse stand-in (attribute access only).

    The defaults mirror a plain text completion; callers override token counts
    when a test pins specific savings math, and supply tool_calls /
    finish_reason to exercise the escalation guard.

    Args:
        text: Assistant message text (None when the reply is tool calls only).
        tool_calls: OpenAI-shaped tool call objects, or None.
        finish_reason: Completion finish reason reported by the provider.
        prompt_tokens: Prompt token count reported in the usage block.
        completion_tokens: Completion token count reported in the usage block.

    Returns:
        A SimpleNamespace shaped like a litellm ModelResponse.
    """
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class Upstream:
    """Mock premium upstream that records every proxied request.

    Used by app-level tests to prove that passthrough turns reach the upstream
    and routed turns do not, and to inspect the body forwarded on fallback.
    """

    def __init__(self) -> None:
        """Initialize with an empty request log."""
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer with a fixed upstream marker.

        Args:
            request: The proxied client request.

        Returns:
            A 200 JSON response carrying a recognizable upstream marker.
        """
        self.requests.append(request)
        return httpx.Response(
            200,
            content=b'{"marker": "upstream"}',
            headers={"content-type": "application/json"},
        )


def make_client(
    config_yaml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: Upstream,
) -> TestClient:
    """Wire env, config, and a mock upstream into a fresh proxy app.

    Args:
        config_yaml: Config file contents written to a temp path.
        tmp_path: Per-test temporary directory.
        monkeypatch: Pytest monkeypatch fixture for env isolation.
        upstream: Mock upstream whose handler backs the injected transport.

    Returns:
        A Starlette TestClient bound to the configured proxy app.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml, encoding="utf-8")
    monkeypatch.setenv("ACR_CONFIG", str(config_file))
    monkeypatch.setenv("ACR_TEST_KEY", "tier-key")
    monkeypatch.setenv("ACR_SAVINGS_LEDGER", str(tmp_path / "savings.jsonl"))
    app = create_app(transport=httpx.MockTransport(upstream.handler))
    return TestClient(app)


def read_ledger(tmp_path: Path) -> list[dict[str, Any]]:
    """Parse the savings ledger written during a test, if any.

    Args:
        tmp_path: Per-test temporary directory holding savings.jsonl.

    Returns:
        Parsed ledger entries, or an empty list when no ledger was written.
    """
    ledger = tmp_path / "savings.jsonl"
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
