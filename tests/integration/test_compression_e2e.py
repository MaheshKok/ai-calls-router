"""End-to-end proof that forward compression reaches the wire and the dashboard.

Drives the real ASGI app: a premium-passthrough turn whose earlier custom-tool
outputs are large JSON arrays is sent through ``/v1/messages``. headroom never
compresses its default-excluded coding tools (Bash/Read/Edit/Grep/Glob/Write --
their outputs are exact reference data), but a custom tool like ``search_files``
is fair game -- so the mock upstream must receive a table-compressed body, and
the ``/metrics`` snapshot that feeds the dashboard must report the realized
character saving. Skipped when the optional ``headroom`` extra is absent, since
compression is best-effort.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.acr_testkit import Upstream, make_client

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("headroom", reason="forward compression needs the headroom extra")

# Edit is a premium tool -> the turn escalates to premium passthrough, which is
# the path that gained forward compression. The fast tier exists only so routing
# is enabled; it is never selected for an Edit tool-result turn.
COMPRESSION_CONFIG = """
server:
  port: 8748
settings:
  tier_precedence: [premium, fast]
  premium_tools: [Edit]
  escalate_on_premium_tools: true
tiers:
  fast:
    model: deepseek/acr-e2e-cheap
    key_env: ACR_TEST_KEY
    max_tokens: 1000
tools:
  Edit: premium
"""


def _json_array(n: int) -> str:
    """Return a JSON array of uniform objects (SmartCrusher's table target)."""
    return json.dumps(
        [{"id": i, "name": f"item_{i}", "status": "ok", "value": i * 7} for i in range(n)]
    )


def _premium_body(tool_outputs: list[str]) -> dict[str, object]:
    """Build search_files tool turns carrying ``tool_outputs``, then a pending Edit.

    ``search_files`` is a custom tool, so its outputs are compressible (unlike the
    default-excluded Bash/Read/Edit/Grep/Glob/Write coding tools); the trailing
    Edit tool_use makes the turn escalate to premium passthrough, which is the
    forward-compression path under test.
    """
    messages: list[dict[str, object]] = [{"role": "user", "content": "run the suite"}]
    for i, out in enumerate(tool_outputs):
        cid = f"b{i}"
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": cid, "name": "search_files", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": cid, "content": out}],
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "applied"}],
        }
    )
    return {"model": "claude-e2e", "max_tokens": 1000, "messages": messages}


def test_premium_passthrough_compresses_json_array_and_records_it(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
) -> None:
    body = _premium_body([_json_array(150) for _ in range(4)])
    raw = json.dumps(body)

    with make_client(
        config_yaml=COMPRESSION_CONFIG,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as client:
        response = client.post(
            "/v1/messages", content=raw, headers={"content-type": "application/json"}
        )
        assert response.status_code == 200

        # The premium turn reached the upstream with a compressed body.
        assert len(upstream.requests) == 1
        forwarded = upstream.requests[0].content.decode("utf-8")
        assert len(forwarded) < len(raw)
        # SmartCrusher's lossless table marker, absent from the raw array body.
        assert "[150]{" in forwarded

        snapshot = client.get("/metrics").json()

    compression = snapshot["compression"]
    assert compression["tokens_saved"] > 0
    assert compression["ratio"] > 0
    assert compression["tokens_before"] > compression["tokens_after"]
    last = snapshot["last_requests"][0]
    assert last["shrink_chars_before"] > last["shrink_chars_after"]


def test_premium_passthrough_leaves_non_array_output_byte_identical(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
) -> None:
    # Plain text tool output is not array-shaped -> SmartCrusher passes through,
    # so the forwarded body must stay byte-identical (no needless cache break).
    body = _premium_body(["just some plain text output, nothing to crush here"])
    raw = json.dumps(body)

    with make_client(
        config_yaml=COMPRESSION_CONFIG,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as client:
        response = client.post(
            "/v1/messages", content=raw, headers={"content-type": "application/json"}
        )
        assert response.status_code == 200
        assert len(upstream.requests) == 1
        forwarded = upstream.requests[0].content.decode("utf-8")
        assert forwarded == raw

        snapshot = client.get("/metrics").json()

    # The compression aggregate is a process-global running total, so assert on
    # this request's own row (newest-first): it recorded no shrink.
    last = snapshot["last_requests"][0]
    assert last["shrink_chars_before"] == last["shrink_chars_after"]


def test_excluded_file_tool_output_is_never_compressed(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: Upstream
) -> None:
    # A JSON array emitted by Read (a file tool) must pass through untouched:
    # headroom keeps Read/Edit/Grep/Glob/Write outputs verbatim for edit matching.
    messages: list[dict[str, object]] = [{"role": "user", "content": "read the data"}]
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "r0", "name": "Read", "input": {}}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r0", "content": _json_array(150)}],
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "applied"}],
        }
    )
    body = {"model": "claude-e2e", "max_tokens": 1000, "messages": messages}
    raw = json.dumps(body)

    with make_client(
        config_yaml=COMPRESSION_CONFIG,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
    ) as client:
        response = client.post(
            "/v1/messages", content=raw, headers={"content-type": "application/json"}
        )
        assert response.status_code == 200
        assert len(upstream.requests) == 1
        forwarded = upstream.requests[0].content.decode("utf-8")
        assert forwarded == raw
        assert "[150]{" not in forwarded

        snapshot = client.get("/metrics").json()

    # Request-scoped row (newest-first): the excluded Read output saved nothing.
    last = snapshot["last_requests"][0]
    assert last["shrink_chars_before"] == last["shrink_chars_after"]
