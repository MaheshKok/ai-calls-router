#!/usr/bin/env python3
"""Option B P0 spike: validate cheaper Codex tiers against the real backend.

Consumes WebSocket frames captured by the env-gated tee in
``websocket_passthrough.py`` (``$ACR_SPIKE_DIR/frames.jsonl`` plus
``headers.json``) and runs the feasibility checks for the stateful WS router.

Real protocol (discovered via capture): each ``response.create`` is a FLAT
frame (fields at top level) with ``store: false`` and a ``previous_response_id``
chaining to the prior turn; ``input`` is DELTA-only (just the new
``function_call_output`` items). The prior turn's ``function_call`` and encrypted
``reasoning`` live server-side, streamed earlier as ``response.output_item.done``
frames (``response.completed`` carries an EMPTY ``output``). To route a turn the
router must reconstruct the FULL input by accumulating every prior delta input
and every prior streamed output item, then replay statelessly.

Checks:
  S1  Per-turn shape and reconstructed-input sizes (offline).
  S2  Replay turn 1's reconstructed full input to gpt-5.3-codex-spark with
      ``image_generation`` stripped, plus a control that keeps it. (network)
  S3  Same against gpt-5.4-mini. (network)
  S4  Cross-model reasoning: replay a later turn whose history contains gpt-5.5
      encrypted reasoning to gpt-5.3-codex-spark, with and without the reasoning
      items, to decide free-routing vs tier-pinning. (network)

WARNING: S2-S4 send real conversation data and the captured OAuth token to
``https://chatgpt.com/backend-api/codex/responses`` and are billed to the user's
ChatGPT account. Run only with consent, then delete ``$ACR_SPIKE_DIR`` (it holds
a live token). This script never prints the token; provider bodies are redacted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

BACKEND_URL = "https://chatgpt.com/backend-api/codex/responses"
SPARK_MODEL = "gpt-5.3-codex-spark"
MINI_MODEL = "gpt-5.4-mini"
HOSTED_TOOL_TO_STRIP = "image_generation"
DROP_TOP_LEVEL = frozenset({"type", "previous_response_id"})
REQUEST_TIMEOUT_SECONDS = 120.0
EXCERPT_CHARS = 500

_BEARER = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SENSITIVE = re.compile(
    r'("(?:access_token|refresh_token|id_token|api_key|authorization)"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    """Return a redacted, bounded excerpt safe to print."""
    cleaned = _BEARER.sub("Bearer [redacted]", text.strip())
    cleaned = _SENSITIVE.sub(r'\1"[redacted]"', cleaned)
    return cleaned[:EXCERPT_CHARS] + ("...<truncated>" if len(cleaned) > EXCERPT_CHARS else "")


def _spike_dir() -> Path:
    """Return the capture directory from $ACR_SPIKE_DIR or exit with guidance."""
    raw = os.environ.get("ACR_SPIKE_DIR")
    if not raw:
        sys.exit("ACR_SPIKE_DIR is not set; point it at the capture directory.")
    path = Path(raw)
    if not (path / "frames.jsonl").exists():
        sys.exit(f"no frames.jsonl in {path}; run a Codex session through the tee first.")
    return path


def _load_frames(spike_dir: Path) -> list[dict[str, Any]]:
    """Return captured frame records (direction, ts, frame)."""
    lines = (spike_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_headers(spike_dir: Path) -> dict[str, str]:
    """Return replay headers from the captured upstream header pairs."""
    pairs = json.loads((spike_dir / "headers.json").read_text(encoding="utf-8"))
    headers = {str(key): str(value) for key, value in pairs}
    headers["Content-Type"] = "application/json"
    return headers


def _parse(frame: str) -> dict[str, Any] | None:
    """Return a parsed JSON frame object, or None when it is not an object."""
    try:
        parsed = json.loads(frame)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _reconstruct_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild each turn's full stateless input from delta inputs + streamed output.

    Walks frames in order, accumulating a conversation ``history``. Each
    ``response.create`` turn records its delta input and the full input a
    stateless backend needs (history so far + this delta). Output items stream as
    ``response.output_item.done`` before ``response.completed``; on completion the
    delta and those output items are appended to history for the next turn.

    Returns:
        One dict per turn: ``create`` (flat frame), ``delta``, ``output``,
        ``full_input``.
    """
    turns: list[dict[str, Any]] = []
    history: list[Any] = []
    pending: dict[str, Any] | None = None
    for record in records:
        parsed = _parse(record.get("frame", ""))
        if parsed is None:
            continue
        msg_type = parsed.get("type")
        if record.get("dir") == "c2u" and msg_type == "response.create":
            delta = parsed.get("input") if isinstance(parsed.get("input"), list) else []
            pending = {
                "create": parsed,
                "delta": delta,
                "output": [],
                "full_input": [*history, *delta],
            }
        elif record.get("dir") == "u2c" and pending is not None:
            if msg_type == "response.output_item.done":
                item = parsed.get("item")
                if isinstance(item, dict):
                    pending["output"].append(item)
            elif msg_type == "response.completed":
                turns.append(pending)
                history = [*history, *pending["delta"], *pending["output"]]
                pending = None
    return turns


def _item_types(items: list[Any]) -> list[str]:
    """Return the ordered ``type`` of each dict item."""
    return [str(i.get("type")) for i in items if isinstance(i, dict)]


def _has_reasoning(items: list[Any]) -> bool:
    """Return whether any item is a reasoning item."""
    return any(isinstance(i, dict) and i.get("type") == "reasoning" for i in items)


def _strip_reasoning(items: list[Any]) -> list[Any]:
    """Return items with reasoning items removed."""
    return [i for i in items if not (isinstance(i, dict) and i.get("type") == "reasoning")]


def _strip_image_tool(tools: object) -> tuple[list[Any], bool]:
    """Return tools without the image_generation hosted tool, and whether it was present."""
    if not isinstance(tools, list):
        return [], False
    kept = [t for t in tools if not (isinstance(t, dict) and t.get("type") == HOSTED_TOOL_TO_STRIP)]
    return kept, len(kept) != len(tools)


def _build_payload(
    create: dict[str, Any], full_input: list[Any], *, model: str, strip_image: bool
) -> dict[str, Any]:
    """Build a stateless OAuth Responses payload from a captured create frame.

    Drops ``type``/``previous_response_id`` (stateless replay), pins the routed
    model, swaps in the reconstructed full input, optionally strips the
    image_generation tool, and forces SSE streaming (OAuth requires it).
    """
    payload = {key: value for key, value in create.items() if key not in DROP_TOP_LEVEL}
    payload["model"] = model
    payload["input"] = full_input
    payload["stream"] = True
    if strip_image:
        payload["tools"], _ = _strip_image_tool(payload.get("tools"))
    return payload


def _post(headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, str]:
    """POST a payload to the backend, returning status and a redacted excerpt."""
    try:
        response = httpx.post(
            BACKEND_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except httpx.HTTPError as exc:
        return -1, f"<transport error: {exc}>"
    return response.status_code, _redact(response.text)


def _report(label: str, status: int, excerpt: str) -> None:
    """Print one probe result line."""
    print(f"  {label}: {'PASS (200)' if status == 200 else f'FAIL ({status})'}")
    if status != 200:
        print(f"    body: {excerpt}")


def _approx_tokens(payload: dict[str, Any]) -> int:
    """Return a rough token proxy (chars/4) for the serialized payload."""
    return len(json.dumps(payload)) // 4


def _run_s1(turns: list[dict[str, Any]]) -> None:
    """Print the offline per-turn shape and reconstruction summary."""
    print("== S1  per-turn shape (offline) ==")
    print(f"  turns: {len(turns)}")
    for index, turn in enumerate(turns):
        create = turn["create"]
        print(
            f"  turn[{index}]: delta={len(turn['delta'])} "
            f"full_input={len(turn['full_input'])} "
            f"reasoning_in_history={_has_reasoning(turn['full_input'])} "
            f"prev_id={'yes' if create.get('previous_response_id') else 'no'}"
        )


def _run_replay(headers: dict[str, str], turn: dict[str, Any], model: str, tag: str) -> None:
    """Replay one turn's reconstructed full input to a routed model (stripped + control)."""
    create = turn["create"]
    _, had_image = _strip_image_tool(create.get("tools"))
    stripped = _build_payload(create, turn["full_input"], model=model, strip_image=True)
    print(
        f"== {tag}  {model}  (turn full_input items={len(turn['full_input'])}, "
        f"~{_approx_tokens(stripped)} tok) =="
    )
    status, excerpt = _post(headers, stripped)
    _report(f"image_generation STRIPPED (had it: {had_image})", status, excerpt)
    kept = _build_payload(create, turn["full_input"], model=model, strip_image=False)
    ctrl_status, ctrl_excerpt = _post(headers, kept)
    _report("control: image_generation KEPT", ctrl_status, ctrl_excerpt)


def _first_clean_turn(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first turn with non-empty input and no reasoning in history."""
    for turn in turns:
        if turn["full_input"] and not _has_reasoning(turn["full_input"]):
            return turn
    return None


def _first_reasoning_turn(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first turn whose reconstructed input carries reasoning items."""
    for turn in turns:
        if _has_reasoning(turn["full_input"]):
            return turn
    return None


def _run_s4(headers: dict[str, str], turns: list[dict[str, Any]]) -> None:
    """Probe whether gpt-5.5 encrypted reasoning is portable to gpt-5.3-codex-spark."""
    print("== S4  cross-model reasoning (gpt-5.5 reasoning -> spark) ==")
    turn = _first_reasoning_turn(turns)
    if turn is None:
        print("  SKIP: no turn with reasoning in reconstructed history.")
        return
    create = turn["create"]
    with_payload = _build_payload(create, turn["full_input"], model=SPARK_MODEL, strip_image=True)
    status_a, excerpt_a = _post(headers, with_payload)
    _report("S4a: history WITH gpt-5.5 reasoning items", status_a, excerpt_a)
    without = _strip_reasoning(turn["full_input"])
    without_payload = _build_payload(create, without, model=SPARK_MODEL, strip_image=True)
    status_b, excerpt_b = _post(headers, without_payload)
    _report("S4b: history WITHOUT reasoning items", status_b, excerpt_b)
    print("  interpret: A pass => free per-turn routing viable; A fail & B pass =>")
    print("             reasoning is model-bound (strip on switch); both fail => investigate.")


def main() -> int:
    """Run the S1-S4 spike probes against captured frames."""
    parser = argparse.ArgumentParser(description="Option B P0 spike probe.")
    parser.add_argument("--offline", action="store_true", help="Run S1 only; no network.")
    args = parser.parse_args()

    spike_dir = _spike_dir()
    turns = _reconstruct_turns(_load_frames(spike_dir))
    _run_s1(turns)
    if args.offline:
        print("\n(offline mode: skipped S2-S4 network probes)")
        return 0

    headers = _load_headers(spike_dir)
    clean = _first_clean_turn(turns)
    if clean is None:
        print("\nno clean (reasoning-free) turn captured; skipping S2/S3.")
    else:
        _run_replay(headers, clean, SPARK_MODEL, "S2")
        _run_replay(headers, clean, MINI_MODEL, "S3")
    _run_s4(headers, turns)
    print("\nDone. Delete the capture dir now -- it holds a live OAuth token:")
    print(f"  rm -rf {spike_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
