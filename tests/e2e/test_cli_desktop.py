"""End-to-end CLI coverage for the acr desktop subcommand."""

from __future__ import annotations

import json
from pathlib import Path

from ai_calls_router import cli


def test_desktop_on_status_off_round_trip_with_tmp_config(
    *,
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    """The desktop CLI operates only on the requested settings file in tests."""
    settings_path = tmp_path / "settings.json"
    acr_home = tmp_path / "acr-home"
    monkeypatch.setenv("ACR_HOME", str(acr_home))

    assert cli.main(["desktop", "on", "--config", str(settings_path)]) == 0
    on_output = capsys.readouterr().out
    assert "http://127.0.0.1:8747" in on_output
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8747"}
    }

    assert cli.main(["desktop", "status", "--config", str(settings_path)]) == 0
    status_output = capsys.readouterr().out
    assert "on" in status_output
    assert "http://127.0.0.1:8747" in status_output

    assert cli.main(["desktop", "off", "--config", str(settings_path)]) == 0
    off_output = capsys.readouterr().out
    assert "disabled" in off_output
    assert not settings_path.exists()
