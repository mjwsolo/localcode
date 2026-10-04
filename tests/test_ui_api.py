"""Discovery of the existing local model endpoint, without starting a model."""
from __future__ import annotations

import json

from localcode import entrypoint
from localcode.ui import api


def test_api_discovery_hides_key_by_default(monkeypatch, capsys):
    monkeypatch.setattr(api, "find_running", lambda: (8323, {"port": 8123, "current": "gemma", "state": "ready"}))
    monkeypatch.setattr(api, "read_auth_file", lambda _: ("control-secret", "server-secret"))
    assert api.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"base_url": "http://127.0.0.1:8123/v1", "model": "localcode/gemma", "state": "ready"}


def test_api_discovery_shows_key_only_when_requested(monkeypatch, capsys):
    monkeypatch.setattr(api, "find_running", lambda: (8323, {"port": 8123, "current": "gemma"}))
    monkeypatch.setattr(api, "read_auth_file", lambda _: ("control-secret", "server-secret"))
    assert api.main(["--json", "--show-key"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["api_key"] == "server-secret"
    assert "control-secret" not in json.dumps(result)


def test_api_discovery_requires_running_server(monkeypatch, capsys):
    monkeypatch.setattr(api, "find_running", lambda: None)
    assert api.main([]) == 1
    assert "start `localcode` first" in capsys.readouterr().err


def test_api_subcommand_routes_to_discovery(monkeypatch):
    called = []
    monkeypatch.setattr(api, "main", lambda argv: called.append(argv) or 0)
    try:
        entrypoint.main(["api", "--json"])
    except SystemExit as exc:
        assert exc.code == 0
    assert called == [["--json"]]


def test_api_command_is_discoverable_in_help():
    assert "localcode api" in entrypoint.build_parser().format_help()
