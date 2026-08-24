"""Add-an-MCP-server flow — writer, name derivation, and the modal build.

Covers the in-TUI replacement for `claude mcp add`:
  * `mcp.add_mcp_server` / `remove_mcp_server` write and prune mcp.json while
    preserving other keys and honoring LOCALCODE_HOME;
  * the URL-first `AddMCPServerScreen` turns a pasted URL into an http entry
    and a command into a stdio entry, deriving a name when one is not given.

CI-safe: no network, no npx, no real server.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import localcode.mcp._config as cfg
from localcode.tui.screens.mcp_add import (
    AddMCPServerScreen,
    _derive_name_from_command,
    _derive_name_from_url,
)


# ── writer ──────────────────────────────────────────────────────────

def test_add_and_remove_preserve_other_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    path = tmp_path / "mcp.json"
    # A file that already has an unrelated top-level key must survive a write.
    path.write_text(json.dumps({"$comment": "keep me", "mcpServers": {}}))

    cfg.add_mcp_server("files", {"command": "npx", "args": ["-y", "srv"]})
    data = json.loads(path.read_text())
    assert data["$comment"] == "keep me"
    assert data["mcpServers"]["files"]["command"] == "npx"
    assert data["mcpServers"]["files"]["args"] == ["-y", "srv"]

    # Adding a second server keeps the first.
    cfg.add_mcp_server("remote", {"transport": "http", "url": "https://x/mcp"})
    data = json.loads(path.read_text())
    assert set(data["mcpServers"]) == {"files", "remote"}

    # Remove one; the other and the extra key remain.
    assert cfg.remove_mcp_server("files") is True
    data = json.loads(path.read_text())
    assert list(data["mcpServers"]) == ["remote"]
    assert data["$comment"] == "keep me"
    # Removing a missing server is a no-op returning False.
    assert cfg.remove_mcp_server("nope") is False


def test_add_creates_file_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))
    assert not (tmp_path / "mcp.json").exists()
    cfg.add_mcp_server("remote", {"transport": "http", "url": "https://x/mcp"})
    data = json.loads((tmp_path / "mcp.json").read_text())
    assert data == {"mcpServers": {"remote": {"transport": "http", "url": "https://x/mcp"}}}


# ── name derivation ─────────────────────────────────────────────────

def test_derive_name_from_url():
    assert _derive_name_from_url("https://api.githubcopilot.com/mcp/") == "githubcopilot"
    assert _derive_name_from_url("https://mcp.sentry.dev/sse") == "sentry"
    assert _derive_name_from_url("http://localhost:9000/mcp") == "localhost"


def test_derive_name_from_command():
    # Runner in command, real package in args -> name from the package.
    assert _derive_name_from_command(
        "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/p"]
    ) == "server-filesystem"
    # Bare command -> its basename.
    assert _derive_name_from_command("/usr/local/bin/my-server", []) == "my-server"


# ── modal build logic ───────────────────────────────────────────────

def _run(coro):
    import asyncio
    asyncio.run(coro)


def _build_with(target: str, name: str = ""):
    """Mount the modal, set field values, return what _build() produces."""
    from textual.app import App
    from textual.widgets import Input

    result = {}

    class _H(App):
        async def on_mount(self) -> None:
            self.push_screen(AddMCPServerScreen())

    async def scenario():
        app = _H()
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.pause(0.05)
            scr = app.screen
            scr.query_one("#add-target", Input).value = target
            scr.query_one("#add-name", Input).value = name
            result["out"] = scr._build()

    _run(scenario())
    return result["out"]


def test_build_url_makes_http_entry():
    out = _build_with("https://api.githubcopilot.com/mcp/")
    assert out == {
        "name": "githubcopilot",
        "entry": {"transport": "http", "url": "https://api.githubcopilot.com/mcp/"},
    }


def test_build_command_makes_stdio_entry_with_split_args():
    out = _build_with('npx -y "@scope/server foo" /path', name="myfiles")
    assert out["name"] == "myfiles"
    assert out["entry"]["command"] == "npx"
    # shlex.split keeps the quoted package as one arg.
    assert out["entry"]["args"] == ["-y", "@scope/server foo", "/path"]


def test_build_blank_target_is_invalid():
    assert _build_with("   ") is None


# ── end-to-end: 'a' from /mcp opens the modal, save writes + reloads ──

def test_add_key_opens_modal_and_saves(tmp_path, monkeypatch):
    from unittest.mock import patch
    from textual.widgets import Input
    from localcode.tui.screens.mcp_screen import MCPScreen

    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))

    async def scenario():
        from textual.app import App

        class _App(App):
            async def on_mount(self) -> None:
                self.push_screen(MCPScreen())

        # Empty catalogue so the screen mounts in its empty state fast.
        with patch("localcode.mcp.connect_all", return_value=(0, [])), \
             patch("localcode.mcp.server_status", return_value=[]):
            app = _App()
            async with app.run_test(size=(100, 40)) as pilot:
                mcp_scr = app.screen
                for _ in range(50):
                    await pilot.pause(0.02)
                    if mcp_scr._status is not None:
                        break
                await pilot.press("a")
                await pilot.pause(0.05)
                assert isinstance(app.screen, AddMCPServerScreen), app.screen
                app.screen.query_one("#add-target", Input).value = (
                    "https://api.githubcopilot.com/mcp/"
                )
                await pilot.press("enter")
                await pilot.pause(0.05)

    _run(scenario())

    # The save wrote the derived server into mcp.json.
    data = json.loads((tmp_path / "mcp.json").read_text())
    assert data["mcpServers"]["githubcopilot"] == {
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
    }
