"""MCPScreen render + flow tests — CI-safe (no npx / no real server).

The interactive `/mcp` panel is driven by `localcode.mcp.server_status()`.
These tests patch that (plus connect_all / shutdown_all / disconnect) so the
screen mounts in Textual's headless driver with deterministic state, and assert
it renders all three honest statuses (connected / failed / not-connected), the
empty state, that a 14+ tool list expands without being truncated, and that the
reload and close keys work.

A separate real-server pilot (run by hand against
@modelcontextprotocol/server-filesystem) covers the live path; this file is the
regression net that keeps CI green without a network or npx.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from textual.app import App

from localcode.tui.screens.mcp_screen import MCPScreen


def _tools(n: int) -> list[dict]:
    return [
        {"name": f"tool_{i}", "description": f"does thing {i}", "inputSchema": {}}
        for i in range(n)
    ]


# Three servers exercising every honest status the client can report.
_STATUS_MIXED = [
    {"name": "files", "transport": "stdio", "oauth": False,
     "connected": True, "tools": _tools(14), "error": None},
    {"name": "brokensrv", "transport": "stdio", "oauth": False,
     "connected": False, "tools": [],
     "error": "brokensrv: command not found: nope"},
    {"name": "remote", "transport": "http", "oauth": True,
     "connected": False, "tools": [], "error": None},
]


class _Harness(App):
    def __init__(self) -> None:
        super().__init__()
        self.dismissed = "UNSET"

    async def on_mount(self) -> None:
        self.push_screen(MCPScreen(), lambda r: setattr(self, "dismissed", r))


async def _wait_status(scr, pilot, timeout=3.0):
    for _ in range(int(timeout / 0.02)):
        await pilot.pause(0.02)
        if scr._status is not None:
            return scr._status
    return scr._status


def _run(coro):
    import asyncio
    asyncio.run(coro)


# ── connected / failed / off rendering ──────────────────────────────

def test_renders_all_three_statuses():
    async def scenario():
        with patch("localcode.mcp.connect_all", return_value=(1, [])), \
             patch("localcode.mcp.server_status", return_value=list(_STATUS_MIXED)):
            app = _Harness()
            async with app.run_test(size=(100, 40)) as pilot:
                scr = app.screen
                await _wait_status(scr, pilot)
                body = scr._render_body()
                # header count
                assert "1 of 3 connected" in body, body
                # connected server: glyph + tool count
                assert "files" in body and "connected · 14 tools" in body, body
                # failed server: error surfaced
                assert "brokensrv" in body and "failed:" in body, body
                assert "command not found" in body, body
                # not-connected server: off state, oauth surfaced (from config)
                assert "remote" in body and "not connected" in body, body
                assert "oauth" in body, body
    _run(scenario())


# ── expand a 14-tool server → tools must all render, not truncated ──

def test_expand_long_tool_list_not_truncated():
    async def scenario():
        with patch("localcode.mcp.connect_all", return_value=(1, [])), \
             patch("localcode.mcp.server_status", return_value=list(_STATUS_MIXED)):
            app = _Harness()
            async with app.run_test(size=(100, 40)) as pilot:
                scr = app.screen
                await _wait_status(scr, pilot)
                await pilot.press("enter")  # expand focused (files, 14 tools)
                await pilot.pause(0.05)
                assert scr._level == scr._LEVEL_TOOLS
                body = scr._render_body()
                assert "14 tools" in body, body
                # First AND last tool present — nothing dropped; the box windows
                # via its own overflow scroll, it does not truncate the content.
                assert "tool_0" in body and "tool_13" in body, body
                # Navigate to the last tool and confirm scroll-into-view runs.
                for _ in range(13):
                    await pilot.press("down")
                await pilot.pause(0.05)
                assert scr._focused_idx == 13
                # Esc backs out to the servers level (does not close).
                await pilot.press("escape")
                await pilot.pause(0.05)
                assert scr._level == scr._LEVEL_SERVERS
                assert app.dismissed == "UNSET"
    _run(scenario())


# ── empty state ─────────────────────────────────────────────────────

def test_empty_state():
    async def scenario():
        with patch("localcode.mcp.connect_all", return_value=(0, [])), \
             patch("localcode.mcp.server_status", return_value=[]):
            app = _Harness()
            async with app.run_test(size=(100, 40)) as pilot:
                scr = app.screen
                await _wait_status(scr, pilot)
                body = scr._render_body()
                assert "None configured" in body, body
                # New empty state points at the in-TUI add flow, not raw JSON.
                assert "Press" in body and "to add one" in body, body
    _run(scenario())


# ── reload key re-derives status ────────────────────────────────────

def test_reload_key_refreshes_status():
    async def scenario():
        # Start empty; reload flips to a connected server.
        seq = [[], list(_STATUS_MIXED)]

        def _status():
            return seq.pop(0) if len(seq) > 1 else seq[0]

        with patch("localcode.mcp.connect_all", return_value=(1, [])), \
             patch("localcode.mcp.shutdown_all") as shut, \
             patch("localcode.mcp.server_status", side_effect=_status):
            app = _Harness()
            async with app.run_test(size=(100, 40)) as pilot:
                scr = app.screen
                await _wait_status(scr, pilot)
                assert scr._render_body().count("None configured") == 1
                await pilot.press("r")
                # wait for the reload worker to land the second status
                for _ in range(150):
                    await pilot.pause(0.02)
                    if scr._status:
                        break
                assert shut.called, "reload must shutdown_all first"
                body = scr._render_body()
                assert "files" in body and "connected · 14 tools" in body, body
    _run(scenario())


# ── close key dismisses ─────────────────────────────────────────────

def test_close_key_dismisses():
    async def scenario():
        with patch("localcode.mcp.connect_all", return_value=(1, [])), \
             patch("localcode.mcp.server_status", return_value=list(_STATUS_MIXED)):
            app = _Harness()
            async with app.run_test(size=(100, 40)) as pilot:
                scr = app.screen
                await _wait_status(scr, pilot)
                await pilot.press("q")
                await pilot.pause(0.05)
                assert app.dismissed is None, app.dismissed
    _run(scenario())
