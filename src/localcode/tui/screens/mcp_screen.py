"""Interactive MCP management screen — localcode's answer to Claude Code's
`/mcp` panel.

Two levels, mirroring the model picker's structure and CSS conventions:

  * Level 1 — the SERVERS list. One row per server configured in
    ``~/.localcode/mcp.json``, each drawn with an honest status glyph sourced
    from what the client can actually report (``mcp.server_status()``):
        ✔ connected · N tools
        ✘ failed: <error from connect_all>
        ○ not connected
    No invented auth states — a stdio server has no auth to report. The
    ``oauth`` flag is surfaced only when it comes straight from the config.
  * Level 2 — the focused server's TOOLS (name + short description), one per
    row, windowed via the box's own scroll so a 14+ tool list never overflows.

Keyboard-only (mouse is off by default): arrows / j-k to move, number hotkeys,
Enter to expand/collapse, ``r`` to reload all (shutdown_all + connect_all),
``d`` to disconnect the focused server (behind a ConfirmScreen), Esc/q to
close (or back out of the tool view). Connecting / reloading runs on a worker
thread so npx-spawned servers don't freeze the UI.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Static

from ...theme import C
from .confirm import ConfirmScreen


class MCPScreen(Screen):
    """Interactive MCP server + tool browser."""

    BINDINGS = [
        Binding("1", "pick(0)", "#1", show=False),
        Binding("2", "pick(1)", "#2", show=False),
        Binding("3", "pick(2)", "#3", show=False),
        Binding("4", "pick(3)", "#4", show=False),
        Binding("5", "pick(4)", "#5", show=False),
        Binding("6", "pick(5)", "#6", show=False),
        Binding("7", "pick(6)", "#7", show=False),
        Binding("8", "pick(7)", "#8", show=False),
        Binding("9", "pick(8)", "#9", show=False),
        Binding("up", "move(-1)", "Previous", show=False),
        Binding("k", "move(-1)", "Previous", show=False),
        Binding("down", "move(1)", "Next", show=False),
        Binding("j", "move(1)", "Next", show=False),
        Binding("enter", "expand_focused", "Expand / collapse", show=False),
        Binding("left", "back", "Back", show=False),
        Binding("h", "back", "Back", show=False),
        Binding("r", "reload", "Reload all", show=False),
        Binding("d", "disconnect", "Disconnect server", show=False),
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    # Navigation levels.
    _LEVEL_SERVERS = "servers"
    _LEVEL_TOOLS = "tools"

    # Status glyphs, coloured to match the palette. Only three states — the
    # honest set the client can actually report.
    _GLYPH_CONNECTED = f"[{C.success}]✔[/]"   # ✔
    _GLYPH_FAILED = f"[{C.error}]✘[/]"         # ✘
    _GLYPH_OFF = "[dim]○[/]"                    # ○

    DEFAULT_CSS = """
    MCPScreen {
        layout: vertical;
        background: ansi_default;
        padding: 1 0;
    }
    #mcp-center {
        background: ansi_default;
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    #mcp-box {
        background: ansi_default;
        width: 92%;
        max-width: 80;
        height: auto;
        max-height: 90%;        /* cap to viewport so long tool lists don't run off-screen */
        overflow-y: auto;       /* scroll instead of clipping */
        padding: 1 2;
        border: round #5f87ff;
        scrollbar-color: #333333;
        scrollbar-color-hover: #555555;
        scrollbar-color-active: #666666;
        scrollbar-background: ansi_default;
        scrollbar-size-vertical: 1;
    }
    #mcp-list {
        background: ansi_default;
        height: auto;
        width: 100%;
    }
    #mcp-footer {
        background: ansi_default;
        dock: bottom;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._level = self._LEVEL_SERVERS
        self._focused_idx = 0
        self._open_idx: int | None = None
        # Rows from mcp.server_status(); None = still loading (connecting).
        self._status: list[dict] | None = None
        self._busy = False  # True while a connect / reload worker is in flight

    # ── compose ─────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Container(id="mcp-center"):
            with Vertical(id="mcp-box"):
                yield Static(self._render_body(), id="mcp-list")
        yield Static(self._footer_markup(), id="mcp-footer")

    def on_mount(self) -> None:
        # Connect (or refresh status) on a worker thread so npx-spawned stdio
        # servers don't freeze the UI while they boot.
        self._busy = True
        self.run_worker(self._connect_worker, thread=True, exclusive=True)

    # ── workers ─────────────────────────────────────────────────────

    def _connect_worker(self) -> None:
        """Connect any not-yet-connected server, then read honest status."""
        from ...mcp import connect_all, server_status
        try:
            connect_all()
            status = server_status()
        except Exception:
            status = []
        self.app.call_from_thread(self._on_status_loaded, status)

    def _reload_worker(self) -> None:
        from ...mcp import connect_all, server_status, shutdown_all
        try:
            shutdown_all()
            connect_all()
            status = server_status()
        except Exception:
            status = []
        self.app.call_from_thread(self._on_status_loaded, status)

    def _on_status_loaded(self, status: list[dict]) -> None:
        self._status = status
        self._busy = False
        # Keep focus in range if the server list shrank.
        n = len(status)
        if n == 0:
            self._focused_idx = 0
            self._level = self._LEVEL_SERVERS
            self._open_idx = None
        elif self._focused_idx >= n:
            self._focused_idx = n - 1
        self._refresh()

    # ── render dispatch ─────────────────────────────────────────────

    def _render_body(self) -> str:
        if self._status is None:
            return "[bold]MCP servers[/]\n\n[dim]connecting…[/]"
        if not self._status:
            return self._render_empty()
        if self._level == self._LEVEL_TOOLS:
            return self._render_tools()
        return self._render_servers()

    def _footer_markup(self) -> str:
        if self._status is None or not self._status:
            return f"[{C.primary}]LocalCode[/] [dim]· r reload · Esc close[/]"
        if self._level == self._LEVEL_TOOLS:
            row = self._status[self._open_idx] if self._open_idx is not None else None
            name = row["name"] if row else ""
            name_bit = f"[dim]·[/] [bold]{name}[/] " if name else ""
            return (
                f"[{C.primary}]LocalCode[/] {name_bit}"
                "[dim]· ↑/↓ scroll · Enter/Esc/← back · r reload[/]"
            )
        n = len(self._status)
        keys = "Enter to expand" if n == 1 else f"↑/↓ + Enter, or 1-{n}"
        return (
            f"[{C.primary}]LocalCode[/] "
            f"[dim]· {keys} · r reload · d disconnect · Esc close[/]"
        )

    # ── Level 1: servers ────────────────────────────────────────────

    def _render_servers(self) -> str:
        lines = ["[bold]MCP servers[/]"]
        n_conn = sum(1 for s in self._status if s["connected"])
        lines.append(f"[dim]{n_conn} of {len(self._status)} connected[/]")
        lines.append("")

        for i, s in enumerate(self._status, start=1):
            focused = (i - 1 == self._focused_idx)
            chevron = "▸" if focused else " "  # ▸
            glyph, label = self._status_glyph_label(s)
            name = f"[bold]{s['name']}[/]"
            lines.append(f" [dim]{chevron}[/] [bold]{i}.[/] {glyph} {name}")
            # Status detail line, dim, middle-dot separated.
            bits = [s["transport"]]
            if s["oauth"]:
                bits.append("oauth")
            bits.append(label)
            lines.append(f"       [dim]" + " · ".join(bits) + "[/]")

        lines.append("")
        lines.append("[dim]Enter → tools · r reload · d disconnect · Esc close[/]")
        return "\n".join(lines)

    def _status_glyph_label(self, s: dict) -> tuple[str, str]:
        """(glyph, short status label) for a server row."""
        if s["connected"]:
            n = len(s["tools"])
            return self._GLYPH_CONNECTED, f"connected · {n} tool{'s' if n != 1 else ''}"
        if s["error"]:
            # Keep the error compact; it can be long.
            err = s["error"].replace("\n", " ")
            if len(err) > 44:
                err = err[:43] + "…"
            return self._GLYPH_FAILED, f"failed: {err}"
        return self._GLYPH_OFF, "not connected"

    # ── Level 2: tools of the open server ───────────────────────────

    def _render_tools(self) -> str:
        s = self._status[self._open_idx]
        glyph, label = self._status_glyph_label(s)
        lines = [f"[bold]{s['name']}[/] [dim]· {label}[/]", ""]

        tools = s["tools"]
        if not tools:
            if s["error"]:
                lines.append(f"[{C.error}]Not connected.[/]")
                lines.append(f"[dim]{s['error']}[/]")
            else:
                lines.append("[dim]No tools exposed by this server.[/]")
            lines.append("")
            lines.append("[dim]Esc/← back to servers[/]")
            return "\n".join(lines)

        for i, t in enumerate(tools):
            focused = (i == self._focused_idx)
            chevron = "▸" if focused else " "
            name = f"[bold]{t.get('name', '?')}[/]"
            desc = (t.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 52:
                desc = desc[:51] + "…"
            tail = f"[dim] · {desc}[/]" if desc else ""
            lines.append(f" [dim]{chevron}[/] {name}{tail}")

        lines.append("")
        lines.append(
            f"[dim]{len(tools)} tools · ↑/↓ scroll · Esc/← back to servers[/]"
        )
        return "\n".join(lines)

    # ── empty state ─────────────────────────────────────────────────

    def _render_empty(self) -> str:
        from ...mcp import MCP_CONFIG_PATH
        return (
            "[bold]MCP servers[/]\n"
            "\n"
            "[dim]None configured. MCP lets the model call tools from external\n"
            "programs you trust.[/]\n"
            "\n"
            f"[dim]Add one in[/] {MCP_CONFIG_PATH}[dim]:[/]\n"
            "\n"
            '[dim]{\n'
            '  "mcpServers": {\n'
            '    "files": {\n'
            '      "command": "npx",\n'
            '      "args": [\n'
            '        "-y", "@modelcontextprotocol/server-filesystem", "/path"\n'
            '      ]\n'
            '    }\n'
            '  }\n'
            '}[/]\n'
            "\n"
            "[dim]Then press r to connect.[/]"
        )

    # ── refresh / scroll ────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            self.query_one("#mcp-list", Static).update(self._render_body())
        except Exception:
            pass
        try:
            self.query_one("#mcp-footer", Static).update(self._footer_markup())
        except Exception:
            pass
        self._scroll_focus_into_view()

    def _focused_line(self) -> int:
        """Line offset of the focused row (2 header lines + 1 blank). Servers
        are 2 lines each, tools 1 line each."""
        if self._level == self._LEVEL_TOOLS:
            return 2 + self._focused_idx
        return 3 + self._focused_idx * 2

    def _scroll_focus_into_view(self) -> None:
        try:
            box = self.query_one("#mcp-box")
            line = self._focused_line()
            top = int(getattr(box.scroll_offset, "y", 0) or 0)
            height = int(getattr(box.size, "height", 0) or 0) or 12
            if line <= top:
                box.scroll_to(y=max(0, line - 1), animate=False)
            elif line >= top + height - 1:
                box.scroll_to(y=max(0, line - height + 2), animate=False)
        except Exception:
            pass

    # ── navigation actions ──────────────────────────────────────────

    def _current_count(self) -> int:
        if self._status is None or not self._status:
            return 0
        if self._level == self._LEVEL_TOOLS and self._open_idx is not None:
            return len(self._status[self._open_idx]["tools"])
        return len(self._status)

    def action_move(self, delta: int) -> None:
        n = self._current_count()
        if n == 0:
            return
        self._focused_idx = (self._focused_idx + delta) % n
        self._refresh()

    def action_expand_focused(self) -> None:
        """Enter — expand the focused server (Level 1) or collapse (Level 2)."""
        if self._status is None or not self._status:
            return
        if self._level == self._LEVEL_TOOLS:
            self._collapse()
            return
        self._expand_at(self._focused_idx)

    def action_pick(self, idx: int) -> None:
        if self._status is None or not self._status:
            return
        if self._level == self._LEVEL_TOOLS:
            return  # number keys are inert in the tool view
        if 0 <= idx < len(self._status):
            self._focused_idx = idx
            self._expand_at(idx)

    def _expand_at(self, idx: int) -> None:
        self._open_idx = idx
        self._level = self._LEVEL_TOOLS
        self._focused_idx = 0
        self._refresh()

    def _collapse(self) -> None:
        prev = self._open_idx if self._open_idx is not None else 0
        self._level = self._LEVEL_SERVERS
        self._open_idx = None
        self._focused_idx = prev
        self._refresh()

    def action_back(self) -> None:
        if self._level == self._LEVEL_TOOLS:
            self._collapse()

    def action_close(self) -> None:
        # At Level 2, Esc backs out to Level 1 rather than closing outright.
        if self._level == self._LEVEL_TOOLS:
            self._collapse()
            return
        self.dismiss(None)

    # ── reload ──────────────────────────────────────────────────────

    def action_reload(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._level = self._LEVEL_SERVERS
        self._open_idx = None
        self._status = None  # show "connecting…"
        self._refresh()
        self.run_worker(self._reload_worker, thread=True, exclusive=True)

    # ── disconnect (Level 1 only, behind a ConfirmScreen) ───────────

    def action_disconnect(self) -> None:
        if self._busy or self._level != self._LEVEL_SERVERS:
            return
        if self._status is None or not self._status:
            return
        if not (0 <= self._focused_idx < len(self._status)):
            return
        s = self._status[self._focused_idx]
        if not s["connected"]:
            self._flash_footer(
                f"[dim]{s['name']} isn't connected — nothing to disconnect.[/]"
            )
            return
        name = s["name"]

        def _after(ok: bool | None) -> None:
            if ok:
                self._do_disconnect(name)

        self.app.push_screen(
            ConfirmScreen(
                f"Disconnect {name}?",
                "Closes the live session. Press r to reconnect it later.",
                confirm_label="Disconnect",
            ),
            _after,
        )

    def _do_disconnect(self, name: str) -> None:
        from ...mcp import disconnect, server_status
        try:
            disconnect(name)
            self._status = server_status()
        except Exception:
            pass
        n = len(self._status or [])
        if n and self._focused_idx >= n:
            self._focused_idx = n - 1
        self._refresh()
        self._flash_footer(f"[dim]Disconnected {name}. Press r to reconnect.[/]")

    def _flash_footer(self, markup: str) -> None:
        try:
            self.query_one("#mcp-footer", Static).update(markup)
        except Exception:
            pass
