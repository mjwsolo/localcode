"""Add-an-MCP-server modal — the in-TUI answer to `claude mcp add`.

localcode has no CLI subcommands; everything happens in the TUI. So adding an
MCP server is a small keyboard form here, not a shell command. It writes
``~/.localcode/mcp.json`` for the user (via ``mcp.add_mcp_server``) so nobody
has to hand-edit JSON.

The common case is the one Codex/Claude make trivial: "use this MCP" + a URL.
So the URL/command field comes first, and:

  * a value starting with ``http://`` / ``https://`` becomes a remote
    streamable-HTTP server (``{"transport": "http", "url": ...}``);
  * anything else is a stdio server (``{"command": ..., "args": [...]}``),
    with args split shell-style.

Name is optional: left blank, it is derived from the URL host (``api.
githubcopilot.com`` -> ``githubcopilot``) or the command/package basename.

Resolves to ``{"name": str, "entry": dict}`` on save, or ``None`` on cancel.
"""
from __future__ import annotations

import shlex
from urllib.parse import urlparse

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from ...theme import C


def _derive_name_from_url(url: str) -> str:
    """A short, sane server name from a URL host, e.g.
    https://api.githubcopilot.com/mcp/ -> 'githubcopilot'."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return ""
    labels = [p for p in host.split(".") if p]
    # Drop leading noise and a trailing TLD so the meaningful label is left.
    while labels and labels[0] in ("api", "mcp", "www", "app"):
        labels.pop(0)
    if len(labels) > 1:
        labels = labels[:-1]  # strip TLD (.com/.dev/.ai/...)
    return labels[0] if labels else host


def _derive_name_from_command(command: str, args: list[str]) -> str:
    """A name for a stdio server. Runners like npx/uvx carry the real name in
    args (@scope/server), so prefer a package-looking arg over the runner."""
    for a in args:
        if a.startswith("-"):
            continue
        if "/" in a or a.startswith("@"):
            base = a.rsplit("/", 1)[-1]  # @scope/server-foo -> server-foo
            if base:
                return base
    base = command.rsplit("/", 1)[-1]
    return base or command


class AddMCPServerScreen(ModalScreen[dict | None]):
    """Modal form to add one MCP server. Resolves to {name, entry} or None."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    AddMCPServerScreen {
        align: center middle;
    }
    AddMCPServerScreen > #add-box {
        width: 68;
        max-width: 92%;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    AddMCPServerScreen #add-title {
        text-style: bold;
        padding-bottom: 1;
    }
    AddMCPServerScreen #add-help {
        color: $text-muted;
        padding-bottom: 1;
    }
    AddMCPServerScreen Input {
        margin-bottom: 1;
    }
    AddMCPServerScreen #add-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="add-box"):
            yield Static("Add MCP server", id="add-title")
            yield Static(
                "Paste a URL for a remote server, or a command for a local one.",
                id="add-help",
            )
            yield Input(
                placeholder="https://… URL, or a command like: npx -y @scope/server /path",
                id="add-target",
            )
            yield Input(placeholder="name (optional)", id="add-name")
            yield Static(
                "Enter save  ·  Tab next field  ·  Esc cancel", id="add-hint"
            )

    def on_mount(self) -> None:
        self.query_one("#add-target", Input).focus()

    # ── build / validate ────────────────────────────────────────────

    def _build(self) -> dict | None:
        target = self.query_one("#add-target", Input).value.strip()
        name = self.query_one("#add-name", Input).value.strip()
        if not target:
            return None

        if target.lower().startswith(("http://", "https://")):
            entry: dict = {"transport": "http", "url": target}
            if not name:
                name = _derive_name_from_url(target)
        else:
            parts = shlex.split(target)
            if not parts:
                return None
            command, args = parts[0], parts[1:]
            entry = {"command": command}
            if args:
                entry["args"] = args
            if not name:
                name = _derive_name_from_command(command, args)

        if not name:
            return None
        return {"name": name, "entry": entry}

    def _flash(self, msg: str) -> None:
        try:
            self.query_one("#add-hint", Static).update(f"[{C.error}]{msg}[/]")
        except Exception:
            pass

    # ── actions ─────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in either field tries to save.
        result = self._build()
        if result is None:
            self._flash("Enter a URL or a command first.")
            return
        self.dismiss(result)

    def action_cancel(self) -> None:
        self.dismiss(None)
