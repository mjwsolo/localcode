"""Mode picker screen — press 1 for Fast, 2 for Reasoning."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center
from textual.screen import Screen
from textual.widgets import Static
from textual.binding import Binding


class ModePickerScreen(Screen):
    """Press 1 or 2 to select mode. No mouse needed."""

    BINDINGS = [
        Binding("1", "select_fast", "Fast", show=False),
        Binding("2", "select_reasoning", "Reasoning", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    DEFAULT_CSS = """
    ModePickerScreen {
        align: center middle;
        layout: vertical;
    }
    #header-rule {
        text-align: center;
        color: $primary;
        margin-bottom: 1;
        width: 100%;
    }
    #picker-box {
        width: 50;
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("──────────── 🏠 [bold]localcode[/] ────────────", id="header-rule")
        with Center():
            yield Static(
                "Select a mode:\n\n"
                "  [bold]1.[/] Fast - quick answers\n"
                "  [bold]2.[/] Reasoning - deep thinking\n\n"
                "[dim]Press 1 or 2[/]",
                id="picker-box",
            )

    def action_select_fast(self) -> None:
        self.app.gem_config.runtime.laptop_26b_runtime_mode = "turbo"
        self.app.switch_screen("chat")

    def action_select_reasoning(self) -> None:
        self.app.gem_config.runtime.laptop_26b_runtime_mode = "turbo-think"
        self.app.switch_screen("chat")

    def action_quit(self) -> None:
        self.app.exit()
