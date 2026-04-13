"""Mode picker screen — press 1 for Fast, 2 for Reasoning."""
from __future__ import annotations

from textual.app import ComposeResult
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
        layout: vertical;
    }
    #picker-header {
        dock: top;
        height: 1;
        padding: 0 1;
        color: #5f87ff;
        background: $surface;
    }
    #picker-center {
        height: 1fr;
        align: center middle;
    }
    #picker-box {
        width: 52;
        height: auto;
        padding: 1 2;
        border: round #5f87ff;
    }
    """

    def compose(self) -> ComposeResult:
        from textual.containers import Container
        yield Static("", id="picker-header")
        with Container(id="picker-center"):
            yield Static(
                    "Select a mode:\n\n"
                    "  [bold]1.[/] Fast — quick answers\n"
                    "  [bold]2.[/] Reasoning — deep thinking\n\n"
                    "[dim]Press 1 or 2[/]",
                    id="picker-box",
                )

    def on_mount(self) -> None:
        self._update_header()

    def on_resize(self) -> None:
        self._update_header()

    def _update_header(self) -> None:
        try:
            width = self.app.size.width or 80
        except Exception:
            width = 80
        usable = width - 2
        left = "🏠 LocalCode"
        left_cols = 14
        remaining = max(0, usable - left_cols)
        line = f"{left} {'─' * remaining}"
        self.query_one("#picker-header", Static).update(line)

    def _save(self) -> None:
        from ...config import save_config
        save_config(self.app.gem_config)

    def _select_mode(self, mode: str) -> None:
        self.app.gem_config.runtime.laptop_26b_runtime_mode = mode
        self._save()
        # Setup screen already verified server is ready — go straight to chat
        self.app.switch_screen("chat")

    def action_select_fast(self) -> None:
        self._select_mode("turbo")

    def action_select_reasoning(self) -> None:
        self._select_mode("turbo-think")

    def action_quit(self) -> None:
        self.app.exit()
