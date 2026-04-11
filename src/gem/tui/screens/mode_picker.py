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
        height: 3;
        padding: 1 1 1 1;
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
        border: solid $primary;
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
        left_cols = 13
        line = f"{left} {'─' * (usable - left_cols)}"
        self.query_one("#picker-header", Static).update(line)

    def _save(self) -> None:
        from ...config import save_config
        save_config(self.app.gem_config)

    def _select_mode(self, mode: str) -> None:
        self.app.gem_config.runtime.laptop_26b_runtime_mode = mode
        self._save()
        # Verify server is reachable before entering chat
        from ...runtime import GemRuntimeGateway
        gw = GemRuntimeGateway(self.app.gem_config.runtime)
        ok, details = gw.healthcheck()
        if ok:
            self.app.switch_screen("chat")
        else:
            self.notify(f"Server not ready: {details}", severity="error")

    def action_select_fast(self) -> None:
        self._select_mode("turbo")

    def action_select_reasoning(self) -> None:
        self._select_mode("turbo-think")

    def action_quit(self) -> None:
        self.app.exit()
