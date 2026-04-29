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
        background: ansi_default;
        padding: 1 0;
    }
    #picker-center {
        background: ansi_default;
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    #picker-box {
        background: ansi_default;
        width: 92%;
        max-width: 56;
        height: auto;
        padding: 1 2;
        border: round #5f87ff;
    }
    /* Brand at bottom — `#brand-bar` styled in tui/styles/app.tcss. */
    """

    def compose(self) -> ComposeResult:
        from textual.containers import Container
        # Brand at bottom-left, same `#brand-bar` shared across screens.
        from ...theme import C as _C
        yield Static(f"🏠[{_C.primary}]LocalCode[/]", id="brand-bar")
        with Container(id="picker-center"):
            yield Static(
                    "Select a mode:\n\n"
                    "  [bold]1.[/] Fast — quick answers\n"
                    "  [bold]2.[/] Reasoning — deep thinking\n\n"
                    "[dim]Press 1 or 2[/]",
                    id="picker-box",
                )

    def on_mount(self) -> None:
        return

    def on_resize(self) -> None:
        return

    def _save(self) -> None:
        from ...config import save_config
        save_config(self.app.config)

    def _select_mode(self, mode: str) -> None:
        self.app.config.runtime.laptop_26b_runtime_mode = mode
        self._save()
        # Setup screen already verified server is ready — go straight to chat
        self.app.switch_screen("chat")

    def action_select_fast(self) -> None:
        self._select_mode("turbo")

    def action_select_reasoning(self) -> None:
        self._select_mode("turbo-think")

    def action_quit(self) -> None:
        self.app.exit()
