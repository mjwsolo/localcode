"""LocalCode Textual TUI — main application."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from textual.app import App

from .bridge import AgentEvent, ApprovalRequest, TUIBridge
from .screens.chat import ChatScreen
from .screens.mode_picker import ModePickerScreen


class LocalCodeTUI(App):
    """Textual-based terminal UI for LocalCode."""

    CSS_PATH = "styles/app.tcss"
    TITLE = "localcode"

    SCREENS = {
        "chat": ChatScreen,
        "mode_picker": ModePickerScreen,
    }

    def __init__(self, show_mode_picker: bool = True) -> None:
        super().__init__()
        self.show_mode_picker = show_mode_picker
        self.gem_app = None
        self.gem_config = None
        self.bridge = None

    def on_mount(self) -> None:
        """Initialize config and show first screen. GemApp loaded lazily."""
        from ..config import load_config

        self.gem_config = load_config()
        self.bridge = TUIBridge(self)

        # Don't init GemApp here — it's heavy and may crash.
        # ChatScreen will init it on first message.

        if self.show_mode_picker:
            self.push_screen("mode_picker")
        else:
            self.push_screen("chat")

    def ensure_backend(self) -> bool:
        """Lazily initialize GemApp backend. Returns True if ready."""
        if self.gem_app is not None:
            return True
        try:
            from ..app import GemApp
            self.gem_app = GemApp(self.gem_config)
            self.gem_app.out.set_event_callback(self.bridge.on_event)
            return True
        except Exception as e:
            self.notify(f"Backend error: {e}", severity="error")
            return False

    # Route bridge messages to the active screen
    def on_agent_event(self, event: AgentEvent) -> None:
        screen = self.screen
        if hasattr(screen, "on_agent_event"):
            screen.on_agent_event(event)

    def on_approval_request(self, event: ApprovalRequest) -> None:
        screen = self.screen
        if hasattr(screen, "on_approval_request"):
            screen.on_approval_request(event)


def main() -> None:
    """Entry point for lc-tui or localcode --tui."""
    app = LocalCodeTUI(show_mode_picker=True)
    app.run()


if __name__ == "__main__":
    main()
