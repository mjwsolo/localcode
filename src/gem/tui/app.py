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
        """Initialize the backend GemApp and bridge."""
        # Import here to avoid circular imports and heavy startup cost
        from ..config import load_config

        self.gem_config = load_config()

        # Create bridge
        self.bridge = TUIBridge(self)

        # Initialize GemApp headless (don't call .run())
        try:
            from ..app import GemApp
            self.gem_app = GemApp(self.gem_config)
            # Replace stdout-based output with our bridge
            self.gem_app.out.set_event_callback(self.bridge.on_event)
        except Exception as e:
            # GemApp init might fail if server isn't running
            self.notify(f"Backend init: {e}", severity="warning")

        if self.show_mode_picker:
            self.push_screen("mode_picker")
        else:
            self.push_screen("chat")

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
