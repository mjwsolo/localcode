"""Mode picker screen — Fast vs Reasoning."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, RadioButton, RadioSet, Rule, Static


class ModePickerScreen(Screen):
    """Initial screen for selecting Fast or Reasoning mode."""

    DEFAULT_CSS = """
    ModePickerScreen {
        align: center middle;
    }
    #picker-box {
        width: 50;
        height: auto;
        padding: 1 2;
    }
    #picker-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #picker-radio {
        margin: 1 2;
    }
    #start-btn {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="picker-box"):
                yield Static("🏠 [bold]localcode[/]", id="picker-title")
                yield Rule()
                yield Static("Select a mode:")
                yield RadioSet(
                    RadioButton("Fast - quicker answers for routine work", id="fast", value=True),
                    RadioButton("Reasoning - deeper thinking for harder tasks", id="reasoning"),
                    id="picker-radio",
                )
                yield Button("Start", variant="primary", id="start-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            radio = self.query_one(RadioSet)
            idx = radio.pressed_index
            if idx == 1:  # Reasoning
                self.app.gem_config.runtime.laptop_26b_runtime_mode = "turbo-think"
            else:  # Fast (default)
                self.app.gem_config.runtime.laptop_26b_runtime_mode = "turbo"
            self.app.switch_screen("chat")
