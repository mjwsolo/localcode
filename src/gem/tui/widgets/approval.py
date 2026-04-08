"""Modal dialog for approving destructive commands."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ApprovalModal(ModalScreen[bool]):
    """Modal that asks user to approve a destructive command."""

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    #approval-box {
        width: 70;
        height: auto;
        max-height: 18;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }
    #approval-cmd {
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
    }
    .approval-buttons {
        margin-top: 1;
        align: center middle;
        height: 3;
    }
    """

    def __init__(self, tool_name: str, command: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.command = command

    def compose(self) -> ComposeResult:
        with Static(id="approval-box"):
            yield Static(f"[bold yellow]Allow this command?[/]")
            yield Static(f"[dim]{self.command[:200]}[/]", id="approval-cmd")
            with Horizontal(classes="approval-buttons"):
                yield Button("Allow", variant="success", id="approve-btn")
                yield Button("Deny", variant="error", id="deny-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve-btn")
