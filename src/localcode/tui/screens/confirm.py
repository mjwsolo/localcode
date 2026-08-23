"""A small reusable confirmation modal for destructive actions.

Before this, every destructive flow (delete a model, clear the conversation)
grew its own inline y/n state machine inside a command handler. That is exactly
the shape the Codex/OpenCode/Claude-Code comparison flagged: they route every
destructive action through ONE audited confirm dialog. This is localcode's.

Usage (from any screen):

    ok = await self.app.push_screen_wait(
        ConfirmScreen("Delete Gemma 4 26B (Q8)?", "Frees 27.6 GB. Cannot be undone.")
    )
    if ok:
        ...

Enter / y confirm, Esc / n cancel. Works with the keyboard alone (mouse is off
by default), and the buttons are also clickable when mouse is on.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmScreen(ModalScreen[bool]):
    """Modal that resolves to True (confirm) or False (cancel)."""

    # Enter activates the FOCUSED button (Cancel by default, so a stray Enter is
    # safe). `y` always confirms, `n`/Esc always cancel, arrows/Tab switch focus.
    BINDINGS = [
        Binding("y", "confirm", "Confirm", show=False),
        Binding("escape,n", "cancel", "Cancel", show=False),
        Binding("left,right,tab", "toggle", "Switch", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    ConfirmScreen > #confirm-box {
        width: 60;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    ConfirmScreen #confirm-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ConfirmScreen #confirm-detail {
        color: $text-muted;
        padding-bottom: 1;
    }
    ConfirmScreen #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    ConfirmScreen Button {
        margin-left: 2;
        min-width: 12;
    }
    ConfirmScreen #confirm-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    def __init__(
        self,
        title: str,
        detail: str = "",
        *,
        confirm_label: str = "Delete",
        cancel_label: str = "Cancel",
        dangerous: bool = True,
    ) -> None:
        super().__init__()
        self._title = title
        self._detail = detail
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._dangerous = dangerous

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self._title, id="confirm-title")
            if self._detail:
                yield Static(self._detail, id="confirm-detail")
            with Horizontal(id="confirm-buttons"):
                # Cancel is the default focus so a stray Enter does not delete.
                yield Button(self._cancel_label, id="confirm-cancel")
                yield Button(
                    self._confirm_label,
                    id="confirm-ok",
                    variant="error" if self._dangerous else "primary",
                )
            yield Static(
                "y confirm  ·  n / esc cancel  ·  \u2190\u2192 switch", id="confirm-hint"
            )

    def on_mount(self) -> None:
        # Land on Cancel: the safe choice must be the default.
        self.query_one("#confirm-cancel", Button).focus()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_toggle(self) -> None:
        cancel = self.query_one("#confirm-cancel", Button)
        ok = self.query_one("#confirm-ok", Button)
        (ok if cancel.has_focus else cancel).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-ok")
