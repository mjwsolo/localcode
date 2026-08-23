"""ConfirmScreen: keyboard-first destructive-action confirmation."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Static

from localcode.tui.screens.confirm import ConfirmScreen


class _Host(App):
    def compose(self) -> ComposeResult:
        yield Static("host")


def _run(keys):
    """Push a ConfirmScreen (callback form), send keys, return what it resolved to."""
    result = {}

    async def scenario():
        app = _Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.push_screen(
                ConfirmScreen("Delete model X?", "Frees 27 GB. Cannot be undone."),
                lambda v: result.__setitem__("value", v),
            )
            await pilot.pause(0.1)
            for k in keys:
                await pilot.press(k)
                await pilot.pause(0.05)

    asyncio.run(scenario())
    return result.get("value")


def test_enter_on_default_cancel_is_safe():
    # Default focus is Cancel, so a stray Enter cancels (does not delete).
    assert _run(["enter"]) is False


def test_arrow_to_confirm_then_enter_confirms():
    assert _run(["right", "enter"]) is True


def test_y_confirms_regardless_of_focus():
    assert _run(["y"]) is True


def test_escape_cancels():
    assert _run(["escape"]) is False


def test_n_cancels():
    assert _run(["n"]) is False


def test_default_focus_is_cancel_so_a_stray_key_is_safe():
    focused = {}

    async def scenario():
        app = _Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app.push_screen(ConfirmScreen("X?", "y"), lambda v: None)
            await pilot.pause(0.1)
            focused["id"] = app.focused.id if app.focused else None
            await pilot.press("escape")

    asyncio.run(scenario())
    assert focused["id"] == "confirm-cancel"
