"""Model-picker screen render + flow tests.

THE GAP THIS CLOSES: the other TUI tests drive the *chat* screen, so nothing
ever rendered ModelPickerScreen — which let a real crash ship (a method named
`_render` collided with Textual's internal `Widget._render`, so the screen
returned a markup string where Textual expected a Visual and blew up at
render time). These tests mount the screen in Textual's headless driver and
actually render BOTH levels, so a render-time break fails CI instead of the
user's terminal.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from textual.app import App

from localcode.tui.screens.model_picker import ModelPickerScreen


class _Harness(App):
    """Minimal app that pushes the picker and captures its dismiss result."""

    def __init__(self) -> None:
        super().__init__()
        self.dismissed = "UNSET"

    async def on_mount(self) -> None:
        self.push_screen(ModelPickerScreen(), lambda r: setattr(self, "dismissed", r))


def _fake_quants():
    from localcode.hf_quants import Quant
    return [
        Quant("model-Q4_K_M.gguf", 7.1, "Q4_K_M", False),
        Quant("model-Q8_0.gguf", 28.0, "Q8_0", False),
        Quant("mmproj-F16.gguf", 0.9, "F16", True),  # sidecar — must be hidden
    ]


async def _wait_quants(scr, pilot):
    for _ in range(100):  # ~2s ceiling
        await pilot.pause(0.02)
        if scr._quants is not None:
            return


def test_picker_level1_renders_without_crashing():
    """Mount the picker and render it. This is the exact path that crashed
    the TUI on the `_render` collision — a regression here fails CI."""
    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen._render_body()
            assert "Choose a model" in body
            # specific VERSIONS are listed, not just "gemma"/"qwen"
            assert "Gemma 4 12B" in body
            assert "Qwen 3.6 35B-A3B" in body
    asyncio.run(scenario())


def test_picker_level2_shows_quants_with_fit_badges(monkeypatch):
    from localcode import hf_quants
    monkeypatch.setattr(hf_quants, "fetch_quants", lambda repo: _fake_quants())

    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            scr = app.screen
            scr._open_group_at(0)            # enter level 2 + kick the (mocked) fetch
            await _wait_quants(scr, pilot)
            body = scr._render_body()
            assert "Q4_K_M" in body and "Q8_0" in body
            # mmproj sidecar is filtered out of the user-facing quant list
            assert "mmproj" not in body.lower()
            # at least one fit badge glyph rendered
            assert any(g in body for g in ("✓", "⚠", "✗"))
            # exactly one quant ROW is starred as recommended (the footer legend
            # also contains a ★, so target rows that name a quant)
            starred_rows = [
                ln for ln in body.splitlines()
                if "★" in ln and ("Q4_K_M" in ln or "Q8_0" in ln)
            ]
            assert len(starred_rows) == 1, starred_rows
    asyncio.run(scenario())


def test_picker_selecting_a_quant_dismisses_a_modelchoice(monkeypatch):
    from localcode import hf_quants
    monkeypatch.setattr(
        hf_quants, "fetch_quants",
        lambda repo: [hf_quants.Quant("gemma-4-12b-it-Q4_K_M.gguf", 7.1, "Q4_K_M", False)],
    )

    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            scr = app.screen
            scr._open_group_at(0)
            await _wait_quants(scr, pilot)
            scr._pick_quant(0)               # Enter on the quant
            await pilot.pause()
        # dismissed with a real, downloadable ModelChoice for the picked quant
        assert app.dismissed not in (None, "UNSET")
        assert app.dismissed.filename == "gemma-4-12b-it-Q4_K_M.gguf"
        assert app.dismissed.size_gb == 7.1

    asyncio.run(scenario())


def test_pick_quant_not_downloaded_shows_confirm_before_downloading(monkeypatch):
    """Enter on a not-downloaded quant (with a usable model already present)
    must pop a ConfirmScreen and start NO download until the user confirms."""
    from localcode import hf_quants, bootstrap
    from localcode.tui.screens.confirm import ConfirmScreen
    monkeypatch.setattr(
        hf_quants, "fetch_quants",
        lambda repo: [hf_quants.Quant("gemma-4-12b-it-Q4_K_M.gguf", 7.1, "Q4_K_M", False)],
    )
    # Nothing is on disk yet, but a model IS usable, so we take the background
    # (confirm-first) branch rather than the first-run foreground dismiss.
    monkeypatch.setattr(bootstrap, "is_download_complete", lambda choice: False)
    started: list = []
    monkeypatch.setattr(
        bootstrap, "start_background_download", lambda choice: started.append(choice)
    )

    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            scr = app.screen
            monkeypatch.setattr(scr, "_has_usable_model", lambda: True)
            scr._open_group_at(0)
            await _wait_quants(scr, pilot)
            scr._pick_quant(0)               # Enter on the not-downloaded quant
            await pilot.pause(0.1)
            # A confirm modal is now on top, and NO download has started.
            assert isinstance(app.screen, ConfirmScreen)
            assert started == []
            # Cancel (Esc) leaves the picker with still no download.
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, ConfirmScreen)
            assert started == []

    asyncio.run(scenario())


def test_pick_quant_confirm_starts_the_background_download(monkeypatch):
    """Confirming the modal (y) starts exactly one background download."""
    from localcode import hf_quants, bootstrap
    monkeypatch.setattr(
        hf_quants, "fetch_quants",
        lambda repo: [hf_quants.Quant("gemma-4-12b-it-Q4_K_M.gguf", 7.1, "Q4_K_M", False)],
    )
    monkeypatch.setattr(bootstrap, "is_download_complete", lambda choice: False)
    started: list = []
    monkeypatch.setattr(
        bootstrap, "start_background_download", lambda choice: started.append(choice)
    )

    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            scr = app.screen
            monkeypatch.setattr(scr, "_has_usable_model", lambda: True)
            scr._open_group_at(0)
            await _wait_quants(scr, pilot)
            scr._pick_quant(0)
            await pilot.pause(0.1)
            await pilot.press("y")           # confirm the download
            await pilot.pause(0.1)
            assert len(started) == 1
            assert started[0].filename == "gemma-4-12b-it-Q4_K_M.gguf"

    asyncio.run(scenario())


def test_first_run_no_usable_model_downloads_without_a_modal(monkeypatch):
    """The first-run path (nothing usable yet) must still dismiss with the
    choice and pop NO confirm modal, or the app has nothing to run."""
    from localcode import hf_quants, bootstrap
    from localcode.tui.screens.confirm import ConfirmScreen
    monkeypatch.setattr(
        hf_quants, "fetch_quants",
        lambda repo: [hf_quants.Quant("gemma-4-12b-it-Q4_K_M.gguf", 7.1, "Q4_K_M", False)],
    )
    monkeypatch.setattr(bootstrap, "is_download_complete", lambda choice: False)

    async def scenario():
        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            scr = app.screen
            monkeypatch.setattr(scr, "_has_usable_model", lambda: False)
            scr._open_group_at(0)
            await _wait_quants(scr, pilot)
            scr._pick_quant(0)
            await pilot.pause(0.1)
        # Dismissed with the choice, no modal shown.
        assert app.dismissed not in (None, "UNSET")
        assert app.dismissed.filename == "gemma-4-12b-it-Q4_K_M.gguf"

    asyncio.run(scenario())
