"""Live TUI driver tests — drive the REAL Textual app with simulated
keystrokes via Textual's headless test driver.

`App.run_test()` runs the app on an in-memory (headless) driver — no real
terminal, fully awaitable — so unlike `--preview-screen` in a subprocess
(which attaches a real TTY and never exits), this does NOT hang. We land
directly on the chat screen (the app's `_preview_screen` short-circuit),
swap in a scripted FakeRuntime backend, type a prompt, press Enter, and
assert the response renders in the chat log.

Each test wraps an async scenario in `asyncio.run(...)` so it needs no
pytest-asyncio/anyio plugin configuration.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tests.e2e.fake_runtime import build_test_app, say, tool_round


def _chat_log_text(app) -> str:
    """Flatten the visible chat log to plain text for assertions."""
    log = app.screen.query_one("#chat-log")
    return "\n".join(s.text for s in log.lines)


async def _drive(tmp_path, project, script, keystrokes_text, configure=None):
    """Boot the TUI on the chat screen with a fake backend, type a line,
    press Enter, wait for the agent worker to finish, and return a plain
    snapshot of what happened.

    All UI reads happen INSIDE the `run_test()` context — once it exits
    the screen stack is torn down, so we must not hand the live app back
    to the caller. Returns a dict: {log_text, input_value, model_calls}.
    """
    from localcode.tui.app import LocalCodeTUI

    os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
    app = LocalCodeTUI()
    app._preview_screen = "chat"  # skip health/server/model-picker → chat

    async with app.run_test() as pilot:
        await pilot.pause()  # let on_mount push the chat screen
        # Replace the (uninitialised) backend with a scripted one bound to
        # a throwaway repo, and rewire its output to the live TUI bridge.
        backend = build_test_app(tmp_path, script=script, cwd=project)
        app.engine = backend
        backend.out.set_event_callback(app.bridge.on_event)
        backend.out.set_approval_callback(app.bridge.request_approval)
        if configure is not None:
            configure(backend)

        chat_input = app.screen.query_one("#chat-input")
        chat_input.value = keystrokes_text
        await pilot.press("enter")

        # Agent runs on a background thread worker; pump the event loop
        # until it finishes (or time out so a hang can't wedge the suite).
        for _ in range(200):  # ~10s ceiling
            await pilot.pause(0.05)
            if not getattr(app.screen, "_agent_busy", False):
                break

        return {
            "log_text": _chat_log_text(app),
            "log_lines": len(app.screen.query_one("#chat-log").lines),
            "input_value": app.screen.query_one("#chat-input").value,
            "model_calls": len(backend.engine.calls),
            "think_calls": list(backend.engine.think_calls),
            "calls": backend.engine.calls,
        }


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n")
    return repo


def test_tui_prompt_renders_model_response(tmp_path, project):
    async def scenario():
        snap = await _drive(
            tmp_path, project, [say("Hello from the model.")], "hi there"
        )
        # The keystroke went through the real Input → submit → worker → loop.
        assert snap["model_calls"] >= 1, "model was never called"
        # The user's line and the model's reply both rendered in the log.
        assert "hi there" in snap["log_text"]
        assert "Hello from the model." in snap["log_text"]
        # Input box cleared after submit.
        assert snap["input_value"] == ""

    asyncio.run(scenario())


def test_tui_collapses_reasoning_by_default(tmp_path, project):
    """Raw chain-of-thought must NOT flood the chat (user report 2026-08-22:
    Muse Glimmer painted paragraphs of reasoning live, in dim italic, including
    the system prompt recited back). Collapsed is the default: the turn ends
    with ONE expandable "▶ …" row holding the reasoning, the compact indicator
    covers the live phase, and the final answer still renders normally."""
    async def scenario():
        # Several hundred chars over many sentences/lines — the shape that
        # used to become a wall of dim text, one rendered row per line.
        reasoning = "\n".join(
            f"Reasoning sentence number {i} about following the system "
            f"instructions and planning the answer carefully." for i in range(12)
        )
        assert len(reasoning) > 800
        snap = await _drive(
            tmp_path,
            project,
            [say("The answer is 42.", thinking=reasoning)],
            "think about it",
        )
        assert snap["model_calls"] >= 1
        # The reasoning body is NOT dumped into the visible log. (The first
        # line may appear truncated in the collapsed header preview.)
        assert "Reasoning sentence number 5" not in snap["log_text"]
        assert "Reasoning sentence number 11" not in snap["log_text"]
        # Exactly one collapsed, expandable thinking row landed instead.
        assert snap["log_text"].count("▶") == 1
        assert "(+11 lines)" in snap["log_text"]
        # The final answer still renders normally.
        assert "The answer is 42." in snap["log_text"]
        # And the whole log stays SHORT: user line + collapsed row + answer
        # + chrome, nowhere near the 12+ rows the reasoning would occupy.
        assert snap["log_lines"] < 15, snap["log_text"]

    asyncio.run(scenario())


def test_tui_runaway_thinking_loop_recovers_with_no_think_retry(tmp_path, project):
    """A degenerate REPETITION loop must be caught by the periodicity detector
    and trigger a REAL decode-mode retry — not just a printed message. Proves the
    recovery actually runs (regression guard for the dead-code path where
    detect_stall returns None on a thinking abort and the turn just ends):

      1. exactly two model requests occurred
      2. the second request decoded with think=False
      3. the second response produced the final answer
      4. the aborted empty round added no message to history
    """
    async def scenario():
        runaway = "planning " * 12000  # exactly periodic, no answer

        def force_thinking_on(backend):
            backend.config.runtime.internal_thinking_mode = "on"

        snap = await _drive(
            tmp_path, project,
            [say("(loops)", thinking=runaway), say("Fixed it.")],
            "build the thing",
            configure=force_thinking_on,
        )
        # 1. Exactly two requests: the aborted one + the no-think retry.
        assert snap["model_calls"] == 2, snap["think_calls"]
        # 2. First decoded with thinking on; the retry forced it off.
        assert snap["think_calls"] == [True, False]
        # 3. The retry's final answer rendered.
        assert "Fixed it." in snap["log_text"]
        assert "repeating itself" in snap["log_text"].lower()
        # 4. The aborted round appended nothing — the retry sees the same
        #    message list the first call saw (no empty assistant message).
        assert len(snap["calls"][1]) == len(snap["calls"][0])
        assert not any(
            m.get("role") == "assistant" and not (m.get("content") or "").strip()
            for m in snap["calls"][1]
        )

    asyncio.run(scenario())


def test_tui_nonperiodic_runaway_recovers_without_thinking(tmp_path, project):
    """Non-repeating reasoning that simply runs too long trips the char/time cap.
    It no longer hard-fails the turn: like a detected loop, it now RECOVERS by
    re-running the step with thinking off (up to the recovery budget), then ends
    honestly if the model keeps over-reasoning. Guards that a slow model's cap
    trip is salvaged instead of throwing the whole turn away."""
    async def scenario():
        # Strictly increasing tokens: long but NOT periodic, so only the length
        # cap should stop it. The mock re-emits it every round (ignores think),
        # so recovery is exhausted and the turn ends with the honest message.
        runaway = " ".join(str(i) for i in range(30000))  # ~150k varied chars
        snap = await _drive(
            tmp_path, project,
            [say("(never reached)", thinking=runaway)],
            "build the thing",
        )
        assert snap["model_calls"] >= 1
        # Collapse wrapping/indent so the multi-line notice matches as one string.
        low = " ".join(snap["log_text"].lower().split())
        # New behavior: the cap trip triggers a no-think RETRY (recovery), never
        # the old "reasoning exceeded" hard-stop that threw the turn away.
        assert "without deep reasoning" in low
        assert "reasoning exceeded" not in low

    asyncio.run(scenario())


def test_tui_tool_call_turn_executes_through_ui(tmp_path, project):
    """A scripted tool round driven entirely from a keystroke: the model
    'calls' write_file, the real tool runs, the file appears on disk, and
    the final answer renders."""
    async def scenario():
        script = [
            tool_round(("write_file", {"path": "made_by_tui.py", "content": "X = 1\n"})),
            say("Created the file."),
        ]
        snap = await _drive(tmp_path, project, script, "create a file")
        assert (project / "made_by_tui.py").read_text() == "X = 1\n"
        assert "Created the file." in snap["log_text"]

    asyncio.run(scenario())


def test_tui_slash_clear_command(tmp_path, project):
    """A slash command typed at the prompt is handled (not sent to model)
    and doesn't crash the app."""
    async def scenario():
        from localcode.tui.app import LocalCodeTUI

        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test() as pilot:
            await pilot.pause()  # let on_mount push the chat screen
            backend = build_test_app(tmp_path, script=[say("hi")], cwd=project)
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)

            chat_input = app.screen.query_one("#chat-input")
            chat_input.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.1)

            # Slash command must NOT have been routed to the model.
            assert backend.engine.calls == []
            # App is still alive and input is clear.
            assert app.screen.query_one("#chat-input").value == ""

    asyncio.run(scenario())


def test_tui_slash_clear_confirms_when_there_is_history(tmp_path, project):
    """/clear on a non-empty conversation must ask before wiping it.

    Wiping the whole conversation can't be undone, so it routes through the
    ConfirmScreen (Cancel keeps the history; Clear wipes it). An empty
    conversation skips the dialog - that path is covered by the test above.
    """
    async def scenario():
        from localcode.tui.app import LocalCodeTUI
        from localcode.tui.screens.confirm import ConfirmScreen

        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test() as pilot:
            await pilot.pause()
            backend = build_test_app(tmp_path, script=[say("hi")], cwd=project)
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)
            # Seed a conversation so /clear has something to lose.
            backend.session.messages.extend(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            )
            chat_screen = app.screen

            # /clear must open the confirm dialog, NOT wipe immediately.
            chat_input = chat_screen.query_one("#chat-input")
            chat_input.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, ConfirmScreen), "no confirm dialog shown"
            assert len(backend.session.messages) == 2, "history wiped before confirming"

            # Cancel keeps the history.
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert not isinstance(app.screen, ConfirmScreen)
            assert len(backend.session.messages) == 2, "cancel should keep history"

            # Now confirm: /clear -> dialog -> y wipes it.
            chat_input = app.screen.query_one("#chat-input")
            chat_input.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause(0.1)
            assert len(backend.session.messages) == 0, "confirm should wipe history"

    asyncio.run(scenario())


def test_tui_slash_menu_windows_selection_and_drops_removed_commands(tmp_path, project):
    """The slash palette must keep the highlighted command VISIBLE.

    #slash-menu is capped at `max-height: 10`; before the windowing fix,
    a bare "/" rendered every command, so rows past the cap were invisible
    and Down-arrow appeared frozen once the highlight left the window.
    Also guards that /paste, /image and /hooks are gone (removed 2026-08-22)."""
    async def scenario():
        from localcode.tui.app import LocalCodeTUI
        from textual.widgets import Static

        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test() as pilot:
            await pilot.pause()
            backend = build_test_app(tmp_path, script=[say("hi")], cwd=project)
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)

            await pilot.press("slash")
            await pilot.pause()
            screen = app.screen
            names = [c for c, _ in screen._slash_matches]
            # Removed commands are gone from the palette and the alias set.
            from localcode.tui.screens.chat import _is_known_command
            for gone in ("/paste", "/hooks", "/image"):
                assert gone not in names
                assert not _is_known_command(gone)
            n = len(names)
            cap = screen._SLASH_MENU_ROWS
            assert n > cap, "windowing test needs more commands than the cap"

            menu = screen.query_one("#slash-menu", Static)
            # Initial window starts at the top and advertises the overflow.
            assert screen._slash_window == 0
            assert f"{n - cap} more ↓" in str(menu.content)

            # Walk PAST the window (and wrap around): the highlighted
            # command must stay inside the rendered slice at every step.
            for _ in range(n + 3):
                await pilot.press("down")
                await pilot.pause()
                sel = screen._slash_selected
                start = screen._slash_window
                assert start <= sel < start + cap
                assert names[sel] in str(menu.content)
            # After walking to the bottom and wrapping, the "↑ N more"
            # affordance appeared at least conceptually — re-check bottom:
            for _ in range(n - 1):
                await pilot.press("down")
                await pilot.pause()
                if screen._slash_selected == n - 1:
                    break
            assert "↑" in str(menu.content)
            await pilot.press("escape")

    asyncio.run(scenario())


def test_startup_always_shows_model_picker(monkeypatch):
    """on_mount must ALWAYS land on the model picker — never silently auto-load
    a configured model.

    Regression: when ``config.runtime.model`` pointed at a downloaded, complete
    model, ``need_picker`` was False so the app skipped the picker and instantly
    started the server. Users expect to choose (or confirm) a model on launch.
    """
    import localcode.health as health

    class _OK:
        ok = True
        stuck_servers = []  # type: ignore[var-annotated]
        message = ""

    monkeypatch.setattr(health, "check_system_health", lambda *a, **k: _OK())
    from localcode.tui.app import LocalCodeTUI

    async def scenario():
        app = LocalCodeTUI(show_mode_picker=False)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            landed = type(app.screen).__name__
            app.exit()
            return landed

    assert asyncio.run(scenario()) == "ModelPickerScreen"


# ─── Scroll / command-visibility / clipboard-image regression tests ────────
#
# Mouse capture is OFF by default (native terminal text selection), so the
# keyboard is the ONLY way to scroll the chat log. These pilots guard:
#   1. PageUp/PageDown/Home/End/Shift+Up/Down scroll the ChatLog even while
#      the chat TextArea has focus.
#   2. A slash command always scrolls its output into view (previously the
#      output landed below the fold when the user had scrolled up).
#   3. Ctrl+G and Ctrl+V attach a clipboard image end to end.
#   4. /status no longer leaks the internal perf-profile name.


async def _boot_chat_app():
    """Start the real TUI headless on the chat screen (no backend needed)."""
    from localcode.tui.app import LocalCodeTUI

    os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
    app = LocalCodeTUI()
    app._preview_screen = "chat"
    return app


def _fill_log(app, n: int = 120) -> None:
    log = app.screen.query_one("#chat-log")
    for i in range(n):
        log.append_info(f"filler line {i}")


def test_tui_rapid_identical_command_resubmit_both_dispatch():
    """Submitting the SAME slash command twice in quick succession must run
    it BOTH times. A post-submit `_double_clear` timer used to wipe the
    input whenever it still held the just-submitted text — which also
    matched a command the user re-typed within ~1.5s, swallowing the second
    Enter. Guard: toggle /permissions three times fast and assert the
    autonomy level flips each time (ON→OFF→ON)."""
    async def scenario():
        from localcode.tui.app import LocalCodeTUI
        from localcode.autonomy import AutonomyLevel
        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        import tempfile
        tp = Path(tempfile.mkdtemp())
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            backend = build_test_app(tp, script=[say("x")], cwd=tp / "proj")
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)
            scr = app.screen
            seen = []
            for _ in range(3):
                inp = scr.query_one("#chat-input")
                inp.focus()
                await pilot.pause()
                for ch in "/permissions":
                    await pilot.press("slash" if ch == "/" else ch)
                await pilot.pause(0.15)
                await pilot.press("enter")
                await pilot.pause(0.25)  # WELL inside the 1.5s clear window
                seen.append(backend._autonomy)
            # Each submit must have toggled: FULL_AUTO→AUTO_EDIT→FULL_AUTO→AUTO_EDIT
            assert seen == [
                AutonomyLevel.AUTO_EDIT,
                AutonomyLevel.FULL_AUTO,
                AutonomyLevel.AUTO_EDIT,
            ], seen

    asyncio.run(scenario())


def test_tui_keyboard_scrolls_chat_log_while_input_focused():
    """Ctrl+U/Ctrl+D scroll the log half a page, Ctrl+Home/Ctrl+End jump to
    top / bottom-and-re-follow, Shift+Up/Down move a line — all while the
    chat input (a TextArea that owns plain Up/Down) has focus.

    PageUp/PageDown are DELIBERATELY not chat-scroll keys: the multi-line
    composer needs them for cursor navigation (see the companion test
    below). This mirrors less/vim and the reference agents (Codex, Claude
    Code), which avoid PageUp/PageDown for scroll to dodge the composer
    collision."""
    async def scenario():
        app = await _boot_chat_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            _fill_log(app)
            log = app.screen.query_one("#chat-log")
            log.scroll_end(animate=False)
            await pilot.pause()
            bottom = log.scroll_offset.y
            assert bottom > 0, "log did not overflow — test setup broken"

            await pilot.press("ctrl+u")
            await pilot.pause()
            assert log.scroll_offset.y < bottom, "Ctrl+U did not scroll up"

            await pilot.press("ctrl+end")
            await pilot.pause()
            assert log.scroll_offset.y == bottom, "Ctrl+End did not return to bottom"

            await pilot.press("ctrl+home")
            await pilot.pause()
            assert log.scroll_offset.y == 0, "Ctrl+Home did not jump to the top"

            await pilot.press("ctrl+d")
            await pilot.pause()
            assert log.scroll_offset.y > 0, "Ctrl+D did not scroll down"

            await pilot.press("ctrl+home")
            await pilot.pause()
            await pilot.press("shift+down")
            await pilot.pause()
            assert log.scroll_offset.y == 1, "Shift+Down did not scroll one line"

            await pilot.press("shift+up")
            await pilot.pause()
            assert log.scroll_offset.y == 0, "Shift+Up did not scroll back"

            await pilot.press("ctrl+end")
            await pilot.pause()
            assert log.scroll_offset.y == bottom, "Ctrl+End did not jump to the bottom"

    asyncio.run(scenario())


def test_tui_pageup_moves_composer_cursor_not_scroll():
    """PageUp/PageDown must move the CURSOR inside a multi-line draft, not
    scroll the chat log. A prior binding routed them to chat scroll, which
    silently stole cursor navigation from the composer (the cursor never
    moved). Regression guard for that fix."""
    async def scenario():
        app = await _boot_chat_app()
        async with app.run_test(size=(120, 16)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.focus()
            await pilot.pause()
            # Build a 5-line draft with Shift+Enter newlines.
            for ln in range(5):
                for ch in f"line{ln}":
                    await pilot.press(ch)
                if ln < 4:
                    await pilot.press("shift+enter")
            await pilot.pause()
            assert inp.cursor_location[0] == 4, "cursor not on last draft line"
            await pilot.press("pageup")
            await pilot.pause()
            assert inp.cursor_location[0] < 4, (
                "PageUp did not move the composer cursor up — scroll binding "
                "stole it again"
            )

    asyncio.run(scenario())


def test_tui_ctrl_o_toggles_reasoning_from_keyboard():
    """Mouse capture is OFF by default, so the ▶/▼ thinking toggle (a click
    handler) is unreachable for keyboard-only users. Ctrl+O must expand a
    collapsed reasoning block (and collapse it again) with the composer
    focused."""
    async def scenario():
        reasoning = "\n".join(
            f"reason line {i} planning the answer carefully" for i in range(10)
        )
        snap_app = {}

        from localcode.tui.app import LocalCodeTUI
        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        import tempfile
        tp = Path(tempfile.mkdtemp())
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            backend = build_test_app(
                tp, script=[say("The answer is 42.", thinking=reasoning)],
                cwd=tp / "proj",
            )
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)
            backend.out.set_approval_callback(app.bridge.request_approval)
            scr = app.screen
            log = scr.query_one("#chat-log")
            inp = scr.query_one("#chat-input")
            inp.focus()
            await pilot.pause()
            for ch in "think":
                await pilot.press(ch)
            await pilot.press("enter")
            for _ in range(80):
                await pilot.pause(0.05)
                if not getattr(scr, "_agent_busy", False):
                    break

            def visible():
                return "reason line 5" in "\n".join(s.text for s in log.lines)

            assert not visible(), "reasoning should start collapsed"
            inp.focus()
            await pilot.pause()
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert visible(), "Ctrl+O did not expand the reasoning block"
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert not visible(), "Ctrl+O did not collapse the reasoning block again"

    asyncio.run(scenario())


def test_tui_command_output_scrolls_into_view():
    """Reproduces the 'command output is invisible until the next input' bug:
    scroll to the top, run /status, and assert the log snapped to the bottom
    so the RUNTIME block is inside the viewport."""
    async def scenario():
        app = await _boot_chat_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            _fill_log(app)
            # Let pre-layout deferred writes replay (they auto-scroll to the
            # bottom) BEFORE we scroll up, or the replay undoes scroll_home.
            await pilot.pause()
            log = app.screen.query_one("#chat-log")
            log.scroll_home(animate=False)
            await pilot.pause()
            assert log.scroll_offset.y == 0

            chat_input = app.screen.query_one("#chat-input")
            chat_input.value = "/status"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()

            text = _chat_log_text(app)
            assert "RUNTIME" in text, "/status produced no output"
            # The log must now be at (or within the follow-tail band of)
            # the bottom, i.e. the freshly written output is on screen.
            assert log.scroll_offset.y >= log.max_scroll_y - 2, (
                f"command output below the fold: offset={log.scroll_offset.y} "
                f"max={log.max_scroll_y}"
            )
            # Fix 4: the internal perf-profile identifier must not be shown.
            assert "profile" not in text.lower().split("performance")[0].split("runtime")[-1], \
                "/status still shows the internal perf-profile name"

    asyncio.run(scenario())


def test_tui_ctrl_g_and_ctrl_v_attach_clipboard_image(monkeypatch):
    """Ctrl+G (app binding) and Ctrl+V (TextArea key hook) both read the OS
    clipboard and attach the image for the next message."""
    fake_png = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"

    async def scenario():
        import localcode.tui.clipboard_image as clip
        monkeypatch.setattr(clip, "read_clipboard_png", lambda: fake_png)

        app = await _boot_chat_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen

            await pilot.press("ctrl+g")
            await pilot.pause()
            assert len(getattr(screen, "_pending_images", [])) == 1, \
                "Ctrl+G did not attach the clipboard image"

            await pilot.press("ctrl+v")
            await pilot.pause()
            assert len(screen._pending_images) == 2, \
                "Ctrl+V did not attach the clipboard image"

            text = _chat_log_text(app)
            assert "image attached" in text

    asyncio.run(scenario())


def test_tui_ctrl_v_with_text_clipboard_pastes_text(monkeypatch):
    """Ctrl+V with NO image on the clipboard falls back to pasting the
    clipboard's text into the composer instead of doing nothing."""
    async def scenario():
        import subprocess as sp
        import localcode.tui.clipboard_image as clip
        monkeypatch.setattr(clip, "read_clipboard_png", lambda: None)

        real_run = sp.run

        def fake_run(cmd, *a, **kw):
            if cmd and cmd[0] == "pbpaste":
                class R:
                    stdout = "pasted-from-clipboard"
                return R()
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(sp, "run", fake_run)

        app = await _boot_chat_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+v")
            await pilot.pause()
            if sys.platform == "darwin":
                chat_input = app.screen.query_one("#chat-input")
                assert "pasted-from-clipboard" in chat_input.text, \
                    "Ctrl+V text fallback did not paste clipboard text"

    asyncio.run(scenario())


def test_tui_attached_clipboard_image_reaches_model_payload(tmp_path, project, monkeypatch):
    """END TO END: Ctrl+G attaches a clipboard image and the NEXT model call's
    user message carries it as an OpenAI-style image_url part — proving the
    image reaches the runtime payload, not just a log line."""
    import base64

    fake_png = b"\x89PNG\r\n\x1a\n" + b"payload-proof"
    fake_b64 = base64.b64encode(fake_png).decode("ascii")

    async def scenario():
        import localcode.tui.clipboard_image as clip
        monkeypatch.setattr(clip, "read_clipboard_png", lambda: fake_png)

        from localcode.tui.app import LocalCodeTUI

        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        app = LocalCodeTUI()
        app._preview_screen = "chat"

        async with app.run_test() as pilot:
            await pilot.pause()
            backend = build_test_app(
                tmp_path, script=[say("I can see it.")], cwd=project
            )
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)
            backend.out.set_approval_callback(app.bridge.request_approval)

            # Attach via the guaranteed key.
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert len(app.screen._pending_images) == 1

            chat_input = app.screen.query_one("#chat-input")
            chat_input.value = "what is in this image?"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause(0.05)
                if not getattr(app.screen, "_agent_busy", False):
                    break

            calls = backend.engine.calls
            assert calls, "model was never called"
            image_parts = [
                part
                for msg in calls[0]
                if msg.get("role") == "user" and isinstance(msg.get("content"), list)
                for part in msg["content"]
                if part.get("type") == "image_url"
            ]
            assert image_parts, (
                "attached image never reached the model payload: "
                f"{[m.get('role') for m in calls[0]]}"
            )
            assert fake_b64 in image_parts[0]["image_url"]["url"]
            # Attach queue cleared — the image sends exactly once.
            assert app.screen._pending_images == []

    asyncio.run(scenario())
