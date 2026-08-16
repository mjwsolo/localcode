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


def test_tui_streams_reasoning_live_before_answer(tmp_path, project):
    """The model's reasoning must render live in the log (like Claude Code),
    not be hidden behind a spinner. Drives a turn whose response is preceded by
    a thinking block and asserts the reasoning text is in the visible log."""
    async def scenario():
        snap = await _drive(
            tmp_path,
            project,
            [say("The answer is 42.",
                 thinking="Consider the constraints.\nWeigh the options carefully.")],
            "think about it",
        )
        assert snap["model_calls"] >= 1
        # Reasoning streamed into the visible log, not swallowed.
        assert "Consider the constraints." in snap["log_text"]
        assert "Weigh the options carefully." in snap["log_text"]
        # The final answer still renders too.
        assert "The answer is 42." in snap["log_text"]

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
