from localcode.tui.screens.chat import (
    _NoTintInput,
    _spinner_label,
    _SPINNER_GERUNDS,
    ChatScreen,
    reconcile_live_tokens,
)
from localcode.tui.widgets.chat_log import ChatLog


def test_live_token_counter_snaps_to_real_cumulative_usage() -> None:
    # Round 1 streams ~200 tokens (char/4 estimate), then stream_done reports
    # real completion tokens. Mid-stream the estimate leads; once real usage
    # arrives the badge must reflect the real cumulative, not the estimate.
    live = 200  # streamed char/4 estimate for round 1
    live = reconcile_live_tokens(live, 900)  # round 1 real completion = 900
    assert live == 900
    # Round 2 decodes another ~150 estimated tokens on TOP of the reconciled
    # value (the handler keeps adding char-deltas mid-stream).
    live += 150
    assert live == 1050
    # Round 2 closes; cumulative real completion is 900 + 600 = 1500.
    live = reconcile_live_tokens(live, 1500)
    assert live == 1500


def test_live_token_counter_never_regresses() -> None:
    # A late/duplicate usage report with a smaller cumulative must not pull
    # the badge backwards.
    assert reconcile_live_tokens(1500, 1200) == 1500
    # Zero / missing real usage leaves the live estimate intact.
    assert reconcile_live_tokens(340, 0) == 340


def test_spinner_label_is_always_a_generic_placeholder() -> None:
    # Every rotation value must yield one of the allowed playful gerunds —
    # never the model's real thinking text.
    for tick in range(len(_SPINNER_GERUNDS) * 3):
        assert _spinner_label(tick) in _SPINNER_GERUNDS


def test_spinner_label_never_echoes_model_thinking() -> None:
    # The helper takes no model text and must not surface any. Simulate a
    # leaked reasoning string and confirm it can't appear in the label.
    leaked = "the user wants me to compute the eigenvalues first"
    assert leaked not in _spinner_label(0)
    assert all(leaked not in g for g in _SPINNER_GERUNDS)


class _Selection:
    is_empty = True


class _PasteEvent:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prevented = False
        self.stopped = False

    def prevent_default(self) -> None:
        self.prevented = True

    def stop(self) -> None:
        self.stopped = True


class _InputLike:
    selection = _Selection()

    def __init__(self) -> None:
        self.inserted = ""

    def insert_text_at_cursor(self, text: str) -> None:
        self.inserted += text


def test_chat_input_paste_prevents_textual_default_insert() -> None:
    event = _PasteEvent("hello")
    target = _InputLike()

    _NoTintInput._on_paste(target, event)

    assert event.prevented
    assert event.stopped
    assert target.inserted == "hello"


def test_chat_input_paste_preserves_multiline_newlines() -> None:
    # Paste now KEEPS newlines (normalizing CRLF/CR → LF) so pasted code
    # / JSON retains its structure on submit. The screen's #input-overflow
    # preview renders the multi-line value; the old behaviour flattened
    # everything to spaces and destroyed any pasted block.
    event = _PasteEvent("line one\nline two\r\nline three")
    target = _InputLike()

    _NoTintInput._on_paste(target, event)

    assert target.inserted == "line one\nline two\nline three"


class _HistInputLike:
    """Minimal stub for _hist_navigate without spinning up Textual."""

    def __init__(self, value: str = "") -> None:
        self.value = value
        self.cursor_position = len(value)
        self._input_history: list[str] = []
        self._input_history_pos: int = -1
        self._input_history_draft: str = ""

    def _hist_init(self) -> None:
        pass


def test_down_arrow_clears_draft_when_not_browsing_history() -> None:
    target = _HistInputLike(value="some half-typed prompt")
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is True
    assert target.value == ""
    assert target.cursor_position == 0
    assert target._input_history_pos == -1


def test_down_arrow_on_empty_input_lets_key_bubble() -> None:
    target = _HistInputLike(value="")
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is False
    assert target.value == ""


def test_down_arrow_while_browsing_history_walks_forward_not_clear() -> None:
    target = _HistInputLike(value="older entry")
    target._input_history = ["older entry", "newer entry"]
    target._input_history_pos = 0
    changed = _NoTintInput._hist_navigate(target, +1)
    assert changed is True
    assert target.value == "newer entry"
    assert target._input_history_pos == 1


def _thinking_log() -> ChatLog:
    log = ChatLog()
    log._history = [("thinking", "first line\nsecond line")]
    log._thinking_states = {0: False}
    log._thinking_line_map = {3: 0}
    log._rerender = lambda: None  # type: ignore[method-assign]
    return log


def test_thinking_disclosure_click_toggles_matching_row() -> None:
    log = _thinking_log()

    handled = log._handle_thinking_click(3, x=3)

    assert handled is True
    assert log._thinking_states[0] is True


def test_thinking_disclosure_duplicate_click_only_toggles_once() -> None:
    log = _thinking_log()

    assert log._handle_thinking_click(3, x=3) is True
    assert log._handle_thinking_click(3, x=3) is True

    assert log._thinking_states[0] is True


def test_thinking_gutter_click_far_from_header_does_not_toggle() -> None:
    log = _thinking_log()

    handled = log._handle_thinking_click(40, x=3)

    assert handled is False
    assert log._thinking_states[0] is False


# --- /model swap auto-restart (regression for "Backend not initialized") ---

class _FakeLog:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []

    def append_info(self, msg: str) -> None:
        self.infos.append(msg)

    def append_error(self, msg: str) -> None:
        self.errors.append(msg)


class _FakeInnerEngine:
    """Stands in for LocalCodeRuntimeGateway: owns _restart_server."""

    def __init__(self) -> None:
        self.restart_calls = 0

        class _Cfg:
            model = ""

        self.config = _Cfg()

    def _restart_server(self) -> bool:
        self.restart_calls += 1
        return True


class _FakeBackend:
    """Stands in for LocalCodeApp: holds the inner gateway as `.engine`."""

    def __init__(self) -> None:
        self.engine = _FakeInnerEngine()

        class _RT:
            model = ""

        class _Cfg:
            runtime = _RT()

        self.config = _Cfg()
        self.runtime_model = ""

        class _Sess:
            model = ""

        self.session = _Sess()


class _FakeTui:
    """Stands in for LocalCodeTUI app. ensure_backend lazy-inits engine."""

    def __init__(self, config) -> None:
        self.config = config
        self.engine = None  # not initialized — the bug's precondition
        self.ensure_backend_calls = 0

    def ensure_backend(self) -> bool:
        self.ensure_backend_calls += 1
        if self.engine is None:
            self.engine = _FakeBackend()
        return True


class _FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


class _SwapScreenStub:
    """Minimal stand-in so we can drive ChatScreen._apply_model_choice
    without spinning up Textual. run_worker executes the worker inline."""

    def __init__(self, tui) -> None:
        self.tui = tui
        self.app = _FakeApp()
        self._log = _FakeLog()
        self._server_restarting = False
        self._pending_messages: list = []

    def query_one(self, *_a, **_k):
        return self._log

    def _update_status(self) -> None:
        pass

    def _update_queue(self) -> None:
        pass

    def _set_download_line(self, *_a, **_k) -> None:
        pass

    def _clear_download_line(self, *_a, **_k) -> None:
        pass

    def _drain_next_queued(self) -> None:
        pass

    # Reuse the real handlers — they're plain UI-thread methods.
    def _on_server_ready(self, model_name):
        return ChatScreen._on_server_ready(self, model_name)

    def _on_server_restart_failed(self, msg):
        return ChatScreen._on_server_restart_failed(self, msg)

    def run_worker(self, fn, *_, **__):
        fn()  # run synchronously


class _Choice:
    def __init__(self) -> None:
        self.name = "New Model"
        self.key = "new-model"
        self.architecture = "qwen3"
        from pathlib import Path
        self.local_path = Path("/tmp/new-model.gguf")


class _RuntimeCfg:
    model = "/tmp/old-model.gguf"


class _AppCfg:
    runtime = _RuntimeCfg()


def test_model_swap_auto_inits_backend_and_restarts(monkeypatch) -> None:
    """Regression: /model swap before the first message must auto-init the
    backend and restart the server, NOT emit "Backend not initialized" and
    defer to the next typed message."""
    import localcode.models_catalog as catalog_mod
    import localcode.config as config_mod
    import localcode.bootstrap as bootstrap_mod

    # _apply_model_choice imports these names locally, so patch them at the
    # source modules. Current model differs from the chosen one (no
    # short-circuit) and the chosen GGUF is treated as fully downloaded so
    # we reach the restart path.
    monkeypatch.setattr(catalog_mod, "current", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "save_config", lambda _cfg: None)
    monkeypatch.setattr(
        bootstrap_mod, "is_download_complete", lambda _choice: True
    )

    tui = _FakeTui(_AppCfg())
    screen = _SwapScreenStub(tui)

    ChatScreen._apply_model_choice(screen, _Choice())

    # Backend was lazily initialized during the swap...
    assert tui.ensure_backend_calls >= 1
    assert tui.engine is not None
    # ...and the server actually restarted with the new model.
    assert tui.engine.engine.restart_calls == 1
    # No "Backend not initialized" error was surfaced; restart succeeded.
    assert screen._log.errors == []
    assert any("ready" in m.lower() for m in screen._log.infos)
    assert screen._server_restarting is False


# ── `!` bash-mode runs at the repo root (recovered fix) ──────────────

class _BashLog:
    def __init__(self):
        self.users, self.results, self.infos = [], [], []
    def append_user(self, m): self.users.append(m)
    def append_tool_result(self, m, error=False): self.results.append((m, error))
    def append_info(self, m): self.infos.append(m)
    def scroll_end(self, *a, **k): pass


class _BashEngine:
    def __init__(self, repo_root): self.repo_root = repo_root


class _BashTui:
    def __init__(self, repo_root): self.engine = _BashEngine(repo_root)


class _BashScreenStub:
    def __init__(self, repo_root):
        self.tui = _BashTui(repo_root)
        self.app = _FakeApp()
        self._log = _BashLog()
    def query_one(self, *_a, **_k): return self._log
    def run_worker(self, fn, *_a, **_k): return fn()  # inline


def test_bang_bash_runs_at_repo_root(tmp_path, monkeypatch):
    captured = {}
    import subprocess as _sp

    def _fake_run(argv, **kw):
        captured["cwd"] = kw.get("cwd")
        class _R: stdout, stderr, returncode = "ok", "", 0
        return _R()

    monkeypatch.setattr(_sp, "run", _fake_run)
    stub = _BashScreenStub(str(tmp_path))
    ChatScreen._run_bash(stub, "echo hi")
    assert captured["cwd"] == str(tmp_path), "user !command must run at the repo root"
    assert stub._log.users == ["! echo hi"]
    assert stub._log.results and stub._log.results[0][1] is False  # rendered, not error


def test_bang_bash_empty_shows_hint_runs_nothing(monkeypatch):
    import subprocess as _sp
    ran = {"n": 0}
    def _fake_run(*a, **k): ran["n"] += 1; raise AssertionError("should not run")
    monkeypatch.setattr(_sp, "run", _fake_run)
    stub = _BashScreenStub("/tmp")
    ChatScreen._run_bash(stub, "")
    assert ran["n"] == 0 and stub._log.infos, "bare ! shows a hint, runs nothing"


class _FakeMenuWidget:
    """Records .update() text and active/hidden class toggles."""
    def __init__(self):
        self.text = None
        self.classes = set()
    def update(self, t): self.text = t
    def add_class(self, c): self.classes.add(c)
    def remove_class(self, c): self.classes.discard(c)


class _HintStub:
    def __init__(self):
        self.menu = _FakeMenuWidget()
        self.status = _FakeMenuWidget()
        self._slash_matches = []
        self._slash_selected = 0
        self._slash_window = 0
        self._ptt_last_input_value = None
    def query_one(self, sel, *_a, **_k):
        return self.menu if sel == "#slash-menu" else self.status


def test_bang_shows_shell_mode_hint_without_capturing_enter():
    # Typing a leading `!` surfaces the shell-mode affordance below the input,
    # but keeps _slash_matches empty so Enter runs the command (falls through
    # to _submit_message) instead of the slash menu selecting a row.
    stub = _HintStub()
    ChatScreen._on_chat_text_changed(stub, "!git status")
    assert "active" in stub.menu.classes, "shell hint must be visible"
    assert "shell mode" in (stub.menu.text or ""), stub.menu.text
    assert "hidden" in stub.status.classes, "status bar steps aside for the hint"
    assert stub._slash_matches == [], "! must not populate slash matches"


def test_normal_text_hides_the_hint():
    stub = _HintStub()
    ChatScreen._on_chat_text_changed(stub, "!ls")   # show it
    ChatScreen._on_chat_text_changed(stub, "hello")  # then a normal message
    assert "active" not in stub.menu.classes
    assert "hidden" not in stub.status.classes


def test_chat_textarea_autosize_grows_and_clamps():
    # The input must grow with content (Codex-style) up to the terminal-relative
    # cap, then stop (scrolls internally). TextArea doesn't auto-grow from CSS,
    # so autosize_height() drives the row count. Unmounted (no app) the cap
    # falls back to the historical fixed value of 10.
    ta = ChatScreen  # ensure import
    from localcode.tui.screens.chat import _ChatTextArea
    cap = 10  # _max_input_lines() fallback when there's no mounted app
    small = _ChatTextArea(text="one\ntwo\nthree")
    small.autosize_height()
    assert small._max_input_lines() == cap
    assert 1 <= small.styles.height.value <= cap
    big = _ChatTextArea(text="\n".join(f"line {i}" for i in range(40)))
    big.autosize_height()
    assert big.styles.height.value == cap  # clamped


def test_chat_textarea_max_lines_is_terminal_relative(monkeypatch):
    # The cap mirrors the reference Codex input: ~half the terminal height,
    # floored at _MIN_INPUT_CAP and ceilinged at _MAX_INPUT_CAP. `app` is a
    # Textual property, so patch it at the class level to feed a fake size.
    from localcode.tui.screens.chat import _ChatTextArea

    class _FakeSize:
        def __init__(self, h):
            self.height = h

    class _FakeApp:
        def __init__(self, h):
            self.size = _FakeSize(h)

    ta = _ChatTextArea(text="x")
    holder = {"h": 24}
    monkeypatch.setattr(
        type(ta), "app",
        property(lambda self: _FakeApp(holder["h"])),
        raising=False,
    )
    # Short terminal → floored at the minimum (8//2 - 6 = -2 → MIN).
    holder["h"] = 8
    assert ta._max_input_lines() == _ChatTextArea._MIN_INPUT_CAP
    # Mid terminal (~24 rows) → ~half minus footer ≈ 6 (24//2 - 6 = 6).
    holder["h"] = 24
    assert ta._max_input_lines() == 6
    # Tall terminal → ceilinged at the maximum (80//2 - 6 = 34 → MAX).
    holder["h"] = 80
    assert ta._max_input_lines() == _ChatTextArea._MAX_INPUT_CAP


# ---------------------------------------------------------------------------
# Live-pilot regressions for the multi-line input + output-copy + /thinking
# gating. These mount the real app on Textual's headless driver so the
# message loop, layout, and call_after_refresh deferral all run for real.
# ---------------------------------------------------------------------------
import asyncio
import os


def _new_app():
    from localcode.tui.app import LocalCodeTUI
    os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
    app = LocalCodeTUI()
    app._preview_screen = "chat"  # skip health/server/model-picker → chat
    return app


def test_chat_input_grows_on_typing_and_paste_live():
    """BUG 1 regression: the TextArea must auto-grow as content wraps to more
    than one visual row, on BOTH typing and paste — measured after a refresh
    (call_after_refresh) so the wrapped-row count is current, not stale."""
    from textual.events import Paste

    async def scenario():
        app = _new_app()
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            ta = app.screen.query_one("#chat-input")
            ta.focus()
            await pilot.pause()
            assert int(ta.styles.height.value) == 1, "starts at one row"

            # Typed multi-line content (Ctrl+J inserts a newline).
            await pilot.press("a", "b", "c", "ctrl+j", "d", "e", "f", "ctrl+j", "g")
            await pilot.pause()
            await pilot.pause()
            assert ta.text == "abc\ndef\ng"
            assert int(ta.styles.height.value) == 3, (
                f"typed 3 lines should grow to 3 rows, got {ta.styles.height.value}"
            )

            # A moderate single-line paste (below the large-paste collapse
            # thresholds: <4 lines AND <300 chars) stays inline and must
            # soft-wrap and grow past one row.
            ta.text = ""
            await pilot.pause()
            ta.post_message(Paste("x" * 200))
            await pilot.pause()
            await pilot.pause()
            assert int(ta.styles.height.value) > 1, (
                "a 200-char paste must wrap and grow the box past one row"
            )

    asyncio.run(scenario())


def test_large_paste_collapses_to_chip_and_expands_on_submit():
    """A large paste is replaced in the composer by a single `[pasted …]`
    chip (not the full flood); the real text is spliced back in when the
    message is submitted."""
    from textual.events import Paste

    async def scenario():
        app = _new_app()
        submitted = {}
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            ta = screen.query_one("#chat-input")
            ta.focus()
            await pilot.pause()

            # 40-line paste → well over the 4-line collapse threshold.
            big = "\n".join(f"line {i}" for i in range(40))
            ta.post_message(Paste(big))
            await pilot.pause()
            await pilot.pause()

            # Composer shows a compact chip, not the 40-line flood.
            assert "[pasted #1 +40 lines]" in ta.text
            assert "line 39" not in ta.text
            assert int(ta.styles.height.value) <= 3, (
                "collapsed paste must not grow the box like the raw flood would"
            )

            # Capture what _start_turn receives (avoid a real backend turn).
            screen._start_turn = lambda text: submitted.update(text=text)
            screen._submit_message(ta.text)
            await pilot.pause()

            # The submitted message carries the REAL pasted text, expanded.
            assert "line 0" in submitted["text"]
            assert "line 39" in submitted["text"]
            assert "[pasted" not in submitted["text"]

    asyncio.run(scenario())


def test_chat_log_output_is_selectable_and_copies_live():
    """BUG 2 regression: agent output in the chat log can be selected by
    click-drag and the selected text is extracted + pushed to the clipboard
    (OSC52 / pbcopy), since Textual's mouse capture blocks native selection."""

    async def scenario():
        app = _new_app()
        copied = {}
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            from localcode.tui.widgets.chat_log import ChatLog as _CL
            log = app.screen.query_one("#chat-log", _CL)
            # Capture whatever the widget tries to put on the clipboard.
            log._set_clipboard_osc52 = lambda text: copied.update(text=text)
            for i in range(6):
                log.append_info(f"agent output row {i}")
            await pilot.pause()

            class _E:
                def __init__(self, x, y):
                    self.x, self.y = x, y

            log.on_mouse_down(_E(2, 1))
            log.on_mouse_move(_E(40, 3))
            log.on_mouse_up(_E(40, 3))

            assert log.get_selection().strip(), "drag must yield selected text"
            assert copied.get("text", "").strip(), "selection must reach the clipboard"

    asyncio.run(scenario())


def test_thinking_command_disabled_for_non_reasoning_model_live():
    """BUG 4 regression: /thinking is greyed-out + non-selectable + a no-op
    on models without a hidden-reasoning channel (diffusion), and fully works
    on a reasoning model."""

    async def scenario():
        from localcode.models_catalog import by_key, current

        app = _new_app()
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.pause()
            scr = app.screen

            # Diffusion model → no thinking channel.
            app.config.runtime.model = by_key("diffusiongemma").filename
            assert current(app.config).key == "diffusiongemma"
            assert scr._model_supports_thinking() is False
            assert scr._slash_cmd_disabled("/thinking") is True
            assert scr._slash_cmd_disabled("/model") is False

            # Menu renders /thinking as unavailable, not as a usable row.
            scr._slash_matches = [
                ("/permissions", "x"),
                ("/thinking", "Show / set hidden reasoning policy"),
                ("/model", "y"),
            ]
            scr._slash_selected = 0
            scr._render_slash_menu()
            menu_text = str(scr.query_one("#slash-menu").render())
            assert "unavailable for this model" in menu_text

            # Navigation skips the disabled /thinking row (1 → 2).
            assert scr._next_selectable_slash(0, +1) == 2

            # Direct command is a no-op on a non-reasoning model.
            app.config.runtime.internal_thinking_mode = "off"
            scr._handle_thinking_command("/thinking auto")
            assert app.config.runtime.internal_thinking_mode == "off"

            # Reasoning model → enabled + selectable + functional.
            app.config.runtime.model = by_key("qwen").filename
            assert scr._model_supports_thinking() is True
            assert scr._slash_cmd_disabled("/thinking") is False
            assert scr._next_selectable_slash(0, +1) == 1
            scr._handle_thinking_command("/thinking auto")
            assert app.config.runtime.internal_thinking_mode == "auto"

    asyncio.run(scenario())
