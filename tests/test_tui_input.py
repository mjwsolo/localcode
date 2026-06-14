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
