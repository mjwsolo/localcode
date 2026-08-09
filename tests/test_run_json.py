"""`localcode run --json` headless API — JSONL event stream.

These exercise `_run_headless_json` with a fully stubbed app/engine so no
real model inference happens. We capture stdout, then assert every emitted
line is a single valid JSON object and that the expected event types +
terminal summary are present.
"""
from __future__ import annotations

import json
import types

import pytest

from localcode import entrypoint
from localcode.headless_json import JsonlEmitter, run_headless_json


# ── Helpers ──────────────────────────────────────────────────────────


class _FakeOut:
    """Minimal stand-in for OutputManager — just records the callback."""

    def __init__(self) -> None:
        self._cb = None

    def set_event_callback(self, cb) -> None:
        self._cb = cb

    def emit(self, event_type, **payload) -> None:
        if self._cb is not None:
            self._cb(event_type, payload)


class _FakeEngine:
    def _restart_server(self) -> bool:  # never called (probe is stubbed up)
        return True


class _FakeApp:
    """Emits a scripted sequence of agent events, then returns final text."""

    last_goal = None

    def __init__(self, config, profile_name=None) -> None:
        self.out = _FakeOut()
        self.engine = _FakeEngine()

    def ask(self, goal, stream=True) -> str:
        _FakeApp.last_goal = goal
        # Simulate the agent loop firing UI events through OutputManager.
        self.out.emit("content", chunk="Hello", chars="5")
        self.out.emit("tool_start", name="bash", args="ls -la", index="0")
        self.out.emit("tool_result", error="false", index="0", result="file.txt", name="bash")
        self.out.emit("turn_tokens", prompt_tokens="100", completion_tokens="20", total_tokens="120")
        self.out.emit("content", chunk=" world", chars="6")
        return "Hello world"


def _make_args(**over):
    base = dict(goal="do a thing", model=None, binary=None, timeout=0,
                quiet=False, json=True, profile=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def _make_config():
    runtime = types.SimpleNamespace(model="", llama_cpp_binary="")
    return types.SimpleNamespace(runtime=runtime)


def _patch_backend(monkeypatch, app_cls=_FakeApp, model_on_disk=True):
    """Stub the heavy imports `_run_headless_json` pulls in lazily."""
    import localcode.app as app_mod
    import localcode.server_manager as sm_mod
    import localcode.bootstrap as bs_mod
    import localcode.models_catalog as mc_mod

    monkeypatch.setattr(app_mod, "LocalCodeApp", app_cls, raising=False)
    # Server already healthy → no restart, no real server.
    monkeypatch.setattr(sm_mod, "_probe_health", lambda *a, **k: True, raising=False)
    monkeypatch.setattr(bs_mod, "get_model_path", lambda name: None, raising=False)

    class _Choice:
        def __init__(self) -> None:
            self.size_gb = 1.0
            self.local_path = types.SimpleNamespace(exists=lambda: model_on_disk)

    monkeypatch.setattr(mc_mod, "CHOICES", [_Choice()], raising=False)


def _parse_lines(captured: str) -> list[dict]:
    lines = [l for l in captured.splitlines() if l.strip()]
    out = []
    for line in lines:
        out.append(json.loads(line))  # raises if any line isn't valid JSON
    return out


# ── Tests ────────────────────────────────────────────────────────────


def test_jsonl_stream_is_all_valid_json(monkeypatch, capsys):
    _patch_backend(monkeypatch)
    code = run_headless_json(_make_config(), _make_args())
    assert code == 0

    events = _parse_lines(capsys.readouterr().out)
    # Each line parsed as JSON (above) and carries a `type`.
    assert all("type" in e for e in events)


def test_expected_event_types_present(monkeypatch, capsys):
    _patch_backend(monkeypatch)
    run_headless_json(_make_config(), _make_args())
    events = _parse_lines(capsys.readouterr().out)
    types_seen = [e["type"] for e in events]
    for expected in ("content", "tool_start", "tool_result", "turn_tokens", "result"):
        assert expected in types_seen, f"missing event type {expected}"
    # `result` is the terminal event — must be last.
    assert types_seen[-1] == "result"


def test_result_event_summary(monkeypatch, capsys):
    _patch_backend(monkeypatch)
    code = run_headless_json(_make_config(), _make_args())
    events = _parse_lines(capsys.readouterr().out)
    result = events[-1]
    assert result["type"] == "result"
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert result["final_text"] == "Hello world"
    # turn_tokens accumulated into the summary.
    assert result["tokens"]["prompt"] == 100
    assert result["tokens"]["completion"] == 20
    assert result["tokens"]["total"] == 120
    assert code == 0


def test_no_model_emits_error_result(monkeypatch, capsys):
    _patch_backend(monkeypatch, model_on_disk=False)
    code = run_headless_json(_make_config(), _make_args())
    assert code == 1
    events = _parse_lines(capsys.readouterr().out)
    assert len(events) == 1
    assert events[0]["type"] == "result"
    assert events[0]["status"] == "error"
    assert events[0]["exit_code"] == 1


def test_ask_exception_becomes_error_result(monkeypatch, capsys):
    class _BoomApp(_FakeApp):
        def ask(self, goal, stream=True) -> str:
            raise ValueError("boom")

    _patch_backend(monkeypatch, app_cls=_BoomApp)
    code = run_headless_json(_make_config(), _make_args())
    assert code == 1
    events = _parse_lines(capsys.readouterr().out)
    result = events[-1]
    assert result["type"] == "result"
    assert result["status"] == "error"
    assert "boom" in result["reason"]


def test_keyboard_interrupt_exit_130(monkeypatch, capsys):
    class _IntApp(_FakeApp):
        def ask(self, goal, stream=True) -> str:
            raise KeyboardInterrupt

    _patch_backend(monkeypatch, app_cls=_IntApp)
    code = run_headless_json(_make_config(), _make_args())
    assert code == 130
    events = _parse_lines(capsys.readouterr().out)
    assert events[-1]["status"] == "interrupted"
    assert events[-1]["exit_code"] == 130


def test_run_subparser_has_json_flag():
    parser = entrypoint.build_parser()
    ns = parser.parse_args(["run", "--goal", "x", "--json"])
    assert ns.json is True
    ns2 = parser.parse_args(["run", "--goal", "x"])
    assert ns2.json is False


def test_emitter_accumulates_tokens():
    import io

    buf = io.StringIO()
    em = JsonlEmitter(buf)
    em.emit("turn_tokens", {"prompt_tokens": "10", "completion_tokens": "5", "total_tokens": "15"})
    em.emit("turn_tokens", {"prompt_tokens": "3", "completion_tokens": "2", "total_tokens": "5"})
    em.result(status="ok", exit_code=0, reason="done", final_text="hi")
    lines = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    result = lines[-1]
    assert result["tokens"] == {"prompt": 13, "completion": 7, "total": 20}


def test_terminal_from_status_maps_failure_to_nonzero_exit():
    """Regression: headless exit code must reflect the loop's real completion
    status. Previously it read the always-empty `emitter.last_turn_end`, so a
    stalled/blocked/errored run still reported ('ok', 0)."""
    from localcode.headless_json import _terminal_from_status as T
    # the only success value from status_for_exit
    assert T(completion_status="completed", blocked_reason="",
             loop_exit_reason="model_done")[:2] == ("ok", 0)
    # a stalled run must NOT report success
    assert T(completion_status="incomplete", blocked_reason="",
             loop_exit_reason="stall_exhausted")[:2] == ("incomplete", 1)
    # blocked-on-user-input: non-zero exit, reason carries the blocked text
    s, code, reason, _ = T(completion_status="blocked_user_input",
                           blocked_reason="need creds",
                           loop_exit_reason="blocked_question")
    assert (s, code) == ("incomplete", 1) and reason == "need creds"
    # an unpopulated status is not a false failure
    assert T(completion_status="", blocked_reason="", loop_exit_reason="")[:2] == ("ok", 0)
