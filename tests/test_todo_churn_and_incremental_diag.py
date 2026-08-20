"""Two fixes grounded in the Anki-run logs:

1. todo_write churn guard — a real run made 43 todo_write calls, all the SAME
   list, completed-count never moved (37% of every action wasted). Re-sending an
   unchanged plan must be refused with a steer, not accepted.
2. Auto-surfaced diagnostics after a code edit — the model used the LSP/diagnostic
   tools ZERO times, so LocalCode runs the typecheck FOR it after an edit and
   appends the errors to the tool result.
"""
from __future__ import annotations

from types import SimpleNamespace

from localcode.tools.base import ToolContext
from localcode.tools import dispatch
from localcode.tools import todo_write as T
import localcode.tools.project_check as pc


class _Out:
    def __getattr__(self, _):
        return lambda *a, **k: None


class _App:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.session = SimpleNamespace(todos=[], current_task=None)


def _ctx(tmp_path):
    return ToolContext(app=_App(tmp_path), out=_Out())


# ── 1. todo_write churn guard ────────────────────────────────────────────────

PLAN = [
    {"content": "Scaffold project", "status": "in_progress"},
    {"content": "Write fsrs module", "status": "pending"},
]


def test_first_todo_write_is_accepted(tmp_path):
    ctx = _ctx(tmp_path)
    assert "updated" in T.execute(ctx, {"todos": PLAN}).lower()


def test_identical_replan_is_refused_with_a_steer(tmp_path):
    ctx = _ctx(tmp_path)
    T.execute(ctx, {"todos": PLAN})
    out = T.execute(ctx, {"todos": PLAN})
    assert "UNCHANGED" in out and "STOP calling todo_write" in out
    assert "Scaffold project" in out  # names the in_progress step to act on


def test_repeated_churn_escalates(tmp_path):
    ctx = _ctx(tmp_path)
    T.execute(ctx, {"todos": PLAN})
    T.execute(ctx, {"todos": PLAN})
    out = T.execute(ctx, {"todos": PLAN})
    assert "same plan" in out.lower() and "loop" in out.lower()


def test_real_progress_is_accepted_and_resets_churn(tmp_path):
    ctx = _ctx(tmp_path)
    T.execute(ctx, {"todos": PLAN})
    T.execute(ctx, {"todos": PLAN})  # churn
    progressed = [
        {"content": "Scaffold project", "status": "completed"},
        {"content": "Write fsrs module", "status": "in_progress"},
    ]
    out = T.execute(ctx, {"todos": progressed})
    assert "updated" in out.lower()
    assert ctx.app.session._todo_churn == 0


# ── 2. auto-surfaced diagnostics after a code edit ───────────────────────────

def test_diagnostics_appended_after_editing_broken_code(tmp_path):
    pc._incr_state["ts"] = 0.0
    pc._incr_state["sig"] = None
    ctx = _ctx(tmp_path)
    # a Python syntax error is caught by ruff OR compileall (either detector)
    out = dispatch("write_file", ctx, {"path": "bad.py", "content": "def f(:\n    pass\n"})
    assert "reported errors" in out or "Typecheck after your edit" in out


def test_no_diagnostics_appended_for_non_code_file(tmp_path):
    pc._incr_state["ts"] = 0.0
    pc._incr_state["sig"] = None
    ctx = _ctx(tmp_path)
    out = dispatch("write_file", ctx, {"path": "notes.md", "content": "# hi\n"})
    assert "Typecheck after your edit" not in out


def test_diagnostics_debounced_on_immediate_second_edit(tmp_path):
    pc._incr_state["ts"] = 0.0
    pc._incr_state["sig"] = None
    ctx = _ctx(tmp_path)
    dispatch("write_file", ctx, {"path": "a.py", "content": "def f(:\n pass\n"})
    out2 = dispatch("write_file", ctx, {"path": "b.py", "content": "def g(:\n pass\n"})
    assert "Typecheck after your edit" not in out2  # within debounce window
