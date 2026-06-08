"""End-to-end agent-loop feature coverage, driven by a scripted fake model.

Every test here runs the REAL agent loop and the REAL tool implementations
against a throwaway repo — only the model is faked (see
tests/e2e/fake_runtime.py). So a green run proves: prompt flow, tool
dispatch, tool side effects on disk, multi-round tool use, thinking,
tool-result feedback, and error handling all work — with no model
download and full determinism.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tests.e2e.fake_runtime import build_test_app, say, tool_round
from tests.e2e.harness import EventRecorder, run_one_turn


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small project tree the agent can operate on."""
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "main.py").write_text("def hello():\n    return 'world'\n")
    (repo / "utils.py").write_text("import os\n\nVALUE = 41\n")
    (repo / "README.md").write_text("# Demo\n")
    return repo


def _run(tmp_path, project, script, prompt):
    app = build_test_app(tmp_path, script=script, cwd=project)
    rec = EventRecorder()
    return run_one_turn(app, rec, prompt), app


# ── Plain conversational turn (no tools) ────────────────────────────


def test_plain_prompt_returns_text(tmp_path, project):
    trace, _ = _run(tmp_path, project, [say("Hello there!")], "hi")
    assert trace.error is None
    assert "Hello there!" in trace.content_text()
    assert trace.tool_calls_made() == []


def test_thinking_is_streamed(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [say("42", thinking="the answer is forty two")],
        "what is the answer",
    )
    assert trace.error is None
    assert "forty two" in trace.thinking_text()
    assert "42" in trace.content_text()


# ── Read / search tools ─────────────────────────────────────────────


def test_read_file_tool(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [tool_round(("read_file", {"path": "main.py"})), say("It returns 'world'.")],
        "what does main.py return?",
    )
    assert trace.error is None
    assert "read_file" in trace.tool_calls_made()
    assert "world" in trace.content_text()


def test_grep_tool(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [tool_round(("grep", {"pattern": "VALUE"})), say("Found it in utils.py.")],
        "where is VALUE defined?",
    )
    assert trace.error is None
    assert "grep" in trace.tool_calls_made()


def test_glob_and_list_files_tools(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [tool_round(("glob", {"pattern": "*.py"}), ("list_files", {})), say("Two py files.")],
        "list the python files",
    )
    assert trace.error is None
    called = trace.tool_calls_made()
    assert "glob" in called and "list_files" in called


# ── Mutating tools — assert real disk side effects ──────────────────


def test_write_file_creates_file(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [tool_round(("write_file", {"path": "created.py", "content": "X = 1\n"})), say("done")],
        "create created.py",
    )
    assert trace.error is None
    assert (project / "created.py").read_text() == "X = 1\n"


def test_edit_file_applies_change(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [
            tool_round(("edit_file", {
                "path": "utils.py",
                "old_string": "VALUE = 41",
                "new_string": "VALUE = 42",
            })),
            say("fixed"),
        ],
        "fix the value",
    )
    assert trace.error is None
    assert "VALUE = 42" in (project / "utils.py").read_text()


def test_bash_tool_runs_command(tmp_path, project):
    trace, _ = _run(
        tmp_path, project,
        [tool_round(("bash", {"command": "echo hello-from-bash"})), say("ran it")],
        "echo something",
    )
    assert trace.error is None
    assert "bash" in trace.tool_calls_made()


# ── Multi-round tool use ────────────────────────────────────────────


def test_multi_round_read_then_edit(tmp_path, project):
    """Model reads a file, then edits it, then answers — 3 model rounds."""
    app = build_test_app(tmp_path, cwd=project)
    app.engine.script = [
        tool_round(("read_file", {"path": "utils.py"})),
        tool_round(("edit_file", {
            "path": "utils.py",
            "old_string": "VALUE = 41",
            "new_string": "VALUE = 100",
        })),
        say("Updated VALUE to 100."),
    ]
    rec = EventRecorder()
    trace = run_one_turn(app, rec, "bump VALUE to 100")
    assert trace.error is None
    assert trace.tool_calls_made() == ["read_file", "edit_file"]
    assert "VALUE = 100" in (project / "utils.py").read_text()
    # The model was called once per round + once for the final answer.
    assert len(app.engine.calls) == 3


def test_tool_result_is_fed_back_to_model(tmp_path, project):
    """After a tool runs, its output must appear in the next model call's
    messages — that's the loop closing the tool→model feedback path."""
    app = build_test_app(tmp_path, cwd=project)
    app.engine.script = [
        tool_round(("read_file", {"path": "main.py"})),
        say("ok"),
    ]
    rec = EventRecorder()
    trace = run_one_turn(app, rec, "read main.py")
    assert trace.error is None
    # Second model call should contain the file contents somewhere in its
    # message payload (fed back as a tool/role message).
    second_call_blob = str(app.engine.calls[1])
    assert "world" in second_call_blob


# ── Error handling ──────────────────────────────────────────────────


def test_model_stream_error_is_captured(tmp_path, project):
    app = build_test_app(tmp_path, cwd=project)
    app.engine.script = [RuntimeError("simulated server crash")]
    rec = EventRecorder()
    trace = run_one_turn(app, rec, "trigger an error")
    # The loop catches the stream failure and surfaces it as a graceful
    # `error` event (no crash, no escaped exception).
    assert trace.error is None, "exception should not escape the loop"
    error_blob = " ".join(trace.errors())
    assert "simulated server crash" in error_blob
    assert "model server" in error_blob.lower()
