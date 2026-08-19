"""End-to-end: an unverifiable project typecheck must block completion.

Drives the REAL agent loop (scripted fake model, real tools, real
project_check) against a throwaway repo whose `tsc` exits nonzero with no
output — the "checker failed, verdict unknown" case that used to read as clean.

Then it feeds the turn status the loop persisted through the SAME function
`localcode run --json` uses, so the assertions cover the JSON contract
(`status: incomplete`, exit 1, reason in `final_text`) and not just internals.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode.headless_json import _terminal_from_status
from tests.e2e.fake_runtime import build_test_app, say, tool_round
from tests.e2e.harness import EventRecorder, run_one_turn

BUILD_PROMPT = "Build me a small todo web app in TypeScript"


def _ts_project(repo: Path, tsc_script: str) -> None:
    """A TS project whose checker behaves as scripted."""
    (repo / "package.json").write_text(json.dumps({"name": "demo", "scripts": {}}))
    (repo / "tsconfig.json").write_text(json.dumps({"include": ["src"]}))
    (repo / "src").mkdir(exist_ok=True)
    binp = repo / "node_modules" / ".bin"
    binp.mkdir(parents=True, exist_ok=True)
    tsc = binp / "tsc"
    tsc.write_text(tsc_script)
    os.chmod(tsc, 0o755)


def _drive(tmp_path: Path, repo: Path):
    script = [
        tool_round(("write_file", {
            "path": "src/app.ts",
            "content": "export const app = () => 'todo';\n",
        })),
        say("Done — the app is built."),
    ]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), BUILD_PROMPT)
    return trace, app


def _headless(app, trace):
    return _terminal_from_status(
        completion_status=str(getattr(app, "_last_turn_completion_status", "") or ""),
        blocked_reason=str(getattr(app, "_last_turn_blocked_reason", "") or ""),
        loop_exit_reason=str(getattr(app, "_last_turn_loop_exit_reason", "") or ""),
        final_text=trace.final_response or "",
    )


# ── the failure case ────────────────────────────────────────────────────────


@pytest.fixture
def failing_checker_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    # Nonzero with NO output: the checker ran but produced no verdict.
    _ts_project(repo, "#!/bin/sh\nexit 2\n")
    return repo


def test_unverifiable_check_does_not_complete_successfully(tmp_path, failing_checker_repo):
    trace, app = _drive(tmp_path, failing_checker_repo)
    assert trace.error is None
    status, code, _reason, text = _headless(app, trace)
    assert status == "incomplete" and code == 1
    assert "not verified" in text or "remains incomplete" in text


def test_reason_reaches_the_json_final_text(tmp_path, failing_checker_repo):
    """print_info is invisible in the TUI and suppressed under --json, so the
    reason has to ride the final result text."""
    trace, app = _drive(tmp_path, failing_checker_repo)
    _status, _code, _reason, text = _headless(app, trace)
    assert "Project typecheck was not verified" in text
    assert "exited 2 with no output" in text


def test_the_failing_checker_is_re_run_every_completion_attempt(
        tmp_path, failing_checker_repo, monkeypatch):
    """The bug: only the FIRST failure forced a retry; later ones fell through
    to a successful completion. The gate must keep re-checking."""
    import localcode.tools.project_check as pc

    calls = []
    real = pc.run_project_check_result

    def _spy(*a, **kw):
        out = real(*a, **kw)
        calls.append(out.status)
        return out

    monkeypatch.setattr(pc, "run_project_check_result", _spy)
    _trace, _app = _drive(tmp_path, failing_checker_repo)
    assert len(calls) >= 3, f"checker ran only {len(calls)} time(s): {calls}"
    assert set(calls) == {"failed"}


def test_the_reason_is_not_carried_by_print_info(tmp_path, failing_checker_repo):
    """print_info writes straight to stdout and emits no event, so it is
    invisible in the TUI and absent from `--json`. Pinning that here keeps the
    final-result path as the channel the reason actually travels on."""
    trace, app = _drive(tmp_path, failing_checker_repo)
    assert not any("stays unverified" in m for m in trace.info_messages())
    _status, _code, _reason, text = _headless(app, trace)
    assert "Project typecheck was not verified" in text


def test_timeout_also_blocks_completion(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nsleep 30\n")
    monkeypatch.setattr("localcode.tools.project_check._TIMEOUT", 0.5)
    trace, app = _drive(tmp_path, repo)
    status, code, _reason, text = _headless(app, trace)
    assert status == "incomplete" and code == 1
    assert "timed out" in text


def test_checker_exception_blocks_completion(tmp_path, monkeypatch):
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 0\n")

    def _boom(*_a, **_kw):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr("localcode.tools.project_check.run_project_check_result", _boom)
    trace, app = _drive(tmp_path, repo)
    status, _code, _reason, text = _headless(app, trace)
    assert status == "incomplete"
    assert "RuntimeError" in text


# ── the clean case must still complete ──────────────────────────────────────


def test_a_green_checker_does_not_block(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 0\n")
    trace, app = _drive(tmp_path, repo)
    _status, _code, _reason, text = _headless(app, trace)
    assert "Project typecheck was not verified" not in text
