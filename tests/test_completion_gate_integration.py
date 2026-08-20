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


def _drive_with_verification(tmp_path: Path, repo: Path):
    """Like `_drive`, but the model also RUNS the project's typecheck itself —
    which is what records `relevant-verification` and lets a turn complete."""
    script = [
        tool_round(("write_file", {
            "path": "src/app.ts",
            "content": "export const app = () => 'todo';\n",
        })),
        tool_round(("bash", {"command": "./node_modules/.bin/tsc --noEmit"})),
        say("Done — the app is built and type-checks."),
    ]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), BUILD_PROMPT)
    return trace, app


def test_a_green_checker_completes_successfully(tmp_path):
    """Asserting only that the unverified reason is ABSENT would pass even if
    the turn still came back incomplete — so assert the completion itself."""
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 0\n")
    trace, app = _drive_with_verification(tmp_path, repo)
    status, code, _reason, text = _headless(app, trace)
    assert "Project typecheck was not verified" not in text
    assert (status, code) == ("ok", 0), f"green turn came back {status}: {text!r}"


def test_a_failing_checker_blocks_even_when_the_model_verified(tmp_path):
    """The registry being satisfied must NOT excuse a checker that never
    returned a verdict — that was the round-2 hole."""
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 2\n")
    trace, app = _drive_with_verification(tmp_path, repo)
    status, code, _reason, text = _headless(app, trace)
    assert (status, code) == ("incomplete", 1)
    assert "Project typecheck was not verified" in text


# ── the TUI must show the reason too ────────────────────────────────────────


def _tui_drive(tmp_path: Path, repo: Path, script, prompt: str) -> str:
    """Drive the REAL Textual app headlessly and return the visible chat log.

    Mirrors tests/test_comprehensive_tui.py — the presentation path the
    integration test above bypasses, which is exactly why the TUI silently
    showed "Done" for a turn the loop had marked incomplete.
    """
    import asyncio

    from localcode.tui.app import LocalCodeTUI

    async def scenario() -> str:
        os.environ["LOCALCODE_AUTONOMY"] = "full_auto"
        app = LocalCodeTUI()
        app._preview_screen = "chat"
        async with app.run_test() as pilot:
            await pilot.pause()
            backend = build_test_app(tmp_path, script=script, cwd=repo)
            app.engine = backend
            backend.out.set_event_callback(app.bridge.on_event)
            backend.out.set_approval_callback(app.bridge.request_approval)
            app.screen.query_one("#chat-input").value = prompt
            await pilot.press("enter")
            for _ in range(400):  # ~20s ceiling
                await pilot.pause(0.05)
                if not getattr(app.screen, "_agent_busy", False):
                    break
            log = app.screen.query_one("#chat-log")
            return "\n".join(s.text for s in log.lines)

    return asyncio.run(scenario())


def test_tui_shows_the_unverified_reason(tmp_path, failing_checker_repo):
    """`response_done` rebuilt the answer from `_stream_buf` (model text only)
    and `_render_markdown` went to the worker's /dev/null stdout, so the user
    saw "Done" while the turn was persisted as incomplete."""
    script = [
        tool_round(("write_file", {
            "path": "src/app.ts",
            "content": "export const app = () => 'todo';\n",
        })),
        say("Done — the app is built."),
    ]
    log_text = _tui_drive(tmp_path, failing_checker_repo, script, BUILD_PROMPT)
    assert "Project typecheck was not verified" in log_text
    assert "remains incomplete" in log_text


def test_tui_still_shows_a_plain_model_answer(tmp_path):
    """No regression on the ordinary path: with nothing appended by the loop,
    the model's own text is what renders."""
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n")
    log_text = _tui_drive(tmp_path, repo, [say("Hello from the model.")], "hi there")
    assert "Hello from the model." in log_text


# ── a checker that stays RED ────────────────────────────────────────────────
#
# Regression for the Anki-clone run that finished with three real tsc errors
# and a `// placeholder` module at the centre of the app. RED is a verdict, so
# it cleared the gate's no-verdict state — but nothing else remembered it. Once
# the bounded nudges ran out, the tier-2 block stopped running altogether and
# the turn completed as "verified" with the project still broken. The
# no-verdict path already states the rule ("running out of retries does NOT
# make the project verified"); RED simply never implemented it.


@pytest.fixture
def red_checker_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir()
    # Nonzero WITH diagnostics: the checker worked and the project is broken.
    _ts_project(repo, (
        "#!/bin/sh\n"
        "echo \"src/pages/Study.tsx(3,10): error TS2305: Module '../lib/fsrs' \"\n"
        "exit 2\n"
    ))
    return repo


def test_a_persistently_red_check_does_not_complete_successfully(tmp_path, red_checker_repo):
    trace, app = _drive(tmp_path, red_checker_repo)
    assert trace.error is None
    status, code, _reason, text = _headless(app, trace)
    assert status == "incomplete" and code == 1, (
        f"a project with real typecheck errors completed as {status!r}")
    assert "not verified" in text or "remains incomplete" in text


def test_a_red_check_is_re_run_after_the_nudges_are_exhausted(
        tmp_path, red_checker_repo, monkeypatch):
    """The bound limits how many times the model is NUDGED, not whether the
    project is checked. If checking stops, a stale red would either be
    forgotten (the bug) or block a build the model just fixed."""
    import localcode.tools.project_check as pc

    calls = []
    real = pc.run_project_check_result

    def _spy(*a, **kw):
        out = real(*a, **kw)
        calls.append(out.status)
        return out

    monkeypatch.setattr(pc, "run_project_check_result", _spy)
    _trace, _app = _drive(tmp_path, red_checker_repo)
    assert len(calls) >= 3, f"checker ran only {len(calls)} time(s): {calls}"
    assert set(calls) == {"errors"}


def test_the_red_diagnostics_reach_the_final_text(tmp_path, red_checker_repo):
    _trace, app = _drive(tmp_path, red_checker_repo)
    _status, _code, _reason, text = _headless(app, _trace)
    assert "typecheck" in text.lower()


def test_a_red_checker_blocks_even_when_the_model_verified(tmp_path, red_checker_repo):
    """The 360 case exactly: the model ran its own build (satisfying the
    verification registry) while the project typecheck stayed RED. RED cleared
    the gate's no-verdict state and nothing else remembered it, so once the
    nudges ran out the turn completed as verified with the build still broken."""
    trace, app = _drive_with_verification(tmp_path, red_checker_repo)
    status, code, _reason, text = _headless(app, trace)
    assert (status, code) == ("incomplete", 1), (
        f"a red typecheck completed as {status!r} because the model self-verified")
    assert "typecheck" in text.lower()


# ── a hollow module under a GREEN checker ───────────────────────────────────


def test_a_placeholder_module_blocks_even_with_a_green_checker(tmp_path):
    """The other half of the Anki failure. A typecheck can be perfectly green
    over a module that was imported and left empty, so the red gate alone would
    let this through — the app builds and the feature does not exist."""
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 0\n")
    script = [
        tool_round(("write_file", {
            "path": "src/lib/fsrs.ts",
            "content": "// placeholder\n",
        })),
        tool_round(("write_file", {
            "path": "src/app.ts",
            "content": "import { schedule } from './lib/fsrs';\nexport const a = schedule;\n",
        })),
        say("Done — spaced repetition is implemented."),
    ]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), BUILD_PROMPT)
    assert trace.error is None
    status, code, _reason, text = _headless(app, trace)
    assert (status, code) == ("incomplete", 1), (
        f"a hollow imported module completed as {status!r}: {text!r}")
    # Assert the HOLLOW gate's own words. Matching a bare "fsrs" passes on the
    # grounded file summary, which is how a dead-code version of this gate first
    # went green here.
    assert "imported but contain no code" in text
    assert "src/lib/fsrs.ts" in text


def test_a_real_module_under_a_green_checker_completes(tmp_path):
    """The false-positive guard at loop level: the same shape, implemented."""
    repo = tmp_path / "project"
    repo.mkdir()
    _ts_project(repo, "#!/bin/sh\nexit 0\n")
    script = [
        tool_round(("write_file", {
            "path": "src/lib/fsrs.ts",
            "content": "export const schedule = (n: number) => n * 2;\n",
        })),
        tool_round(("write_file", {
            "path": "src/app.ts",
            "content": "import { schedule } from './lib/fsrs';\nexport const a = schedule(1);\n",
        })),
        tool_round(("bash", {"command": "./node_modules/.bin/tsc --noEmit"})),
        say("Done — the app is built and type-checks."),
    ]
    app = build_test_app(tmp_path, script=script, cwd=repo)
    trace = run_one_turn(app, EventRecorder(), BUILD_PROMPT)
    status, code, _reason, text = _headless(app, trace)
    assert (status, code) == ("ok", 0), f"a real implementation came back {status}: {text!r}"
