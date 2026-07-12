from pathlib import Path

from localcode.agent.goal import infer_goal_state
from localcode.agent.hooks import EvidenceLedger, TurnState, completion_gate, quality_monitor
from localcode.agent.app_tasks import is_focused_blocking_question
from localcode.agent.prompt_context import (
    build_incremental_milestones_block,
    build_target_grounding_block,
    build_task_goal_block,
)
from localcode.launcher import detect_launch_candidate
from localcode.process_registry import (
    ProcessRecord,
    latest_live_record,
    load_records,
    process_summary,
    record_process,
)
from localcode.skills import (
    Skill,
    SkillRegistry,
    dynamic_skill_block,
    load_registry,
    select_dynamic_skills,
)
from localcode.tools.read_file import execute as read_file
from localcode.tools.write_file import execute as write_file
from localcode.tools.append_file import execute as append_file
from localcode.tools.base import ToolContext
from localcode.tools.bash import (
    _looks_like_detached_server_command,
    _redirect_shell_file_write,
    execute as bash_execute,
)
from localcode.tools.facts import extract_tool_facts, facts_suffix
from localcode.tools import (
    dispatch_result,
    schemas_for_goal,
)
from localcode.agent.context import _prepare_model_messages, _semantic_tool_summary
from localcode.agent.tool_execution import (
    _HARD_STOP_THRESHOLD,
    ToolExecutionState,
    bash_cmd_key,
    canonical_args,
    dedup_stub_for_tool,
    track_tool_result,
    tool_result_is_error,
)
from localcode.agent.helpers import _execute_tool
from localcode.session import SessionStore


class _App:
    repo_root: Path

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.session = type("_Session", (), {"current_task": None})()


class _Out:
    pass


def _new_app_task(slug: str = "demo-app") -> object:
    return type(
        "_Task",
        (),
        {"task_kind": "new_app", "task_slug": slug, "goal_type": "build_app", "active_port": 0},
    )()


def test_write_file_overwrites_existing_file(tmp_path: Path) -> None:
    # write_file is the create-or-full-rewrite tool. Pre-2026-04-29 it
    # REJECTed any existing-path write to nudge the model toward
    # edit_file, but that forced wasteful read+plan+edit cycles every
    # time the model wanted to rewrite a stub it had just scaffolded.
    # The redaction-stub guard inside execute() still catches the
    # data.py-style "model copies the in-memory stub back" disaster,
    # which was the one case where rejecting was load-bearing.
    existing = tmp_path / "app.py"
    existing.write_text("print('old')\n")
    result = write_file(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"path": "app.py", "content": "print('new')\n"},
    )
    # A rewrite of an existing file now returns a unified diff (consistent
    # with edit_file) so the TUI renders a diff card; both the old and new
    # lines appear in it. New-file writes still return the "Created …" summary.
    assert "-print('old')" in result and "+print('new')" in result
    assert existing.read_text() == "print('new')\n"


def test_write_file_rejects_placeholder_code_across_languages(tmp_path: Path) -> None:
    result = write_file(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {
            "path": "src/main.rs",
            "content": "fn load() {\n    // TODO placeholder\n    unimplemented!();\n}\n",
        },
    )
    assert result.startswith("REJECTED:")


def test_append_file_adds_chunk_without_edit_anchor(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("def a():\n    return 1\n")

    result = append_file(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"path": "src/app.py", "content": "def b():\n    return 2\n"},
    )

    assert result.startswith("Appended ")
    assert target.read_text() == "def a():\n    return 1\ndef b():\n    return 2\n"


def test_append_file_creates_missing_file_for_chunked_generation(tmp_path: Path) -> None:
    target = tmp_path / "src" / "data" / "seed.py"

    result = append_file(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"path": "src/data/seed.py", "content": "WORDS = []\n"},
    )

    assert result.startswith("Created ")
    assert target.read_text() == "WORDS = []\n"


def test_session_load_discards_legacy_recovery_escalations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path / "home"))
    store = SessionStore()
    session = store.create(tmp_path / "repo")
    task = store.create_task(
        session,
        user_request="build an app",
        goal_type="build_app",
        task_kind="new_app",
        task_slug="demo-app",
        goal_summary="build an app",
        success_criteria=[],
    )
    store.save(session)
    path = store.sessions_dir / f"{session.session_id}.json"
    data = __import__("json").loads(path.read_text())
    data["current_task"]["large_write_escalations"] = 3
    path.write_text(__import__("json").dumps(data))

    loaded = store.load(session.session_id)

    assert loaded.current_task is not None
    assert not hasattr(loaded.current_task, "large_write_escalations")


def test_multi_edit_refuses_whole_file_collapse(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = "\n".join(f"print({i})" for i in range(100)) + "\n"
    target.write_text(original)
    result = _execute_tool(
        _App(tmp_path),
        "multi_edit",
        {"path": "app.py", "edits": [{"old_string": original, "new_string": "pass\n"}]},
        _Out(),
    )
    assert result.startswith("REJECTED:")
    assert target.read_text() == original


def test_multi_edit_is_atomic_when_later_edit_fails(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = "alpha\nbeta\ngamma\n"
    target.write_text(original)

    result = _execute_tool(
        _App(tmp_path),
        "multi_edit",
        {
            "path": "app.py",
            "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "missing", "new_string": "MISSING"},
            ],
        },
        _Out(),
    )

    assert "applied 0/2" in result
    assert target.read_text() == original


def test_multi_edit_rejects_overlapping_edits(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    original = "abcdef\n"
    target.write_text(original)

    result = _execute_tool(
        _App(tmp_path),
        "multi_edit",
        {
            "path": "app.py",
            "edits": [
                {"old_string": "abc", "new_string": "ABC"},
                {"old_string": "bcd", "new_string": "BCD"},
            ],
        },
        _Out(),
    )

    assert "overlaps edit" in result
    assert target.read_text() == original


def test_multi_edit_counts_as_changed_file_for_completion_state() -> None:
    state = ToolExecutionState()

    track_tool_result(
        tool_name="multi_edit",
        args={"path": "src/app.py", "edits": [{"old_string": "a", "new_string": "b"}]},
        tool_result="Edited src/app.py",
        round_num=1,
        state=state,
        dedup_stub=None,
    )

    assert state.changed_files == ["src/app.py"]


def test_task_slug_uses_content_words_not_prompt_prefix() -> None:
    state = infer_goal_state(
        "Help me build an app to learn music theory, all levels, audio and quiz."
    )
    # The slug is built from content words in order, not the imperative
    # prefix ("help me build an app to ...").
    assert state.task_slug.startswith("learn-music-theory")
    assert len(state.task_slug) <= 30
    assert "help-me-build" not in state.task_slug
    assert "music-theory" in state.task_slug


def test_legacy_recovery_content_limit_is_not_schema_routing_anymore() -> None:
    schemas = schemas_for_goal(
        "build_app",
        "build a data-heavy utility",
        task_stage="implementing",
        recovery_mode="legacy_recovery_final",
    )
    write_schema = next(schema for schema in schemas if schema["function"]["name"] == "write_file")

    assert "maxLength" not in write_schema["function"]["parameters"]["properties"]["content"]


def test_legacy_recovery_content_limit_does_not_reject_append(tmp_path: Path) -> None:
    app = _App(tmp_path)
    app._tool_content_max_chars = 12

    result = dispatch_result(
        "append_file",
        ToolContext(app=app, out=_Out()),
        {"path": "src/data.json", "content": "x" * 13},
    )

    assert result.ok
    assert (tmp_path / "src" / "data.json").read_text() == "x" * 13


def test_legacy_recovery_content_limit_does_not_reject_edit(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("alpha\n")
    app = _App(tmp_path)
    app._tool_content_max_chars = 12

    result = dispatch_result(
        "edit_file",
        ToolContext(app=app, out=_Out()),
        {"path": "src/app.py", "old_string": "alpha", "new_string": "x" * 13},
    )

    assert result.ok
    assert target.read_text() == "x" * 13 + "\n"


def test_repeated_failed_edit_call_is_rejected_before_execution() -> None:
    state = ToolExecutionState()
    args = {
        "path": "requirements.txt",
        "old_string": "flask==3.1.0",
        "new_string": "flask==3.1.0",
    }

    for round_num in (1, 2):
        track_tool_result(
            tool_name="edit_file",
            args=args,
            tool_result="Error: no-op edit on requirements.txt.",
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )

    stub = dedup_stub_for_tool("edit_file", args, state)

    assert stub is not None
    assert stub.startswith("REJECTED:")
    assert "already failed 2 times" in stub


def test_file_not_found_edit_counts_as_repeated_failure() -> None:
    state = ToolExecutionState()
    args = {
        "path": "requirements.txt",
        "old_string": "placeholder",
        "new_string": "flask==3.1.0\n",
    }

    assert tool_result_is_error("File not found: requirements.txt")

    for round_num in (1, 2):
        track_tool_result(
            tool_name="edit_file",
            args=args,
            tool_result="File not found: requirements.txt",
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )

    stub = dedup_stub_for_tool("edit_file", args, state)

    assert stub is not None
    assert stub.startswith("REJECTED:")
    assert "already failed 2 times" in stub


def test_repeated_same_path_rewrite_is_rejected_as_churn() -> None:
    """Rewrite churn: the model wrote `useData.ts` 14× in one turn. Each
    write SUCCEEDS with slightly different content, so the exact-args
    failed_calls/success_counts guards never fire — only a path-keyed
    counter catches it. After the churn threshold, the next rewrite of the
    same path is rejected with a 'stop and verify' nudge."""
    state = ToolExecutionState()
    path = "src/hooks/useData.ts"

    # Three SUCCESSFUL writes to the same path, each with DIFFERENT content
    # (near-duplicate rewrites — distinct arg blobs every round).
    for round_num, body in enumerate(
        ("export const useData = () => 1;\n",
         "export const useData = () => 2;\n",
         "export const useData = () => 3;\n"),
        start=1,
    ):
        args = {"path": path, "content": body}
        assert dedup_stub_for_tool("write_file", args, state) is None
        track_tool_result(
            tool_name="write_file",
            args=args,
            tool_result=f"Rewrote {path} (1 lines)",
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )

    # The 4th rewrite of the same path is now churn — reject before execution.
    stub = dedup_stub_for_tool(
        "write_file", {"path": path, "content": "export const useData = () => 4;\n"}, state
    )
    assert stub is not None
    assert stub.startswith("REJECTED:")
    assert path in stub
    assert "3 times" in stub
    assert "build" in stub.lower()

    # An edit_file to the SAME path is also caught (path-keyed, not tool-keyed).
    edit_stub = dedup_stub_for_tool(
        "edit_file", {"path": path, "old_string": "1", "new_string": "9"}, state
    )
    assert edit_stub is not None and edit_stub.startswith("REJECTED:")

    # A DIFFERENT path is unaffected — the guard is per-path.
    assert dedup_stub_for_tool(
        "write_file", {"path": "src/other.ts", "content": "x\n"}, state
    ) is None


def test_two_rewrites_of_same_path_are_allowed() -> None:
    """The churn guard must not fire on normal scaffold→fill-in→fix cycles.
    Two prior writes to a path is under threshold and stays silent."""
    state = ToolExecutionState()
    path = "app.py"
    for round_num in (1, 2):
        args = {"path": path, "content": f"print({round_num})\n"}
        track_tool_result(
            tool_name="write_file",
            args=args,
            tool_result=f"Rewrote {path} (1 lines)",
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )
    assert dedup_stub_for_tool(
        "write_file", {"path": path, "content": "print(3)\n"}, state
    ) is None


def test_failed_writes_do_not_count_toward_churn() -> None:
    """A rejected/failed write never landed on disk, so it isn't churn.
    Only successful writes accrue against the per-path threshold."""
    state = ToolExecutionState()
    path = "broken.py"
    for round_num in (1, 2, 3, 4):
        # Distinct content each round so the exact-args failed_calls guard
        # (keyed on canonical_args) stays under its own threshold; we're
        # isolating the path-keyed churn behavior here.
        args = {"path": path, "content": f"def f(:\n# {round_num}\n"}
        track_tool_result(
            tool_name="write_file",
            args=args,
            tool_result="Error: could not write broken.py",
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )
    # Failed writes leave write_path_counts empty → no churn stub.
    assert state.write_path_counts.get(path, 0) == 0
    assert dedup_stub_for_tool(
        "write_file", {"path": path, "content": "def f():\n    return 1\n"}, state
    ) is None


def test_read_file_dedup_is_disabled_to_unblock_debug_loops() -> None:
    """Removed 2026-04-29 after observing a 17-minute hang where the
    model was legitimately re-reading a file to fix a verification gap
    (bash probe failed, model needed fresh inspection) but every
    read_file got starved with `[DEDUP …]` instead of bytes. The dedup
    couldn't tell "useless re-read" from "active debugging." Re-reads
    are cheap; recovery from a starved debug loop is not."""
    state = ToolExecutionState()
    args = {"path": "src/app.py"}

    track_tool_result(
        tool_name="write_file",
        args={"path": "src/app.py", "content": "print('one')\n"},
        tool_result="Created src/app.py (1 lines)",
        round_num=1,
        state=state,
        dedup_stub=None,
    )
    track_tool_result(
        tool_name="read_file",
        args=args,
        tool_result="1\tprint('one')",
        round_num=2,
        state=state,
        dedup_stub=None,
    )

    stub = dedup_stub_for_tool("read_file", args, state)

    assert stub is None  # read_file dedup intentionally disabled


def test_repeated_failed_bash_cd_steers_with_concrete_cause() -> None:
    """Regression for the 2026-06-19 spin: a `cd` into a not-yet-created
    directory failed and the model re-emitted it 11× because the rejection
    blocked but never named the cause or escalated. The stub must (1) name
    the real error ("does NOT exist") and (2) hard-stop after the threshold.
    """
    state = ToolExecutionState()
    args = {
        "command": 'cd "/Users/x/localcode_test/Anki/Qwen 3.6 35B-A3B (Q8)" && ls',
    }
    err = (
        "[exit code 1] /bin/sh: line 0: cd: "
        "/Users/x/localcode_test/Anki/Qwen 3.6 35B-A3B (Q8): "
        "No such file or directory"
    )

    # First two real failures: stub should now name the concrete cause.
    for round_num in (1, 2):
        track_tool_result(
            tool_name="bash",
            args=args,
            tool_result=err,
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )
    stub = dedup_stub_for_tool("bash", args, state)
    assert stub is not None and stub.startswith("REJECTED")
    assert "does NOT exist" in stub  # concrete steering, not generic
    assert "HARD STOP" not in stub  # not terminal yet

    # Keep re-emitting; the model ignores the rejection (each REJECTED stub
    # counts as another failure, exactly like the live loop).
    for round_num in range(3, _HARD_STOP_THRESHOLD + 2):
        track_tool_result(
            tool_name="bash",
            args=args,
            tool_result=stub,  # the rejected stub IS the result, like prod
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )
        stub = dedup_stub_for_tool("bash", args, state)
        assert stub is not None

    assert "HARD STOP" in stub
    assert "permanently blocked" in stub
    # The underlying cause survives even though later results were REJECTED
    # stubs (which must not overwrite the real diagnosis).
    assert "does NOT exist" in stub


def test_whitespace_varying_bash_reemission_keeps_breaker_and_nudge_aligned() -> None:
    """Regression for F2: the bash breaker keys on the whitespace-normalized
    `bash_failures`, but the loop's steering nudge used to key on the raw
    `failed_calls` (tool, canonical_args). When the model re-emitted the same
    command with cosmetic whitespace differences each variant became a
    distinct `failed_calls` key, so the nudge undercounted and never fired
    while the breaker still hard-stopped — a block with no steering.

    The fix routes the nudge through `bash_cmd_key` too. This test pins the
    invariant: whitespace variants collapse to ONE bash_failures key (the
    nudge's source of truth) while splitting failed_calls.
    """
    state = ToolExecutionState()
    # Same logical command, re-emitted with cosmetic whitespace drift.
    variants = [
        "cd /tmp/nope && ls",
        "cd  /tmp/nope  &&  ls",
        "cd /tmp/nope &&  ls",
        "cd /tmp/nope\t&& ls",
    ]
    err = "[exit code 1] cd: /tmp/nope: No such file or directory"
    for round_num, cmd in enumerate(variants, start=1):
        track_tool_result(
            tool_name="bash",
            args={"command": cmd},
            tool_result=err,
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )

    # All variants share one normalized key — the count the nudge reads.
    assert len(state.bash_failures) == 1
    norm_key = bash_cmd_key({"command": variants[0]})
    assert state.bash_failures[norm_key] == len(variants)
    # The nudge now reads this >= _HARD_STOP_THRESHOLD count, matching the
    # breaker — even though raw failed_calls splintered into separate keys.
    assert state.bash_failures[norm_key] >= _HARD_STOP_THRESHOLD
    assert len({canonical_args({"command": c}) for c in variants}) == len(variants)
    assert all(v == 1 for v in state.failed_calls.values())

    # The breaker hard-stops on the normalized count for any variant.
    stub = dedup_stub_for_tool("bash", {"command": variants[-1]}, state)
    assert stub is not None and "HARD STOP" in stub


def test_repeated_failed_bash_syntax_error_steers_on_quoting() -> None:
    state = ToolExecutionState()
    args = {"command": "cd /Users/x/Qwen (Q8) && ls"}
    err = "[exit code 2] /bin/sh: -c: syntax error near unexpected token `('"
    for round_num in (1, 2):
        track_tool_result(
            tool_name="bash",
            args=args,
            tool_result=err,
            round_num=round_num,
            state=state,
            dedup_stub=None,
        )
    stub = dedup_stub_for_tool("bash", args, state)
    assert stub is not None and "SHELL SYNTAX" in stub
    assert "double quotes" in stub


def test_bash_rejects_shell_redirection_file_writes(tmp_path: Path) -> None:
    result = _redirect_shell_file_write(
        "cat > src/app.py <<'EOF'\nprint('hello')\nEOF",
        str(tmp_path),
    )
    assert result.startswith("REJECTED:")
    assert "write_file/edit_file/multi_edit" in result


def test_bash_allows_generated_pipeline_file_writes(tmp_path: Path) -> None:
    # `python generate.py > out.json` / `curl URL > out.json` / `python -c
    # "..." > out.json` are legitimate "compute or fetch, save in one
    # step" patterns. Forcing them through write_file costs a turn on
    # every data-import / scratch save. Only hand-typed file content
    # (cat > file / echo > file / tee file) is still rejected.
    for cmd in (
        "python generate.py > data/output.json",
        'python3 -c "import json; print(json.dumps({}))" > data/output.json',
        "curl -s https://example.com/seed.json > data/output.json",
    ):
        assert _redirect_shell_file_write(cmd, str(tmp_path)) == "", cmd


def test_build_app_goal_block_is_empty() -> None:
    # 2026-04-29: stripped the agentic goal block (Current goal /
    # Continue until complete / build_app stage guidance / sibling-
    # directory reminder). None of it measurably prevented the failure
    # modes it targeted on Qwen3.6 IQ2_M, and mainstream agentic CLIs
    # don't ship anything similar. Only the question-type protective
    # wording remains; build_app gets nothing.
    goal = infer_goal_state("build a small dashboard app")
    task = _new_app_task(goal.task_slug)
    task.current_stage = "scaffolding"

    assert build_task_goal_block("build a small dashboard app", goal, task) == ""


# (Removed: the path-placeholder detection/substitution feature — trying to
# regex every placeholder shape a user might write was task-specific bloat.)


class _BuildGoal:
    goal_type = "build_app"
    task_slug = "demo"


def test_target_grounding_block_pins_project_root(tmp_path: Path) -> None:
    block = build_target_grounding_block(tmp_path, _BuildGoal())
    assert str(tmp_path.resolve()) in block  # grounds the model to the project root
    # Non-build goals get no block (keeps the cached prefix unchanged).
    assert build_target_grounding_block(tmp_path, infer_goal_state("what is 2+2")) == ""


def test_incremental_milestones_guidance_present_for_build_app() -> None:
    block = build_incremental_milestones_block(_BuildGoal())
    low = block.lower()
    # Must steer toward incremental, verified milestones — not a whole app.
    assert "milestone" in low
    assert "scaffold" in low
    assert "one feature at a time" in low
    assert "build" in low and ("run" in low or "verify" in low)
    # Non-build goals get nothing (cached prefix unaffected).
    assert build_incremental_milestones_block(infer_goal_state("what is 2+2")) == ""


def test_quality_monitor_rejects_placeholder_url() -> None:
    state = TurnState(user_text="run it", goal_state=infer_goal_state("run it"))
    verdict = quality_monitor("Open http://localhost:[FRONTEND_PORT]", state)
    assert not verdict.ok
    assert verdict.reason == "fake-placeholder-url"


def test_quality_monitor_allows_evidenced_final_claims(tmp_path: Path) -> None:
    app_file = tmp_path / "app.js"
    app_file.write_text(
        "localStorage.setItem('progress', '1');\n"
        "const u = new SpeechSynthesisUtterance('hello');\n"
        "speechSynthesis.speak(u);\n"
    )
    state = TurnState(
        user_text="build an app with audio and saved progress",
        goal_state=infer_goal_state("build an app with audio and saved progress"),
        changed_files=[str(app_file)],
    )

    verdict = quality_monitor(
        "Built browser speech synthesis with localStorage persistence.",
        state,
    )

    assert verdict.ok


def test_greeting_question_is_not_treated_as_blocking() -> None:
    assert not is_focused_blocking_question("Hi! How can I help you today?")
    assert is_focused_blocking_question("Which package manager should I use?")


def test_launcher_ignores_docs_site_when_repo_has_no_app(tmp_path: Path) -> None:
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "index.html").write_text("<html>docs</html>")
    assert detect_launch_candidate(tmp_path) is None


def test_process_registry_round_trip(tmp_path: Path) -> None:
    record = ProcessRecord(
        pid=123,
        pgid=123,
        port=4567,
        url="http://localhost:4567",
        cwd=str(tmp_path),
        kind="test",
        command="serve",
        log_path="log",
        verified=False,
        started_at=1.0,
    )
    record_process(tmp_path, record)
    loaded = load_records(tmp_path)
    assert loaded[-1].port == 4567
    assert loaded[-1].command == "serve"


def test_process_registry_does_not_treat_alive_pid_as_healthy_url(tmp_path: Path) -> None:
    import os

    record_process(
        tmp_path,
        ProcessRecord(
            pid=os.getpid(),
            pgid=os.getpid(),
            port=9,
            url="http://localhost:9",
            cwd=str(tmp_path),
            kind="test",
            command="serve",
            log_path="log",
            verified=True,
            started_at=2.0,
        ),
    )
    assert latest_live_record(tmp_path) is None
    loaded = load_records(tmp_path)
    assert loaded[-1].verified is False


def test_process_summary_reports_registry_state(tmp_path: Path) -> None:
    record_process(
        tmp_path,
        ProcessRecord(
            pid=999999,
            pgid=999999,
            port=4321,
            url="http://localhost:4321",
            cwd=str(tmp_path),
            kind="test",
            command="serve",
            log_path="log",
            verified=False,
            started_at=3.0,
        ),
    )
    summary = process_summary(tmp_path)
    assert "LocalCode-managed processes:" in summary
    assert "port=4321" in summary


def test_dynamic_skill_injection_is_small_and_intent_based(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".localcode" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "run-tests.md").write_text(
        "---\n"
        "name: run-tests\n"
        "description: Run project tests.\n"
        "when_to_use: User asks to run tests.\n"
        "---\n"
        "Run the smallest relevant test command and report failures clearly.\n"
    )
    registry = load_registry(tmp_path)
    selected = select_dynamic_skills("run the tests", registry)
    block = dynamic_skill_block(selected)
    assert "run-tests" in block
    assert "Run the smallest relevant test command" in block


def test_recent_tools_do_not_select_stale_skills(tmp_path: Path) -> None:
    registry = SkillRegistry(
        skills={
            "run-tests": Skill(
                name="run-tests",
                description="Run tests.",
                when_to_use="User asks to run tests.",
                body="Run the smallest relevant test command.",
                source_path=tmp_path / "run-tests.md",
                origin="bundled",
            )
        }
    )
    selected = select_dynamic_skills(
        "refactor this project to rust",
        registry,
        recent_tools=["bash"],
    )
    assert selected == []


def test_read_file_default_caps_large_files(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_text("\n".join(f"line {i} " + ("x" * 120) for i in range(1000)))
    result = read_file(ToolContext(app=_App(tmp_path), out=_Out()), {"path": "large.txt"})
    assert "Large file summarized" in result
    assert len(result) < 13_000


def test_bash_does_not_background_probe_commands_that_mention_app_py() -> None:
    assert not _looks_like_detached_server_command("ps aux | grep -i 'flask\\|app.py'")
    assert not _looks_like_detached_server_command("curl -s http://localhost:5001/app.py")
    assert _looks_like_detached_server_command("cd demo && python3 app.py")


def test_bash_redirection_is_not_explicit_backgrounding() -> None:
    from localcode.tools.bash import _BG_RE, _masked_shell_failure

    assert _BG_RE.search("npx tsc --noEmit 2>&1 | head -60") is None
    assert _BG_RE.search("command 3<&0") is None
    assert _BG_RE.search("npm run dev &") is not None
    assert _masked_shell_failure("Error: Cannot find module 'canvas'\nMODULE_NOT_FOUND")
    assert not _masked_shell_failure("vite build completed successfully")


def test_successful_build_mutation_leaves_reasoning_heavy_scaffolding() -> None:
    from localcode.thinking import next_task_stage_after_tool, should_use_thinking

    assert should_use_thinking("", "auto", goal_type="build_app", task_stage="scaffolding")
    stage = next_task_stage_after_tool("scaffolding", "write_file", succeeded=True)
    assert stage == "running"
    assert not should_use_thinking("", "auto", goal_type="build_app", task_stage=stage)
    assert next_task_stage_after_tool("scaffolding", "read_file", succeeded=True) == "scaffolding"
    assert next_task_stage_after_tool("scaffolding", "bash", succeeded=False) == "scaffolding"


def test_bash_server_launch_surfaces_startup_crash(tmp_path: Path, monkeypatch) -> None:
    # A server command that exits during startup must report the FAILURE +
    # the captured error (so the agent fixes it), NOT a blind "it's running".
    # This is the dev-server class of failure that made the model give up.
    import localcode.tools.bash as b
    monkeypatch.setattr(b, "_looks_like_detached_server_command", lambda c: True)
    result = bash_execute(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"command": "sh -c 'echo boom-startup-error; exit 1'"},
    )
    assert "NOT running" in result
    assert "boom-startup-error" in result


def test_bash_server_launch_detects_printed_url(tmp_path: Path, monkeypatch) -> None:
    # A server that prints its URL (vite-style, possibly a hopped port) must
    # be surfaced with the REAL url — not a hardcoded/blind port (the 404 bug).
    import localcode.tools.bash as b
    monkeypatch.setattr(b, "_looks_like_detached_server_command", lambda c: True)
    try:
        result = bash_execute(
            ToolContext(app=_App(tmp_path), out=_Out()),
            {"command": "sh -c 'echo \"  Local:   http://localhost:5174/\"; sleep 20'"},
        )
        assert "http://localhost:5174" in result
        assert "UP at" in result
        assert "Do NOT relaunch" in result
    finally:
        b.reap_background_processes()


def test_bash_redirects_shell_file_reads_to_read_file(tmp_path: Path) -> None:
    target = tmp_path / "src" / "App.jsx"
    target.parent.mkdir()
    target.write_text("\n".join(f"line {i}" for i in range(120)))

    result = bash_execute(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"command": "cat -n src/App.jsx | sed -n '10,40p'"},
    )

    assert result.startswith("REJECTED:")
    assert "read_file" in result
    assert "path='src/App.jsx'" in result
    assert "offset=9" in result
    assert "limit=31" in result


def test_bash_redirects_shell_file_reads_after_cd(tmp_path: Path) -> None:
    target = tmp_path / "frontend" / "src" / "App.jsx"
    target.parent.mkdir(parents=True)
    target.write_text("\n".join(f"line {i}" for i in range(120)))

    result = bash_execute(
        ToolContext(app=_App(tmp_path), out=_Out()),
        {"command": "cd frontend && sed -n '1,60p' src/App.jsx"},
    )

    assert result.startswith("REJECTED:")
    assert "path='frontend/src/App.jsx'" in result
    assert "offset=0" in result
    assert "limit=60" in result


def test_new_app_writes_can_choose_natural_project_dir(tmp_path: Path) -> None:
    app = _App(tmp_path)
    app.session.current_task = _new_app_task("canonical-app")

    result = _execute_tool(
        app,
        "write_file",
        {"path": "random_app/app.py", "content": "print('x')\n"},
        _Out(),
    )

    assert result.startswith("Created")
    assert (tmp_path / "random_app" / "app.py").exists()


def test_new_app_bash_can_choose_natural_project_dir(tmp_path: Path) -> None:
    app = _App(tmp_path)
    app.session.current_task = _new_app_task("canonical-app")

    result = bash_execute(
        ToolContext(app=app, out=_Out()),
        {"command": "mkdir -p random_app/src"},
    )

    assert result == "all good!"
    assert (tmp_path / "random_app" / "src").exists()


def test_new_app_bash_allows_relative_paths_after_cd_into_canonical_dir(tmp_path: Path) -> None:
    app = _App(tmp_path)
    app.session.current_task = _new_app_task("canonical-app")
    (tmp_path / "canonical-app").mkdir()

    result = bash_execute(
        ToolContext(app=app, out=_Out()),
        {"command": f"cd {tmp_path / 'canonical-app'} && mkdir -p data"},
    )

    assert result == "all good!"


def test_new_app_defers_dependency_install_until_source_exists(tmp_path: Path) -> None:
    app = _App(tmp_path)
    app.session.current_task = _new_app_task("canonical-app")
    app_root = tmp_path / "canonical-app"
    app_root.mkdir()
    (app_root / "requirements.txt").write_text("flask\n")

    rejected = bash_execute(
        ToolContext(app=app, out=_Out()),
        {"command": "cd canonical-app && uv pip install flask"},
    )
    assert rejected.startswith("REJECTED:")
    assert "source files" in rejected

    (app_root / "app.py").write_text("print('ready')\n")
    accepted = bash_execute(
        ToolContext(app=app, out=_Out()),
        {"command": "cd canonical-app && python3 -c 'print(1)'"},
    )
    assert accepted == "1"


def test_tool_facts_extracts_launch_contract_fields() -> None:
    facts = extract_tool_facts(
        "launch_app",
        {"action": "start"},
        "App launched and verified.\nURL: http://localhost:4321\nPID: 99\nLog: /tmp/app.log",
    )
    assert facts["ok"] is True
    assert facts["verified"] is True
    assert facts["url"] == "http://localhost:4321"
    assert facts["port"] == 4321
    assert facts["pid"] == 99
    assert "verified=True" in facts_suffix(facts)


def test_tool_facts_extracts_error_type_and_verification_signal() -> None:
    error_facts = extract_tool_facts(
        "bash",
        {"command": "python app.py"},
        "[exit code 1]\nSyntaxError: invalid syntax",
    )
    assert error_facts["ok"] is False
    assert error_facts["error_type"] == "syntax_error"

    verified_facts = extract_tool_facts(
        "bash",
        {"command": "npm test"},
        "Tests passed\n",
    )
    assert verified_facts["ok"] is True
    assert verified_facts["verification_signal"] is True


def test_tool_facts_marks_missing_file_as_error() -> None:
    facts = extract_tool_facts(
        "read_file",
        {"path": "missing.json"},
        "File not found: missing.json",
    )
    assert facts["ok"] is False
    assert facts["error_type"] == "not_found"


def test_dispatch_result_returns_typed_tool_result(tmp_path: Path) -> None:
    result = dispatch_result("read_file", ToolContext(app=_App(tmp_path), out=_Out()), {})
    assert result.ok is False
    assert result.facts["missing_args"] == ["path"]
    assert "path" in result.text


def test_semantic_tool_summary_keeps_facts_neutralizes_old_errors() -> None:
    success = "body\n\n[tool facts: ok=True, url=http://localhost:1234, port=1234]"
    assert "port=1234" in _semantic_tool_summary(success)
    # OLD errors are neutralized (self-conditioning, arXiv:2509.09677): the
    # failure TEXT must not be echoed back — only a neutral note. Recent
    # errors (within keep_recent) are left intact by _compact_old_tool_results.
    error = "[exit code 1]\nline1\nTraceback details"
    summary = _semantic_tool_summary(error)
    assert "errored and was handled" in summary
    assert "Traceback" not in summary and "exit code" not in summary


def test_prepare_model_messages_microcompacts_large_turn_history() -> None:
    # Fixture must outweigh post-redaction byte budget. With
    # REDACT_KEEP_RECENT_WRITES=1, all but one write_file arg gets stubbed,
    # so we beef up per-turn bulk (and the surviving verbatim write) until
    # post-aging total still exceeds _microcompact_for_prompt_budget's
    # 36_000-byte threshold — that's the path under test.
    messages = [{"role": "system", "content": "system"}]
    for idx in range(40):
        messages.append({"role": "user", "content": f"request {idx}"})
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": (
                        '{"path":"src/file_%d.py","content":"%s"}'
                        % (idx, "print(1)\\n" * 2500)
                    ),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{idx}",
            "name": "write_file",
            "content": "Created src/file_%d.py\n%s" % (idx, "ok\n" * 2500),
        })

    compacted = _prepare_model_messages(messages)

    assert len(compacted) < len(messages)
    assert "Earlier context summarized" in compacted[1]["content"]
    assert compacted[-1]["role"] == "tool"


def test_ran_build_or_test_detects_builds_not_dev_server() -> None:
    from localcode.agent.hooks import ran_build_or_test
    # Build/typecheck/test commands = evidence the code was compiled.
    assert ran_build_or_test([("npm run build", "ok")])
    assert ran_build_or_test([("npx tsc --noEmit", "ok")])
    assert ran_build_or_test([("pytest -q", "ok"), ("ls", "")])
    assert ran_build_or_test([("cargo test", "ok")])
    # A dev-server launch or plain inspection is NOT a build.
    assert not ran_build_or_test([("npm run dev", "")])
    assert not ran_build_or_test([("ls -la", ""), ("cat x", "")])
    assert not ran_build_or_test([])
