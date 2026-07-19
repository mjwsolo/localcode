"""The agent turn engine.

Final T0.1 slice. `run_agent_loop` used to live in agent/__init__.py
and everyone imported it as `from localcode.agent import run_agent_loop`.
That import path still works — `agent/__init__.py` now re-exports this
function from `agent/loop.py` so external callers don't notice the move.

The function itself is the core of LocalCode: one call per user turn,
drives the model / tool-dispatch loop until either (a) the model emits
a final response with no tool calls, (b) MAX_ROUNDS is hit, or (c) a
stall-recovery cap is exhausted.

It depends on every other submodule in agent/:
  • constants — policy knobs, safety caps
  • prompts   — SYSTEM_PROMPT, REASONING_RULES
  • context   — _prepare_model_messages, _compact_messages, truncation
  • recovery  — detect_stall, nudge_for
  • helpers   — _execute_tool, _needs_confirmation, display helpers

If you're reading this to understand the agent behaviour, start with
`run_agent_loop` below and follow the imports into the submodules as
you hit them.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..app import LocalCodeApp
    from ..output import OutputManager


__all__ = ["run_agent_loop"]


from .constants import (
    MAX_ROUNDS,
    CHURN_FILE_WRITE_LIMIT,
    MAX_OUTPUT_TOKENS,
    MAX_THINKING_SECONDS,
    MAX_THINKING_CHARS,
    MAX_AGGREGATE_PER_TURN,
    CROSS_ROUND_REPEAT_LIMIT,
)
from .goal import GoalState, infer_goal_state
from .context import (
    _msg_bytes,
    _prepare_model_messages,
    _summarize_args,
    _truncate_result,
    build_progress_ledger,
)
from .helpers import (
    _execute_tool,
    _execute_tool_result,
    _needs_confirmation,
    _render_markdown,
    _brief_result,
    _grounded_file_summary,
    _tool_stage_label,
    _first_token,
)
from .prompts import (
    SYSTEM_PROMPT,
)
from .prompt_context import build_agent_system_prompt
from .recovery import (
    rewrite_hard_stop,
    StallMode,
    detect_stall,
    nudge_for,
    MAX_EMPTY_ROUND_RETRIES,
    ChurnMode,
    detect_churn,
    churn_nudge_for,
    command_token,
    detect_reject_reread_loop,
    reject_reread_nudge,
    todo_close_verification_suffix,
)
from .tool_orchestration import prefetch_parallel_tool_calls
from .streaming import finish_thinking_display, stream_model_round
from .tool_execution import (
    _HARD_STOP_THRESHOLD,
    ToolExecutionState,
    bash_cmd_key,
    canonical_args,
    dedup_stub_for_tool,
    oversize_stub_for_tool,
    repeat_stub_for_tool,
    tool_result_is_error,
    track_tool_result,
)
from .turn_finalization import finalize_turn, strip_ephemeral_nudges
from ..tools import schemas_for_goal
from ..tools.facts import extract_tool_facts, facts_suffix
from .hooks import (
    TurnState,
    after_tool as hook_after_tool,
    before_model as hook_before_model,
    before_turn as hook_before_turn,
    completion_gate as hook_completion_gate,
    quality_monitor as hook_quality_monitor,
    ran_build_or_test,
    _changed_code_files,
)
from .app_tasks import (
    extract_port,
    ground_run_or_launch_text,
    has_runtime_verification_signal,
    is_focused_blocking_question,
    looks_like_partial_handoff,
)

def run_agent_loop(
    app: "LocalCodeApp",
    user_text: str,
    composed_messages: list[dict],
    out: "OutputManager",
    *,
    system_prompt: str | None = None,
) -> str:
    """Model-driven agent loop.

    The model decides what to do via native Gemma 4 tool calls.
    We execute tools and feed results back until the model is done.

    Returns the final text response from the model.

    Parameters
    ----------
    system_prompt : optional
        The unformatted prompt template. When None (the default), the
        module-level `SYSTEM_PROMPT` from `agent/prompts.py` is used
        — which is what production and app.py always want. Eval and
        tests pass an alternative here (e.g. the rendered minimal-core
        variant) to exercise different prompts without monkey-patching
        module globals. The string still goes through `.format(cwd=...,
        project_instructions=..., network_status=..., skills_block=...,
        reasoning_rules=...)` so any caller-supplied
        override must expose the same placeholder slots (our variant
        renderers all produce strings that have already had these
        placeholders baked in to no-op values, so `.format()` is
        effectively an identity pass when there are no placeholders
        left — that's why the existing variant registry works here
        too).

    Telemetry: emits a `turn_start` lifecycle event so `tail -f
    lifecycle.log` shows the new code is actually loaded and a turn
    is in flight — without this, conversational turns (no tool calls,
    no compactable history) leave the log silent for the entire turn
    and the user can't tell whether the latest code is running.
    """
    # DI default — reach into module globals only if caller didn't
    # pass an override. Means app.py keeps calling without worrying
    # about prompts while eval can inject a variant per run.
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    # Clear any leftover cancel from a PRIOR turn. The flag is set by the
    # TUI when the user types "stop"; it's consumed at the round/tool
    # boundaries below. If a previous turn broke out via an exception
    # path between "set" and the TUI's own reset, the flag stayed True
    # and poisoned THIS turn — the loop saw cancel_requested on round 0
    # and bailed before a single model call. Resetting at entry makes
    # each turn start from a clean slate regardless of how the last one
    # ended.
    app.cancel_requested = False
    # Per-turn telemetry. Captures FULL user input so the events log
    # is a complete record of "what was asked + what was done." The
    # log is per-project, gitignored, never shipped to users — it's
    # for the dev team to replay and debug sessions later.
    import time as _time_mod
    import uuid as _uuid
    _turn_started_mono = _time_mod.monotonic()
    _turn_id = _uuid.uuid4().hex[:12]

    def _emit_internal_error(where: str, exc: BaseException) -> None:
        try:
            from ..events import emit as _emit_error
            _emit_error(
                "error",
                turn_id=_turn_id,
                where=where,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception:
            pass

    _active_goal_payload = getattr(app, "_active_goal_state", None)
    if isinstance(_active_goal_payload, dict):
        try:
            _payload = dict(_active_goal_payload)
            _payload["success_criteria"] = tuple(_payload.get("success_criteria") or ())
            _goal_state = GoalState(**_payload)
        except Exception as exc:
            _emit_internal_error("active_goal_state_parse", exc)
            _goal_state = infer_goal_state(user_text)
    else:
        _goal_state = infer_goal_state(user_text)
    _task_state = getattr(getattr(app, "session", None), "current_task", None)
    _last_announced_task_stage = str(getattr(_task_state, "current_stage", "") or "")

    def _announce_task_stage(stage: str, *, force: bool = False) -> None:
        nonlocal _last_announced_task_stage
        stage = (stage or "").strip()
        if not stage:
            return
        if not force and stage == _last_announced_task_stage:
            return
        _last_announced_task_stage = stage
        try:
            out.set_stage(f"task: {stage}")
        except Exception as exc:
            _emit_internal_error("announce_task_stage_output", exc)
        try:
            if hasattr(app, "store") and getattr(app, "session", None) is not None:
                app.store.update_task(
                    app.session,
                    status="in_progress",
                    current_stage=stage,
                )
        except Exception as exc:
            _emit_internal_error("announce_task_stage_store", exc)
    from ..thinking import should_use_thinking
    _mode = getattr(app.config.runtime, "laptop_26b_runtime_mode", "") or ""
    _reasoning = should_use_thinking(
        _mode,
        getattr(app.config.runtime, "internal_thinking_mode", "off"),
        goal_type=_goal_state.goal_type,
        task_stage=getattr(_task_state, "current_stage", "") if _task_state is not None else "",
        user_text=user_text,
    )
    try:
        from ..events import emit
        emit("turn_start",
             turn_id=_turn_id,
             task_id=getattr(_task_state, "task_id", ""),
             task_status=getattr(_task_state, "status", ""),
             task_stage=getattr(_task_state, "current_stage", ""),
             task_kind=getattr(_task_state, "task_kind", ""),
             task_slug=getattr(_task_state, "task_slug", ""),
             chars=len(user_text),
             text=user_text,                       # FULL input
             model=str(app.config.runtime.model or ""),
             runtime_mode=_mode,
             reasoning=_reasoning,
             goal_type=_goal_state.goal_type,
             goal_summary=_goal_state.goal_summary,
             success_criteria=_goal_state.success_criteria)
    except Exception as exc:
        _emit_internal_error("turn_start_emit", exc)

    # ── Build messages ──
    # composed_messages already has system prompt + context + full conversation + current user msg
    # Inject our tool-loop system prompt at the front
    from ..app import is_online
    online = is_online()
    if online:
        network_status = (
            "Network: ONLINE — you can download files, install packages, fetch URLs. "
            "PREFER your tools over writing from memory: web_search / web_fetch to "
            "confirm current facts, versions, and library docs; your skills and other "
            "tools for anything they cover. Don't hand-write what a tool can get right."
        )
    else:
        network_status = (
            "Network: OFFLINE — NO internet. Do NOT attempt downloads, pip install, curl, wget, or any network requests. "
            "Use only local files, already-installed packages, and your LOCAL skills and "
            "tools (read/grep/edit, installed CLIs). Generate sample/mock data locally "
            "instead of downloading."
        )
    def _current_task_stage_for_thinking() -> str:
        stage = str(_last_announced_task_stage or getattr(_task_state, "current_stage", "") or "").strip()
        if stage:
            return stage
        if _goal_state.goal_type == "build_app":
            return "planning"
        return ""

    use_thinking = _reasoning
    if _goal_state.goal_type == "build_app":
        _announce_task_stage(getattr(_task_state, "current_stage", "") or "planning", force=True)
    prompt_result = build_agent_system_prompt(
        app=app,
        user_text=user_text,
        goal_state=_goal_state,
        task_state=_task_state,
        base_system_prompt=system_prompt,
        network_status=network_status,
        use_thinking=use_thinking,
    )
    agent_system = prompt_result.system_prompt
    try:
        from ..events import emit as _emit_skills
        _emit_skills(
            "skill_selection",
            turn_id=_turn_id,
            candidates=prompt_result.skill_candidates,
            candidate_count=len(prompt_result.skill_candidates),
            selected=prompt_result.selected_skills,
            selected_count=len(prompt_result.selected_skills),
            selected_origins=prompt_result.selected_skill_origins,
            selected_chars=prompt_result.selected_skill_chars,
        )
        _emit_skills(
            "skill_injection",
            turn_id=_turn_id,
            skills=prompt_result.selected_skills,
            origins=prompt_result.selected_skill_origins,
            chars=prompt_result.selected_skill_chars,
            count=len(prompt_result.selected_skills),
        )
    except Exception:
        pass

    messages: list[dict[str, Any]] = []

    # Our agent prompt goes first — it's the most important
    messages.append({"role": "system", "content": agent_system})

    if _goal_state.goal_type == "question":
        # Question/diagnostic turns should not replay old tool protocol.
        # A one-character "?" after a failed build used to inherit the
        # previous tool chain and the model resumed building instead of
        # answering. Keep only recent conversational text; if evidence is
        # needed, the model can inspect logs/files explicitly.
        conversational = [
            m
            for m in composed_messages
            if m.get("role") in {"user", "assistant"}
            and not str(m.get("content", "")).lstrip().startswith("SYSTEM:")
        ]
        for m in conversational[-6:]:
            messages.append(m)
    elif use_thinking:
        # Thinking mode: keep context shorter but preserve recent assistant responses
        # Only drop very large blocks (repo structure dumps, huge tool results).
        # The "large block" threshold scales with the model's real context window
        # instead of a fixed 1500 chars — on a 256K window 1500 chars is ~1% of
        # the budget, so the old fixed value silently stripped tool results the
        # big machine had ample room to keep (and needs, to track its work).
        try:
            _tw = int(app.engine._target_num_ctx())
        except Exception:
            _tw = 0
        _big_block_chars = max(1500, int(_tw * 3.5 * 0.02)) if _tw else 1500
        for m in composed_messages:
            if m.get("role") == "system":
                continue
            content = str(m.get("content", ""))
            # Skip very large context blocks (repo structure, bulk retrieval results)
            # but keep normal assistant responses so model remembers recent conversation
            if len(content) > _big_block_chars and m.get("role") not in ("user", "assistant"):
                continue
            messages.append(m)
    else:
        for m in composed_messages:
            if m.get("role") == "system":
                continue  # skip old system prompt — ours is authoritative
            messages.append(m)

    # Add current user message if not already there
    if not messages or messages[-1].get("content") != user_text:
        messages.append({"role": "user", "content": user_text})

    # Attach any images pasted in the TUI to THIS turn's user message.
    # app.ask()/compose_messages is text-only, so the base64 images ride
    # on the app object (stashed by the chat screen). We rebuild the
    # matching user message here — where the loop owns the message list —
    # via the composer's OpenAI-compatible image_url parts. Cleared after
    # so images send exactly once, and only when the model can see them.
    _pending_imgs = getattr(app, "_pending_images", None)
    if _pending_imgs:
        app._pending_images = []
        _profile = getattr(app, "profile", None)
        if _profile is not None and getattr(_profile, "supports_vision", False):
            try:
                from ..composer import _build_user_message
                _provider = getattr(getattr(app, "config", None).runtime, "provider", "llama_cpp")
                for _i in range(len(messages) - 1, -1, -1):
                    if (messages[_i].get("role") == "user"
                            and messages[_i].get("content") == user_text):
                        messages[_i] = _build_user_message(
                            user_text, list(_pending_imgs), _profile, _provider
                        )
                        break
            except Exception:
                # Never let image plumbing break a normal text turn.
                pass

    full_response: list[str] = []
    start_time = time.time()
    tools_called: list[str] = []
    changed_files: list[str] = []
    bash_history: list[tuple[str, str]] = []
    _turn_prompt_tokens = 0
    _turn_completion_tokens = 0
    _turn_total_tokens = 0
    loop_detected = False
    _hook_state = TurnState(
        user_text=user_text,
        goal_state=_goal_state,
        task_state=_task_state,
        changed_files=changed_files,
        bash_history=bash_history,
        tools_called=tools_called,
    )
    hook_before_turn(_hook_state)
    # Per-turn tool-output budget (in chars) scales with the model's REAL
    # context window instead of a fixed cap: allow tool results to fill up to
    # ~35% of the window before further output this turn is truncated. Dynamic
    # per RAM/model — a 256K window keeps far more than a 64K one. A single
    # fixed cap starves big machines; the floor keeps small windows usable, and
    # within-turn overflow is still caught by the window-aware compaction pass
    # at the top of each round.
    try:
        _ctx_tokens_turn = int(app.engine._target_num_ctx())
    except Exception:
        _ctx_tokens_turn = 0
    _aggregate_budget = (
        max(MAX_AGGREGATE_PER_TURN, int(_ctx_tokens_turn * 3.5 * 0.35))
        if _ctx_tokens_turn
        else MAX_AGGREGATE_PER_TURN
    )
    _completion_gate_retries = 0
    _MAX_COMPLETION_GATE_RETRIES = 1
    # Open-todo completion discipline: don't let a long task end while the
    # model's own todo list still has open items. Bounded overall, plus a
    # diminishing-returns guard (stop nagging if the open-count stops falling).
    _todo_continue_count = 0
    _MAX_TODO_CONTINUATIONS = 15
    _todo_stuck_count = 0
    _last_todo_remaining = 10**9
    _edit_recovery_nudges = 0
    _MAX_CONSECUTIVE_CORRECTIONS = 2
    _generic_correction_nudges = 0
    # Fires at most once per turn when an identical failing call crosses the
    # hard-stop threshold — bypasses the soft correction cap above.
    _hard_stop_nudge_fired = False
    # Per-tool caps removed 2026-04-26 (commits ce8a714 → this one).
    # Telemetry showed `3-in-a-row` exact-repeat guard fired 0 times
    # ever in observed sessions; `same-tool > 10` fired once and it
    # was a false-positive on legitimate iterative data analysis;
    # `file-edit > 3` never appeared. Keeping them was paying
    # bookkeeping cost on every tool call to catch nothing real, and
    # cutting legitimate work when they did fire. Matches agent /
    # terminal coding tools pattern: rely on user Ctrl+C plus the targeted
    # surviving guards (thinking caps, empty-round nudge,
    # investigation-spin / looks-fine nudges) for loop termination.

    # Per-turn counter of auto-recovered empty rounds. Caps how many
    # times we'll nudge a stream that ended with just thinking (no
    # content, no tool calls) before giving up and ending the turn
    # with a visible message. Recovery logic + nudge text live in
    # agent/recovery.py (T0.1-d split).
    _empty_rounds_this_turn = 0
    _MAX_EMPTY_ROUND_RETRIES = MAX_EMPTY_ROUND_RETRIES
    _last_round_signature: tuple[int, str] | None = None
    _same_round_signature_count = 0
    _same_round_synthetic_rejections = 0
    _MUTATING_TOOLS = frozenset({
        "write_file",
        "append_file",
        "edit_file",
        "multi_edit",
        "edit_diff",
    })

    # ── Investigation-spin detector ──
    # The 20-round "let me check the CSS / let me check the JS / everything
    # looks fine / let me check…" pathology, observed 2026-04-26 on a
    # browser-rendering bug the agent could never see (no DOM access, no
    # screenshots). The existing loop-breakers don't trip:
    #   - identical-3-in-a-row guard: each read_file is a different path
    #   - same-tool > N: read_file count was 7, under the 10 cap
    # …so the agent kept reading until MAX_ROUNDS hit at round 20.
    #
    # New detector: count rounds where (a) only read-only tools were used
    # AND (b) content matched a "looks fine"-shaped phrase. After 3 such
    # rounds in a row (or 5 read-only rounds total, "looks fine" or not),
    # inject ONE synthetic user message telling the model to either
    # commit to a concrete change or ask the user a focused question, and
    # forbid further read-only investigation this turn. One nudge per
    # turn — if the model ignores it, the next round's natural exit
    # path applies.
    _READONLY_TOOLS = frozenset({
        "read_file", "grep", "glob", "list_files", "web_fetch", "web_search",
    })
    _LOOKS_FINE_RE = re.compile(
        r"\b(?:looks?\s+(?:fine|correct|good|ok)|seems?\s+(?:fine|correct|good)"
        r"|works?\s+fine|(?:load|run|work|render)(?:s|ing)?\s+(?:fine|correctly)"
        r"|checks?\s+out|all\s+(?:good|correct|fine)"
        r"|everything\s+(?:looks?|seems?)\s+(?:fine|correct|good|ok))\b",
        re.IGNORECASE,
    )
    _readonly_streak = 0
    _looks_fine_streak = 0
    _spin_nudge_done = False
    # Why did the loop exit? Set right before each `break` so the
    # turn_end event records the actual exit path. Without this, we
    # had to guess from indirect signals (last round_end finish_reason,
    # tool counts, etc.) and there was no way to tell whether a stall
    # guard, exception, or user cancel ended it.
    _loop_exit_reason: str = ""

    # Read-dedup state: maps path → round_idx of last read so we can
    # detect "model is re-reading the same file with no edit between."
    # Modified-set tracks any path the model wrote / edited / multi-
    # edited this turn — that invalidates a prior read (file changed).
    # Both reset per turn (this loop body is per-turn).
    _tool_exec_state = ToolExecutionState(changed_files=changed_files)
    _read_file_chars_this_turn = 0
    _edit_failures_this_turn = 0
    _write_existing_rejections_this_turn = 0
    _edit_context_seen = False
    # ── Semantic-churn counters (see recovery.detect_churn) ──
    # Per-turn signals the byte-identical breakers miss:
    #   _file_write_counts   path → # write/edit/append calls (any content)
    #   _command_fail_counts cmd-family-token → # failed runs
    # _readonly_streak (defined above) is reused as the spin signal.
    # _churn_nudge_done caps the churn nudge to ONE per turn so we don't
    # pile SYSTEM messages on a model that's already mid-recovery.
    _file_write_counts: dict[str, int] = {}
    _command_fail_counts: dict[str, int] = {}
    _churn_nudge_done = False
    _last_churn_mode = ""
    # Planning-without-progress streak (recovery.detect_churn PLANNING_SPIN):
    # consecutive rounds that changed no NEW file, ran no build/verify, and
    # produced thinking/narration. Catches a model re-deriving the same plan
    # round after round — which readonly-spin misses, because think-only
    # rounds (zero tools) reset the read-only streak. Resets to 0 the moment
    # a round changes a file or runs a build.
    _planning_streak = 0
    # Build-verification STOP gate: a true completion gate (claude-code
    # query.ts stop-hook pattern). When the model tries to END a build_app turn
    # that changed code, we run the project's real typecheck/test and, if it
    # reports errors, inject them and FORCE another round instead of accepting
    # "done". Bounded by _MAX_BUILD_VERIFY_RETRIES so a genuinely unfixable
    # project can't spin forever; each retry RE-RUNS the check so the gate keeps
    # holding until the project is clean or the bound is hit.
    _build_verify_nudges = 0
    _MAX_BUILD_VERIFY_RETRIES = 2
    # Reject → re-read → reject spin: ONE actionable redirect per turn
    # (recovery.detect_reject_reread_loop, recomputed from the transcript).
    _reject_reread_nudge_done = False
    # Last synthetic-nudge kind we injected, so the recovery ladder never fires
    # the SAME nag twice in a row (claude-code attachments.ts two-and-threshold
    # discipline). Routed through `_append_nudge` below.
    _last_nudge_kind = ""

    def _append_nudge(content: str, kind: str, *, ephemeral: bool = True) -> bool:
        """Append a synthetic SYSTEM nudge unless the SAME `kind` was the
        immediately-previous nudge (never the same nag twice running — the
        two-and-threshold discipline). Records the injected index for
        end-of-turn ephemeral cleanup. Returns True when it actually appended.
        """
        nonlocal _last_nudge_kind
        if kind and kind == _last_nudge_kind:
            return False
        messages.append({"role": "user", "content": content})
        if ephemeral:
            _ephemeral_nudge_indices.append(len(messages) - 1)
        _last_nudge_kind = kind
        return True
    # Cross-round repeated-call breaker: count identical (tool, args) calls
    # across the WHOLE turn (the in-round breaker only catches one round).
    # NUDGE-only — never withholds a tool result (avoids the read-dedup-stub
    # starvation regression). One nudge per turn.
    _turn_call_sigs: dict[tuple, int] = {}
    _xround_repeat_nudge_done = False
    # Generalized dedup for cacheable read-only investigations:
    # list_files / glob / grep with identical args within a turn return
    # the same content (modulo external file-system changes we can't
    # observe). Key is (tool_name, canonical_args_json) → round_num so
    # the dedup-stub can name the round the model already ran this. Any
    # write/edit invalidates the WHOLE map — cheap, safe, less granular
    # than per-path tracking but list_files/glob/grep can target
    # directories whose contents change after a write.
    # Indices of nudge messages we inject into `messages` so we can
    # strip them BEFORE this turn returns. Without this, every nudge
    # ("STOP investigating, do NOT call read_file…", "your last
    # response had no action — try again") becomes a PERMANENT
    # SYSTEM-prefixed user message in the conversation history, and
    # subsequent turns inherit those instructions. Real failure mode
    # observed 2026-04-26: a session accumulated 6+ spin nudges and
    # 4+ stall nudges, then "continue" produced 540 chars of "Now let
    # me update X" with ZERO tool calls — the model was obeying the
    # accumulated "do not call read tools" instructions, not the
    # current user intent. Ephemeral cleanup at end-of-turn.
    _ephemeral_nudge_indices: list[int] = []
    # The PURE read-only streak threshold moved to
    # constants.CHURN_READONLY_STREAK_LIMIT (6) and is enforced by the
    # semantic-churn detector — tighter than the old local value of 10,
    # which let an investigation spin run 5+ minutes before tripping.
    # `_looks_fine_streak` keeps its own threshold here because it's a
    # higher-confidence, distinctly-framed signal ("X looks fine, let me
    # check Y" repeated) that warrants its specific runtime-bug nudge.
    _MAX_LOOKS_FINE_STREAK = 3
    _deterministic_launch_done = False
    _verified_launch_summary = ""
    _observed_ttft_ms = 0

    # Open todos mean the model is mid-BUILD. It may start a preview dev server,
    # but a running server is NOT task completion while the plan still has
    # pending/in_progress items — taking the deterministic launch-and-finish
    # fast-path here is exactly how the Anki build "completed" with 4 todos
    # still open ("The app is now running" → done, at ~10% of the work). Only
    # take the launcher fast-path when nothing is left on the list; otherwise
    # fall through to normal agentic rounds and let the open-todo completion
    # gate keep the model building until every item is done.
    _open_todos_at_launch = [
        t for t in list(getattr(getattr(app, "session", None), "todos", []) or [])
        if str(t.get("status", "")).lower() != "completed"
    ]
    if _goal_state.goal_type == "run_or_launch" and not _open_todos_at_launch:
        try:
            from ..launcher import launch_project_app
            preferred_port = int(getattr(_task_state, "active_port", 0) or 0)
            launch = launch_project_app(
                app.repo_root,
                preferred_port=preferred_port,
                open_browser=True,
            )
            if launch.port and hasattr(app, "store") and getattr(app, "session", None) is not None:
                app.store.update_task(
                    app.session,
                    active_port=launch.port,
                    current_stage="verified" if launch.verified else "running",
                )
            try:
                from ..events import emit as _emit_launch
                _emit_launch(
                    "launcher_result",
                    turn_id=_turn_id,
                    ok=launch.ok,
                    verified=launch.verified,
                    kind=launch.kind,
                    root=launch.root,
                    command=launch.command,
                    port=launch.port,
                    url=launch.url,
                    log_path=launch.log_path,
                    message=launch.message,
                    browser_opened=launch.browser_opened,
                    browser_error=launch.browser_error,
                )
            except Exception:
                pass
            if launch.ok:
                summary = (
                    f"The app is running and verified.\n\n"
                    f"URL: {launch.url}\n"
                    f"Command: `{launch.command}`\n"
                    f"Log: `{launch.log_path}`\n"
                    f"Browser opened: {'yes' if launch.browser_opened else 'no'}"
                    + (f"\nBrowser error: {launch.browser_error}" if launch.browser_error else "")
                )
                _render_markdown(summary, app.console if hasattr(app, 'console') else None)
                full_response.append(summary)
                _loop_exit_reason = "verified_run_or_launch"
                _deterministic_launch_done = True
                _verified_launch_summary = summary
            messages.append({
                "role": "user",
                "content": (
                    "SYSTEM: Deterministic launcher could not complete this run_or_launch turn.\n"
                    f"Launcher result: {launch.message}\n"
                    "Inspect the project minimally, then run exactly one explicit launch command, "
                    "verify with one probe, and report the concrete URL. Do not loop on repeated "
                    "npm start / package edits."
                ),
            })
            _ephemeral_nudge_indices.append(len(messages) - 1)
        except Exception as _launcher_err:
            try:
                from ..events import emit as _emit_launch_err
                _emit_launch_err("launcher_error", turn_id=_turn_id, error=str(_launcher_err))
            except Exception:
                pass

    # ── Main loop ──
    # MAX_ROUNDS=0 → no hard cap (agent pattern). The
    # targeted safety nets above catch real failure modes; round
    # counting was emergency insurance with no observed claims.
    # `itertools.count()` gives an unbounded loop; positive
    # MAX_ROUNDS still works for eval / batch use cases.
    import itertools as _itertools
    _round_iter = (
        () if _deterministic_launch_done else
        _itertools.count() if MAX_ROUNDS <= 0 else range(MAX_ROUNDS)
    )
    for round_num in _round_iter:
        # Cancel check — user typed "stop" while the turn was running.
        # Poll at round boundary so an in-flight stream can finish rather
        # than leaving a partial corrupt response in history.
        if getattr(app, "cancel_requested", False):
            out.print_info("Stopped by user.")
            _loop_exit_reason = "user_cancel"
            break

        # Signal TUI that we're starting generation.
        out.start_thinking()

        # Conversation compaction: if the running prompt is about to
        # crowd out the context window, mutate `messages` in place with
        # a summarized version before the next model call. See
        # src/localcode/compaction.py for the minimal-agent-inspired design.
        try:
            from ..compaction import should_compact, compact, estimate_tokens
            # Authoritative source: ask the runtime gateway what context
            # size the server was actually launched with. The previous
            # `getattr(..., "llama_cpp_ctx_size", 0) or 32768` fallback
            # was wrong on every modern setup — turbo mode on a 16 GB
            # Mac runs at 65536 (see runtime.py `_target_num_ctx`),
            # which meant `should_compact` triggered at ~50% of real
            # utilization and we summarized the conversation HALF AS
            # OFTEN AS NEEDED, NO — TWICE AS OFTEN AS NEEDED, throwing
            # away useful recent history. Read the real value.
            try:
                ctx_tokens = int(app.engine._target_num_ctx())
            except Exception:
                ctx_tokens = (
                    int(getattr(app.config.runtime, "llama_cpp_ctx_size", 0))
                    or 65536
                )
            if should_compact(messages, context_window=ctx_tokens):
                # RAM-tier the summary: capable machines spend a model
                # generation for a rich summary; small machines fall back
                # to an instant deterministic one (compact() decides).
                try:
                    _ram_gb = int(app.engine._system_ram_gb())
                except Exception:
                    _ram_gb = None
                out.print_info(
                    f"Compacting conversation (≈{estimate_tokens(messages)} tokens "
                    f"of {ctx_tokens} context → summary)..."
                )
                _before_compact_count = len(messages)
                messages[:] = compact(
                    messages, app.engine, context_window=ctx_tokens, ram_gb=_ram_gb
                )
                try:
                    if getattr(app, "hooks", None) is not None:
                        app.hooks.on_post_compaction(_before_compact_count, len(messages))
                except Exception:
                    pass
        except Exception as _compact_err:
            # Never let compaction failure kill the agent loop — continue
            # with the unchanged messages and let the user see the error.
            out.print_info(f"(compaction skipped: {_compact_err})")

        # Per-round diagnostic state. Populated as the stream runs;
        # flushed via `round_end` after the round terminates so we can
        # correlate "what the model did" with "what we then did" without
        # guessing. Without this, debugging an agent-loop misbehaviour
        # means re-running with `--debug` and watching live — these
        # fields make the events.jsonl trace self-contained.
        _round_finish_reason = ""
        _round_raw_tail = ""
        _round_content_chars = 0
        _round_reasoning_chars = 0
        _round_pending_tool_count = 0
        # Per-round timing breakdown so events.jsonl can answer
        # "where does the time go?" without guessing. Three buckets:
        #   ttft_ms      — request-start → first token (prompt eval)
        #   decode_ms    — first token → end of stream (token gen)
        #   tool_exec_ms — sum of wall-time for tools in this round
        # Sum of these ≤ duration_ms; the remainder is bridge / agent-
        # loop overhead. Captured from stream_done + the tool dispatch
        # loop below.
        _round_ttft_ms = 0
        _round_decode_ms = 0
        _round_tool_exec_ms = 0
        _round_started_at = time.monotonic()
        # Snapshots for the planning-without-progress signal: how many files
        # were changed and how many bash commands had run BEFORE this round.
        # Compared after the round's tools run to decide if this round made
        # concrete progress (new file / build) or was pure (re-)planning.
        _changed_files_at_round_start = len(changed_files)
        _bash_history_at_round_start = len(bash_history)
        round_task_stage = _current_task_stage_for_thinking()
        round_use_thinking = should_use_thinking(
            app.config.runtime.laptop_26b_runtime_mode,
            app.config.runtime.internal_thinking_mode,
            goal_type=_goal_state.goal_type,
            task_stage=round_task_stage,
            user_text=user_text,
        )
        try:
            _ctx_chars = None
            try:
                _nctx = app.engine._target_num_ctx()
                if _nctx:
                    _ctx_chars = int(_nctx * 3.5)  # tokens → ~chars
            except Exception:
                _ctx_chars = None
            # ── Append-only transcript between DISCRETE compactions ──
            # (codex/opencode/pi/claude-code pattern). The old behavior ran the
            # full shrink pass EVERY round with a moving age boundary: each
            # round another message crossed "old" and got stubbed, changing
            # bytes near the START of the prompt, so llama.cpp's prefix cache
            # missed and the ENTIRE conversation re-prefilled every round. On a
            # measured 55-min build that was 77% of wall-clock (TTFT grew from
            # 3 s at round 3 to 95 s at round 53 — pure re-prefill). All four
            # reference harnesses append-only within a turn and compact as a
            # discrete event; on a LOCAL model, where prefill is the dominant
            # cost, the prefix cache is the single biggest speed lever.
            #
            # Trigger is dynamic (window-derived, RAM-scaled upstream): compact
            # when the serialized transcript passes ~55% of the context window
            # (headroom for the trailing ephemeral block + generation). The
            # compaction is DURABLE (messages[:] = shrunk) so subsequent rounds
            # are again byte-stable prefixes — one cache miss per compaction,
            # zero per round.
            _trigger_bytes = int((_ctx_chars or 250_000) * 0.55)
            if _msg_bytes(messages) > _trigger_bytes:
                # Mid-turn nudges are per-round steering the model has already
                # consumed; their stored INDICES are only valid on the
                # uncompacted list, so strip them first and reset bookkeeping.
                if _ephemeral_nudge_indices:
                    strip_ephemeral_nudges(messages, _ephemeral_nudge_indices)
                    _ephemeral_nudge_indices.clear()
                messages[:] = _prepare_model_messages(
                    messages, ctx_window_chars=_ctx_chars,
                    observed_ttft_ms=_observed_ttft_ms,
                )
                try:
                    from ..events import emit as _emit_compact
                    _emit_compact("discrete_compaction", turn_id=_turn_id,
                                  round_idx=round_num,
                                  bytes_after=_msg_bytes(messages),
                                  trigger_bytes=_trigger_bytes)
                except Exception:
                    pass
            model_messages = hook_before_model(list(messages), _hook_state)
            # Inject a window-scaled progress ledger (Codex-style tool-state
            # awareness): the model always sees what it has already done, so it
            # stops re-reading files / restarting from scratch. Built from
            # DURABLE loop state (not the compacted messages), so it survives
            # compaction — which is exactly what a small-RAM machine needs.
            # Appended LAST (after the cached history) to minimize prefix-cache
            # invalidation; ephemeral (model_messages is a per-round copy).
            try:
                from ..model_config import progress_ledger_budget_chars
                _win_tokens = int((_ctx_chars or 0) / 3.5)
                _ledger = build_progress_ledger(
                    changed_files,
                    bash_history,
                    list(getattr(_tool_exec_state, "files_read", {}) or {}),
                    progress_ledger_budget_chars(_win_tokens),
                )
            except Exception:
                _ledger = ""
            # Working-memory checklist (todo_write): model sees its own plan.
            _todo_note = ""
            try:
                from ..tools.todo_write import render_todo_reminder
                _todo_note = render_todo_reminder(list(getattr(app.session, "todos", []) or []))
            except Exception:
                _todo_note = ""
            # Filesystem reconciliation (pi/codex/opencode pattern): the actual
            # files on disk, scanned in CODE each round. Ground truth the model
            # can't forget or hallucinate — kills the "re-check what exists /
            # restart" loop that memory-based tracking couldn't.
            _fs_state = ""
            try:
                from .context import build_filesystem_state
                _fs_state = build_filesystem_state(changed_files)
            except Exception:
                _fs_state = ""
            # Inject ledger + fs-state + todo as ONE trailing SYSTEM message (not
            # role:user — as a user turn the model re-greeted every round). Ephemeral.
            try:
                _ctx_block = "\n\n".join(b for b in (_ledger, _fs_state, _todo_note) if b)
                if _ctx_block:
                    model_messages = list(model_messages) + [
                        {"role": "system", "content": _ctx_block}
                    ]
            except Exception:
                pass
            round_tool_schemas = schemas_for_goal(
                _goal_state.goal_type,
                user_text,
                task_stage=round_task_stage,
            )
            try:
                app._tool_content_max_chars = None
            except Exception:
                pass
            round_tool_names = [
                str(((schema.get("function") or {}).get("name") or ""))
                for schema in round_tool_schemas
            ]
            # Pre-stream snapshot of what we're sending. Char count +
            # message count are cheap and tell us if the prompt was
            # truncated by upstream compaction in a way that would
            # explain a cold response.
            try:
                from ..events import emit as _emit_round
                _emit_round(
                    "round_start",
                    turn_id=_turn_id,
                    round_idx=round_num,
                    messages_count=len(model_messages),
                    prompt_chars=sum(
                        len(str(m.get("content") or "")) for m in model_messages
                    ),
                    use_thinking=bool(round_use_thinking),
                    max_output_tokens=int(MAX_OUTPUT_TOKENS),
                    tool_schema_count=len(round_tool_schemas),
                    tool_schemas=round_tool_names,
                )
            except Exception:
                pass
            _stream_result = stream_model_round(
                app,
                out,
                model_messages,
                round_use_thinking=round_use_thinking,
                retry_messages=messages,
                tool_schemas=round_tool_schemas,
                recovery_mode="",
                stream_policy=_goal_state.goal_type,
            )
            _round_finish_reason = _stream_result.finish_reason
            _round_raw_tail = _stream_result.raw_tail
            _round_content_chars = _stream_result.content_chars
            _round_reasoning_chars = _stream_result.reasoning_chars
            _round_pending_tool_count = _stream_result.pending_tool_count
            _round_ttft_ms = _stream_result.ttft_ms
            _observed_ttft_ms = _round_ttft_ms
            _round_decode_ms = _stream_result.decode_ms
            _stream_tool_calls = _stream_result.tool_calls
            _primary_round_tool = (
                getattr(_stream_result, "limited_tool_name", "") or ""
            )
            if not _primary_round_tool and _stream_tool_calls:
                try:
                    _primary_round_tool = str(
                        ((_stream_tool_calls[0].get("function") or {}).get("name") or "")
                    )
                except Exception:
                    _primary_round_tool = ""
            _round_signature = (_round_content_chars, _primary_round_tool)
            if _primary_round_tool and _round_signature == _last_round_signature:
                _same_round_signature_count += 1
            else:
                _last_round_signature = _round_signature
                _same_round_signature_count = 1 if _primary_round_tool else 0
            _turn_prompt_tokens += int(_stream_result.prompt_tokens or 0)
            _turn_completion_tokens += int(_stream_result.completion_tokens or 0)
            _turn_total_tokens += int(
                _stream_result.total_tokens
                or ((_stream_result.prompt_tokens or 0) + (_stream_result.completion_tokens or 0))
            )
        except KeyboardInterrupt:
            out.print_info("Interrupted.")
            _loop_exit_reason = "stream_interrupt"
            break
        except Exception as exc:
            from ..errors import format_for_user
            out.set_error(format_for_user(exc, fallback_code="E3102"))
            _loop_exit_reason = f"stream_error:{type(exc).__name__}"
            break

        # If we bailed on a stuck thinking loop, tell the user explicitly so
        # they know why the turn ended and can try a different prompt.
        if _stream_result.thinking_abort:
            out.notice(
                f"Stopped: model reasoning exceeded the per-round cap "
                f"({MAX_THINKING_SECONDS}s or {MAX_THINKING_CHARS} chars) without "
                f"emitting a response. Turn off deep reasoning with `/thinking off`, "
                f"switch to a faster model with `/model`, or rephrase the task in "
                f"smaller steps."
            )

        # Show thinking summary if present (collapsed, dim)
        # Skip if already emitted when content started streaming
        finish_thinking_display(_stream_result, out)

        content = _stream_result.content
        tool_calls = _stream_result.tool_calls

        # Recover tool calls the model emitted but the server failed to parse.
        # Quantized Qwen quants sometimes emit `<tool_call><function=…>` (XML)
        # or `<tool_call>{…}` (JSON) shapes --jinja doesn't recognize, so they
        # arrive as plain CONTENT with no structured tool_calls: the call never
        # runs (wasted turn) and leaked <think> reasoning makes the model narrate
        # "the user wants…". Parse them out of content, execute them, and strip
        # the raw XML + leaked thinking from what's stored/shown.
        if content and ("<tool_call>" in content or "</think>" in content):
            try:
                from ..tools.tool_call_repair import repair_tool_calls
                _cleaned, _recovered = repair_tool_calls(content)
                if _recovered and not tool_calls:
                    tool_calls = _recovered
                    _stream_result.tool_calls = _recovered
                if _cleaned != content:
                    content = _cleaned
            except Exception:
                pass

        # Clear the indicator before rendering output
        out._stop_indicator()
        sys.stdout.write("\r\033[K")  # clear indicator line
        sys.stdout.flush()

        # Build assistant message for history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)
        _assistant_msg_idx = len(messages) - 1

        # ── Round-end telemetry ──
        # Single self-contained record of what THIS round did. Has every
        # field needed to attribute "model bailed mid-turn" failures
        # (raw_tail + finish_reason + parse_markers_seen) without
        # re-running with --debug. Lives in events.jsonl bucket=round.
        try:
            from ..events import emit as _emit_round_end
            _round_total_ms = int((time.monotonic() - _round_started_at) * 1000)
            _emit_round_end(
                "round_end",
                round_idx=round_num,
                duration_ms=_round_total_ms,
                # Latency breakdown so optimization decisions can be
                # data-driven instead of guessing. Sum of these three
                # ≤ duration_ms; the residual (overhead_ms) is bridge,
                # event dispatch, dedup checks, history mutation, etc.
                ttft_ms=_round_ttft_ms,
                decode_ms=_round_decode_ms,
                tool_exec_ms=_round_tool_exec_ms,
                overhead_ms=max(
                    0,
                    _round_total_ms - _round_ttft_ms - _round_decode_ms - _round_tool_exec_ms,
                ),
                finish_reason=_round_finish_reason,
                content_chars=_round_content_chars,
                reasoning_chars=_round_reasoning_chars,
                tool_count=len(tool_calls) if tool_calls else 0,
                tool_names=[
                    (tc.get("function") or {}).get("name", "")
                    for tc in (tool_calls or [])
                ][:8],
                pending_tool_count=_round_pending_tool_count,
                prompt_tokens=int(_stream_result.prompt_tokens or 0),
                completion_tokens=int(_stream_result.completion_tokens or 0),
                total_tokens=int(
                    _stream_result.total_tokens
                    or ((_stream_result.prompt_tokens or 0) + (_stream_result.completion_tokens or 0))
                ),
                usage_estimated=bool(getattr(_stream_result, "usage_estimated", False)),
                tool_args_limited=bool(getattr(_stream_result, "tool_args_limited", False)),
                limited_tool_name=getattr(_stream_result, "limited_tool_name", ""),
                limited_args_chars=int(getattr(_stream_result, "limited_args_chars", 0) or 0),
                # Tail of raw content stream — captured ONLY when the
                # round looks suspect (content but no tools, or zero
                # output entirely) so we don't bloat normal logs.
                raw_tail=(
                    _round_raw_tail[-500:]
                    if (not tool_calls and (content or _round_pending_tool_count == 0))
                    else ""
                ),
                content_tail=content[-200:] if content else "",
                # ── Churn snapshot ──
                # Cumulative semantic-churn signals for the turn so we can
                # SEE thrashing in events.jsonl instead of inferring it.
                # Captured BEFORE this round's tools run, so values reflect
                # rounds 0..round_num-1; the next round_end includes this
                # round's calls. `max_file_writes` = the worst single path's
                # write count; `file_write_counts` = the full per-path map
                # (only paths written >1×, to keep the record small);
                # `command_fail_counts` = per-command-family failure counts;
                # `readonly_streak` = current pure-investigation run length;
                # `churn_nudge` = which churn nudge has fired this turn (if any).
                max_file_writes=(max(_file_write_counts.values()) if _file_write_counts else 0),
                file_write_counts={
                    p: n for p, n in _file_write_counts.items() if n > 1
                },
                command_fail_counts=dict(_command_fail_counts),
                readonly_streak=_readonly_streak,
                planning_streak=_planning_streak,
                churn_nudge=_last_churn_mode,
            )
        except Exception:
            pass

        if _same_round_signature_count >= 5:
            try:
                from ..events import emit as _emit_same_signature_loop

                _emit_same_signature_loop(
                    "auto_nudge",
                    signal="same_round_signature_loop",
                    reason="same_content_chars_and_tool_repeated",
                    repeat_count=_same_round_signature_count,
                    content_chars=_round_content_chars,
                    tool_name=_primary_round_tool,
                    finish_reason=_round_finish_reason,
                )
            except Exception:
                pass
            if tool_calls:
                # This detector runs before normal tool dispatch. Previously it
                # only appended a user nudge and continued, so no tool_result was
                # ever produced and the failure ledger never advanced. Feed a
                # synthetic failed tool result into the conversation instead; the
                # next round sees a normal recoverable tool failure and the
                # schema selector can remove the repeated tool.
                _same_round_synthetic_rejections += 1
                for _tc in tool_calls:
                    _fn = _tc.get("function") or {}
                    _name = str(_fn.get("name") or _primary_round_tool or "tool")
                    _args = _fn.get("arguments") or {}
                    if isinstance(_args, str):
                        try:
                            _args_for_key = json.loads(_args)
                        except Exception:
                            _args_for_key = {"arguments": _args[:500]}
                    elif isinstance(_args, dict):
                        _args_for_key = _args
                    else:
                        _args_for_key = {"arguments": str(_args)[:500]}
                    _tool_exec_state.failed_calls[(_name, canonical_args(_args_for_key))] = (
                        _tool_exec_state.failed_calls.get((_name, canonical_args(_args_for_key)), 0) + 1
                    )
                    messages.append({
                        "role": "tool",
                        "content": (
                            "REJECTED: repeated identical tool pattern. "
                            "Use a materially different tool or arguments next; "
                            "do not retry this exact action."
                        ),
                        "tool_call_id": _tc.get("id", ""),
                    })
                if _same_round_synthetic_rejections >= 3:
                    messages.append({
                        "role": "user",
                        "content": (
                            "SYSTEM: The last repeated action was rejected several "
                            "times. Change strategy now: inspect current files, use "
                            "a different write/edit path, or verify what exists. "
                            "Do not call the same tool with the same shape again."
                        ),
                    })
                    _ephemeral_nudge_indices.append(len(messages) - 1)
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM: You are repeating the same output pattern. "
                        "Take a materially different next action now. Continue "
                        "the task; do not summarize or ask the user."
                    ),
                })
                _ephemeral_nudge_indices.append(len(messages) - 1)
            _empty_rounds_this_turn = 0
            continue

        # If generation hit the token cap while a tool call was being
        # produced, do not execute it. The args may be syntactically
        # salvageable but semantically truncated, which caused a real
        # refactor failure to run write_file with `{}` after a 6-minute
        # decode. Retry once with explicit chunking instructions.
        if (
            _round_finish_reason in ("length", "tool_args_limit")
            and (tool_calls or _round_pending_tool_count)
        ):
            try:
                if 0 <= _assistant_msg_idx < len(messages):
                    del messages[_assistant_msg_idx]
            except Exception:
                pass
            _limited_tool_name = getattr(_stream_result, "limited_tool_name", "") or "tool"
            if _limited_tool_name == "tool" and _primary_round_tool:
                _limited_tool_name = _primary_round_tool
            _limited_args_chars = int(getattr(_stream_result, "limited_args_chars", 0) or 0)
            _limited_reason = getattr(_stream_result, "limited_reason", "") or "adaptive stream guard"
            _limited_args_snippet = getattr(_stream_result, "limited_args_snippet", "") or ""
            if _limited_tool_name:
                # A tool_args_limit round never reaches normal tool execution,
                # so record it explicitly in the same failure ledger used by
                # repeated executed tool errors. The key is intentionally
                # coarse: partial JSON snippets vary, but the failure mode is
                # the same tool repeatedly failing to finish an argument stream.
                _stream_limit_key = (
                    _limited_tool_name,
                    f"stream_limit:{_limited_reason or _round_finish_reason}",
                )
                _tool_exec_state.failed_calls[_stream_limit_key] = (
                    _tool_exec_state.failed_calls.get(_stream_limit_key, 0) + 1
                )
            if _empty_rounds_this_turn < _MAX_EMPTY_ROUND_RETRIES:
                _empty_rounds_this_turn += 1
                try:
                    from ..events import emit as _emit_truncated_tool
                    _emit_truncated_tool(
                        "auto_nudge",
                        signal="tool_args_limited",
                        reason="truncated_tool_call",
                        stall_mode="truncated_tool_call",
                        retry=_empty_rounds_this_turn,
                        retry_cap=_MAX_EMPTY_ROUND_RETRIES,
                        finish_reason=_round_finish_reason,
                        pending_tool_count=_round_pending_tool_count,
                        tool_count=len(tool_calls) if tool_calls else 0,
                        limited_tool_name=getattr(_stream_result, "limited_tool_name", ""),
                        limited_args_chars=int(getattr(_stream_result, "limited_args_chars", 0) or 0),
                        limited_reason=getattr(_stream_result, "limited_reason", ""),
                    )
                except Exception:
                    pass
                _limited_args_snippet = (
                    _limited_args_snippet
                ).replace("\n", " ")[:300]
                messages.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: Your previous {_limited_tool_name} call was "
                        f"truncated mid-args at {_limited_args_chars} chars — "
                        "the output token budget ran out before the JSON "
                        "closed, so the call was discarded.\n"
                        "To recover, write the file in CHUNKS:\n"
                        f"  1. Call write_file with the FIRST ~100 lines.\n"
                        f"  2. Call append_file repeatedly to add the rest, "
                        f"~100 lines per call.\n"
                        "Each call's args must fit the per-round token "
                        "budget. Do NOT retry write_file with the same large "
                        "content — it will hit the same wall.\n"
                        f"- Reason: {_limited_reason}\n"
                        f"- Leading snippet: {_limited_args_snippet}"
                    ),
                })
                _ephemeral_nudge_indices.append(len(messages) - 1)
                continue
            _loop_exit_reason = "truncated_tool_call_exhausted"
            break

        # ── Investigation-spin detector ──
        # After every round, decide if THIS round was pure investigation
        # ("looks fine" + read-only tools). Streak length drives whether
        # we inject a corrective nudge into the next round.
        _round_tool_names = {
            (tc.get("function") or {}).get("name", "") for tc in (tool_calls or [])
        }
        _round_was_readonly = bool(_round_tool_names) and _round_tool_names.issubset(_READONLY_TOOLS)
        _round_said_looks_fine = bool(content and _LOOKS_FINE_RE.search(content))
        if _round_was_readonly:
            _readonly_streak += 1
        else:
            _readonly_streak = 0
        if _round_said_looks_fine and _round_was_readonly:
            _looks_fine_streak += 1
        else:
            _looks_fine_streak = 0
        # Trip on the "looks fine" spin — a model that says "looks fine"
        # repeatedly while only reading. The PURE read-only streak (read
        # N files without commenting) is now owned by the semantic-churn
        # detector below at CHURN_READONLY_STREAK_LIMIT (6), which is
        # TIGHTER than the old _MAX_READONLY_STREAK of 10 — so we drop the
        # readonly-streak clause here to avoid a double nudge. This
        # detector keeps its distinctive "files are fine, bug is runtime"
        # framing for the looks_fine case; the churn detector handles the
        # generic "investigating in circles" case. `_churn_nudge_done` is
        # shared so only ONE spin/churn nudge fires per turn.
        if not _spin_nudge_done and not _churn_nudge_done and (
            _looks_fine_streak >= _MAX_LOOKS_FINE_STREAK
        ):
            _spin_nudge_done = True
            try:
                from ..events import emit as _emit_spin
                _emit_spin(
                    "auto_nudge",
                    kind="investigation_spin",
                    readonly_streak=_readonly_streak,
                    looks_fine_streak=_looks_fine_streak,
                    round_idx=round_num,
                )
            except Exception:
                pass
            messages.append({
                "role": "user",
                "content": (
                    "SYSTEM: You've spent several rounds investigating with "
                    "read-only tools and concluding things look correct, but "
                    "the bug the user reported is real. The files are fine; "
                    "the failure is runtime/browser/environment-side and not "
                    "visible from source. Stop investigating. Your next "
                    "response MUST be one of:\n"
                    "  (a) ONE concrete change you think might fix it — add "
                    "console.error hooks, add a try/catch with a visible "
                    "error message in the UI, add defensive null checks, "
                    "add a debug endpoint, or similar — then write/edit a "
                    "file and verify.\n"
                    "  (b) ONE focused question to the user describing what "
                    "you can't see — e.g. \"Is the page blank, unstyled, or "
                    "showing data? Any errors in the browser console "
                    "(Cmd+Opt+J)? What URL did you open?\"\n"
                    "Do NOT call read_file, grep, list_files, glob, "
                    "web_fetch, or web_search this turn. Pick (a) or (b) "
                    "and respond now."
                ),
            })
            _ephemeral_nudge_indices.append(len(messages) - 1)

        # ── Stalled-round recovery ──
        # Three failure modes end a round with the model NOT done: (A) empty
        # round (long reasoning, stream closes with no content/tools); (B)
        # intent-without-action (one narration sentence, no tool call); (C)
        # gave-up-after-rejection (last tool returned REJECTED/Error, round
        # ended with no retry). Detection + nudge live in agent/recovery.py;
        # `stall` is None for a productive round else a StallMode. Gated on
        # Feature.AUTO_NUDGE_RECOVERY (off → stalls end the turn silently).
        from ..features import Feature, is_enabled as _is_enabled
        stall = detect_stall(
            tool_calls=tool_calls,
            content=content,
            tools_called_prior=tools_called,
            messages=messages,
            thinking_abort=_stream_result.thinking_abort,
        ) if _is_enabled(Feature.AUTO_NUDGE_RECOVERY) else None

        if stall is not None:
            # Narration-only / empty stalled assistant messages should not
            # stay in history. Keeping them conditions the next round on the
            # same failed "about to act" text and wastes context.
            if stall in {StallMode.EMPTY, StallMode.NARRATION}:
                try:
                    if 0 <= _assistant_msg_idx < len(messages):
                        del messages[_assistant_msg_idx]
                except Exception:
                    pass
            if _empty_rounds_this_turn < _MAX_EMPTY_ROUND_RETRIES:
                _empty_rounds_this_turn += 1
                mode_label = stall.value
                # Telemetry event for every auto-nudge. Lets us answer
                # "how often does the model stall + recover via nudge
                # vs stall and truly give up?" from events.jsonl alone.
                try:
                    from ..events import emit as _emit_ev
                    _emit_ev("auto_nudge",
                             stall_mode=mode_label,
                             retry=_empty_rounds_this_turn,
                             retry_cap=_MAX_EMPTY_ROUND_RETRIES)
                except Exception:
                    pass
                out.print_info(
                    f"Model round ended with {mode_label} — nudging it to "
                    f"continue… (auto-retry {_empty_rounds_this_turn}/"
                    f"{_MAX_EMPTY_ROUND_RETRIES})"
                )
                # Route through the guard so the SAME stall nudge never fires
                # two rounds running (two-and-threshold discipline). If it's
                # suppressed we still continue — the retry counter advanced and
                # the assistant's dead "about to act" message was already
                # dropped, so the model gets another clean shot without the
                # self-conditioning duplicate nag.
                _append_nudge(nudge_for(stall), kind=f"stall:{stall.name}")
                continue  # next round of the main loop
            else:
                out.print_info(
                    f"Model ended {_MAX_EMPTY_ROUND_RETRIES + 1} rounds in a "
                    "row without acting. Ending turn. Try rephrasing, "
                    "splitting the task, or `/model` to switch models."
                )
                _loop_exit_reason = "stall_exhausted"
                break

        # ── No tool calls = model is done ──
        # Also require no pending tool call: the runtime promotes pending
        # tools into `tool_calls`, but if a future path doesn't, we keep
        # looping instead of ending early (small models often emit a stop
        # token with a call still pending).
        if not tool_calls and not _round_pending_tool_count:
            # Match v0.2.12's no-tool-calls path exactly: render content,
            # render any grounded file summary, break. Nothing else.
            #
            # Removed 2026-04-29:
            #   • completion-gate retry loop — caused multi-round
            #     paraphrase loops on Q&A turns by deleting the assistant
            #     answer and forcing more rounds.
            #   • quality-monitor retry path — same shape (delete msg +
            #     re-run with a "SYSTEM: Do NOT stop…" injection). Even
            #     with the gate gone, this path could still re-fire on
            #     a subset of phrasings ("would you like me to…",
            #     "should I…") and reproduce the multi-round artefact.
            # The model's final answer stands as written; if it asks a
            # permission question, the user gets to answer — the runtime
            # does not silently force a retry.
            _blocking_question = is_focused_blocking_question(content)
            # ── Open-todo completion gate (GENERAL — the goal-typed gates below
            # are dead because infer_goal_state always returns general_task). If
            # the model tries to END its turn while its OWN todo list still has
            # open items, it stopped early: force it to continue to the next item.
            # `session.todos` is cleared to [] by todo_write only when ALL are
            # completed, so a non-empty list is real unfinished work. Bounded by
            # _MAX_TODO_CONTINUATIONS, with a diminishing-returns guard so a model
            # that can't make progress on its plan isn't nagged forever.
            _open_todos = list(getattr(app.session, "todos", []) or [])
            _todo_remaining = sum(
                1 for t in _open_todos if str(t.get("status", "")).lower() != "completed"
            )
            if (
                not _blocking_question
                and _todo_remaining > 0
                and _todo_continue_count < _MAX_TODO_CONTINUATIONS
                and _todo_stuck_count < 3
                and _is_enabled(Feature.AUTO_NUDGE_RECOVERY)
            ):
                # Diminishing returns: if the open-count didn't fall since the
                # last continuation, the model isn't advancing its plan.
                if _todo_remaining >= _last_todo_remaining:
                    _todo_stuck_count += 1
                else:
                    _todo_stuck_count = 0
                _last_todo_remaining = _todo_remaining
                if _todo_stuck_count < 3:
                    _todo_continue_count += 1
                    _nxt = next(
                        (t for t in _open_todos if str(t.get("status", "")).lower() == "in_progress"),
                        None,
                    ) or next(
                        (t for t in _open_todos if str(t.get("status", "")).lower() != "completed"),
                        None,
                    )
                    _nxt_label = (_nxt or {}).get("content", "the next item")
                    out.print_info(
                        f"{_todo_remaining} todo(s) still open — continuing (not stopping early)."
                    )
                    messages.append({"role": "user", "content": (
                        f"SYSTEM: You still have {_todo_remaining} unfinished todo(s). "
                        f"The task is NOT complete — do not stop. Continue now with: "
                        f"{_nxt_label}. Mark a todo completed via todo_write only when it "
                        f"is genuinely done, and keep going until every item is completed."
                    )})
                    _ephemeral_nudge_indices.append(len(messages) - 1)
                    _last_nudge_kind = "todo_continue"
                    continue  # force another round — reject the early completion
            # ── Build-verification STOP gate (claude-code query.ts stop-hook) ──
            # The model wants to END a build_app turn that changed CODE. This is
            # a TRUE completion gate, not a one-shot nudge: we RUN the project's
            # own typecheck/test ourselves and, if it reports errors, inject them
            # and FORCE another round — the model cannot declare "done" while the
            # gate is red. Each completion attempt RE-RUNS the check, so the gate
            # keeps holding until the project is clean or _MAX_BUILD_VERIFY_RETRIES
            # is hit (a genuinely unfixable project can't spin forever).
            #
            # NOTE: gated on build_app, which `infer_goal_state` (now generalist)
            # never sets — so this stop-gate is currently DORMANT. That's
            # intentional for 0.3.20: making it general ("any code-changing turn
            # + a checker") fires a full project typecheck on every one-line edit,
            # which is too aggressive and slow. A proper general trigger (fire on
            # a multi-file / new-file BUILD completion, not a small edit) is a
            # follow-up. Left dormant rather than shipped half-tuned.
            if (
                not _blocking_question
                and _goal_state.goal_type == "build_app"
                and _build_verify_nudges < _MAX_BUILD_VERIFY_RETRIES
                and _is_enabled(Feature.AUTO_NUDGE_RECOVERY)
                and _changed_code_files(changed_files)
            ):
                _self_verified = (
                    has_runtime_verification_signal(bash_history)
                    or ran_build_or_test(bash_history)
                )
                # TIER-2 VERIFICATION: don't just NUDGE the model to typecheck
                # (a small model skips it or mishandles output) — RUN the
                # project's real typecheck/lint ourselves and feed the concrete
                # errors back. Catches semantic errors (wrong names, missing
                # imports) the per-write syntax check can't. See project_check.
                _proj_errors = None
                try:
                    from ..tools.project_check import run_project_check
                    out.print_info("Verifying — running the project's typecheck…")
                    _proj_errors = run_project_check(str(app.repo_root), ctx_tokens=_ctx_tokens_turn)
                except Exception:
                    _proj_errors = None
                try:
                    from ..events import emit as _emit_bg
                    _emit_bg("auto_nudge", signal="build_verify_gate",
                             round_idx=round_num, attempt=_build_verify_nudges,
                             had_errors=bool(_proj_errors),
                             self_verified=bool(_self_verified))
                except Exception:
                    pass
                if _proj_errors:
                    # STOP: cannot finish while the gate is red. Errors are fresh
                    # actionable feedback (they change as the model fixes them),
                    # so this is NOT routed through the "not twice in a row" nag
                    # guard — but we record the kind so a following reminder-style
                    # nudge doesn't stack on top of it.
                    _build_verify_nudges += 1
                    out.print_info("Typecheck found errors — sending them back to fix.")
                    messages.append({"role": "user", "content": (
                        "SYSTEM: the project's typecheck/build was run for you and "
                        "reported errors. FIX each one (targeted edit_file changes), "
                        "then finish. Do not claim it works until these are gone:\n\n"
                        + _proj_errors
                    )})
                    _ephemeral_nudge_indices.append(len(messages) - 1)
                    _last_nudge_kind = "build_verify_errors"
                    continue  # don't accept completion — the gate is red
                if not _self_verified and _build_verify_nudges < 1:
                    # Gate is clean (checker passed OR none configured) AND the
                    # model never ran a build/test itself — advise ONCE to verify.
                    # This one IS a recurring-style nag → routed through the guard.
                    _build_verify_nudges += 1
                    out.print_info("Changed code but never built/ran it — nudging to verify.")
                    if _append_nudge(
                        "SYSTEM: You changed code but never built, type-checked, or ran "
                        "it. Run the project's build/typecheck (npm run build / npx tsc "
                        "--noEmit for TS; tests or an import smoke-check for Python), fix "
                        "every error, then finish. Don't claim it works without building.",
                        kind="build_verify_advise",
                    ):
                        continue  # let it verify before finishing
                # else: gate clean and either self-verified or already advised →
                # fall through and accept completion.
            if (not _blocking_question and _goal_state.goal_type == "edit_existing"
                    and _changed_code_files(changed_files)
                    and not ran_build_or_test(bash_history)
                    and _completion_gate_retries < 1):
                _completion_gate_retries += 1
                if _append_nudge(
                    "SYSTEM: The requested edit exists, but no relevant test, build, "
                    "typecheck, or import check has passed. Run the narrowest "
                    "deterministic verification now, fix failures, then finish.",
                    kind="edit_verify_advise",
                ):
                    continue
            if (not _blocking_question
                    and _goal_state.goal_type in {"build_app", "edit_existing"}
                    and changed_files
                    and "relevant-verification" in _hook_state.verification_registry.requirements
                    and _hook_state.verification_registry.satisfied("relevant-verification", os.environ)):
                from .state_machine import TaskEvent, transition
                _done_transition = transition("verify", TaskEvent.REQUIREMENTS_SATISFIED)
                _announce_task_stage(_done_transition.after.value)
            _completion_blocked = (
                not _blocking_question
                and _goal_state.goal_type in {"build_app", "edit_existing"}
                and bool(_changed_code_files(changed_files))
                and (
                    "relevant-verification" not in _hook_state.verification_registry.requirements
                    or not _hook_state.verification_registry.satisfied(
                        "relevant-verification", os.environ
                    )
                )
            )
            if _completion_blocked:
                content = (
                    "Implementation changes were made, but LocalCode could not record "
                    "a passing build, test, typecheck, or import check for the current "
                    "file hashes. The task remains incomplete rather than claiming success."
                )
            if _goal_state.goal_type == "run_or_launch":
                _task_port = int(getattr(_task_state, "active_port", 0) or 0)
                content = ground_run_or_launch_text(content, _task_port)
            if content:
                _render_markdown(content, app.console if hasattr(app, 'console') else None)
                full_response.append(content)
            # Always show grounded file summary after model response
            if changed_files:
                grounded = _grounded_file_summary(app.repo_root, changed_files)
                if grounded:
                    _render_markdown(grounded, app.console if hasattr(app, 'console') else None)
                    full_response.append(grounded)
            if _goal_state.goal_type == "run_or_launch":
                _task_port = int(getattr(_task_state, "active_port", 0) or 0)
                if _task_port:
                    grounded_access = (
                        f"Running app URL: http://localhost:{_task_port}\n"
                        f"Use that exact port if you open the browser or re-run the server."
                    )
                    _render_markdown(grounded_access, app.console if hasattr(app, 'console') else None)
                    full_response.append(grounded_access)
            _loop_exit_reason = (
                "blocked_question" if _blocking_question
                else "completion_gate:unverified" if _completion_blocked
                else "model_done"
            )
            break

        # ── Execute tools ──
        aggregate_size = 0

        _parallel_futures: dict[int, "object"] = {}
        _parallel_pool = None

        def _skip_parallel(_idx: int, _name: str, _args: dict) -> bool:
            if _name == "read_file":
                _p = _args.get("path") or _args.get("file_path") or ""
                if (isinstance(_p, str) and _p
                        and _p in _tool_exec_state.files_read
                        and _p not in _tool_exec_state.files_modified):
                    return True
            return False

        _parallel_futures, _parallel_pool = prefetch_parallel_tool_calls(
            tool_calls,
            should_skip=_skip_parallel,
            execute_tool=lambda _name, _args: _execute_tool_result(app, _name, _args, out),
        )
        if _parallel_futures:
            try:
                from ..events import emit as _emit_par
                _emit_par("parallel_tool_dispatch",
                          turn_id=_turn_id,
                          round=round_num,
                          count=len(_parallel_futures),
                          names=[tool_calls[_i].get("function", {}).get("name", "")
                                 for _i in _parallel_futures])
            except Exception:
                pass

        for _tc_idx, tc in enumerate(tool_calls):
            # Cancel check — poll before EACH tool so a long tool chain
            # stops the moment the user asks, not after all queued calls
            # have run. We can't interrupt a tool mid-execution
            # (subprocess would need SIGTERM); per-call granularity is
            # the honest trade-off.
            if getattr(app, "cancel_requested", False):
                out.print_info("Stopped by user.")
                loop_detected = True
                _loop_exit_reason = "user_cancel_mid_tool"
                break
            fn = tc.get("function", {})
            tool_name = fn.get("name", "unknown")
            try:
                _raw_args = fn.get("arguments", "{}")
                # `arguments` is normally a JSON string, but some providers/
                # parsers hand back an already-decoded dict — json.loads on a
                # dict raises TypeError, which used to escape and crash the loop.
                args = _raw_args if isinstance(_raw_args, dict) else json.loads(_raw_args)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {}
                out.print_info(f"Warning: malformed args for {tool_name}")

            # Update indicator immediately
            stage = _tool_stage_label(tool_name, args)
            out.set_stage(stage)
            idx = out.log_tool(tool_name, _summarize_args(args))
            # RL trace fidelity: the normal `tool_start` event carries only a
            # summarized + 200-char-truncated arg preview. When capturing
            # trajectories for fine-tuning we need the FULL arguments (e.g. the
            # whole write_file body). Env-gated so normal TUI/headless runs are
            # unaffected; dev/rl/collect sets LOCALCODE_TRACE_FULL_ARGS=1.
            if os.environ.get("LOCALCODE_TRACE_FULL_ARGS"):
                out._emit_event(
                    "tool_call_full", index=str(idx),
                    name=tool_name, arguments=args,
                )

            # Safety: confirm destructive commands (honors current autonomy
            # level so /permissions toggles take effect immediately).
            # The approval returns one of "once" / "always" / "deny".
            # "always" adds the command's first token (e.g. "git") to the
            # session allowlist so we stop asking for that family.
            if _needs_confirmation(tool_name, args, app):
                # `cmd` drives the approval prompt display and the "always
                # allow `<token>`" key. Shell tools carry a command; file-write
                # tools carry a path — show that instead of a blank line.
                cmd = args.get("command", "")
                if not cmd:
                    _wpath = args.get("path") or args.get("file_path") or ""
                    cmd = f"{tool_name} {_wpath}".strip()
                verdict = "deny"
                if out._approval_callback is not None:
                    raw = out._approval_callback(tool_name, cmd)
                    # Callback may be a bool (legacy) or the new verdict string.
                    if isinstance(raw, bool):
                        verdict = "once" if raw else "deny"
                    else:
                        verdict = str(raw)
                else:
                    # CLI mode: terminal-based approval with 3 options.
                    import tty
                    import termios
                    out._stop_indicator()
                    rule = app._composer_rule() if hasattr(app, "_composer_rule") else "  " + ("─" * 60)
                    first_tok = _first_token(cmd) or tool_name
                    sys.stdout.write("\n\033[33m  Allow this command?\033[0m\n")
                    sys.stdout.write(f"\033[2m  {cmd[:80]}\033[0m\n")
                    sys.stdout.write("  \033[1m1\033[0m  allow once\n")
                    sys.stdout.write(f"  \033[1m2\033[0m  always allow `{first_tok}` (this session)\n")
                    sys.stdout.write("  \033[1m3\033[0m  deny\n")
                    sys.stdout.write("\033[s")
                    sys.stdout.write(f"\033[2m{rule}\033[0m\n")
                    sys.stdout.write("  › ")
                    sys.stdout.write(f"\n\033[2m{rule}\033[0m")
                    sys.stdout.write("\033[1A\r    ")
                    sys.stdout.flush()
                    try:
                        fd = sys.stdin.fileno()
                        old = termios.tcgetattr(fd)
                        try:
                            tty.setraw(fd)
                            ch = sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        try:
                            ch = input().strip()
                        except EOFError:
                            ch = "3"
                    sys.stdout.write("\033[u\033[J")
                    if ch in ("1", "y"):
                        verdict = "once"
                    elif ch == "2":
                        verdict = "always"
                    else:
                        verdict = "deny"

                # Act on the verdict.
                if verdict == "always":
                    allow_set = getattr(app, "_session_allow", None)
                    if allow_set is None:
                        app._session_allow = set()
                        allow_set = app._session_allow
                    allow_set.add(_first_token(cmd))
                    sys.stdout.write(
                        f"\033[2m  └ always allowing `{_first_token(cmd)}` for this session\033[0m\n"
                    )
                elif verdict == "once":
                    sys.stdout.write("\033[2m  └ approved command\033[0m\n")
                else:  # deny
                    sys.stdout.write("\033[2m  └ denied\033[0m\n")
                    messages.append({"role": "tool", "content": "Denied by user.", "tool_call_id": tc.get("id", "")})
                    continue

            # Per-tool loop guards REMOVED 2026-04-26 (option C).
            # Telemetry showed:
            #   - 3-in-a-row exact-repeat guard: fired 0× ever
            #   - same-tool > 10 guard: fired 1×, false-positive on
            #     legitimate iterative data analysis (11 different
            #     python3 probes against an 11K-record JSON)
            #   - file-edit > 3 guard: fired 0× ever
            # These guards were paying bookkeeping cost on every tool
            # dispatch to catch loops that don't happen, and cutting
            # legitimate iteration when they did fire. agent (`maxTurns`
            # opt-in) and terminal coding tools (no limit) ship without these too.
            #
            # Loop termination now comes from:
            #   - User Ctrl+C / cancel_requested (always fires)
            #   - thinking-time / thinking-char caps (per-round)
            #   - empty-round nudge (3 strikes)
            #   - investigation-spin nudge (10+ read-only rounds)
            #   - looks-fine streak nudge (3 rounds)
            # If a residual failure pattern emerges in production we
            # add a TARGETED guard for it then. No more coarse counters.

            # Read-dedup: if this is `read_file` for a path we already
            # read THIS TURN AND nobody edited it since, skip the actual
            # dispatch and inject a tiny stub. Real-world heat cause
            # observed 2026-04-26: model re-reads `app.py` (445 lines),
            # `index.html` (76), `app.js` (220) two-three times each
            # while investigating. Each duplicate adds 5-10K chars to
            # the prompt; by round 13 the prompt was 66K chars =
            # 70-90s TTFT/round = full GPU = laptop heat.
            _dedup_stub = dedup_stub_for_tool(tool_name, args, _tool_exec_state)
            # Same-call 3× guard for tools without their own dedup
            # (bash / web_fetch / web_search / launch_app). Took the
            # place of the broader exact-repeat counter that was retired
            # 2026-04-26 — re-added narrowly after an info-fetch turn
            # fired the same read-only `curl` command 4× in a
            # row. dedup_stub wins ties so list_files/glob/grep keep
            # their existing message.
            if _dedup_stub is None:
                _repeat_stub = repeat_stub_for_tool(tool_name, args, _tool_exec_state)
                if _repeat_stub is not None:
                    _dedup_stub = _repeat_stub

            # Tool-arg size guard. Blocks the runaway-edit class
            # (model emitting 100K+ JSON in a single edit_file/
            # write_file/multi_edit). Real failure 2026-04-26: model
            # tried to stuff hundreds of common English words into
            # one edit; the args reached 112,964 chars before
            # truncating mid-string, llama-server's tool-call parser
            # returned HTTP 500, the turn died with E3102. We REJECT
            # with a specific instruction that tells the model HOW
            # to recover (split into multiple edits) so the existing
            # POST_REJECTION stall path nudges it to retry — model
            # doesn't just stop, it actually splits the work.
            _oversize_stub = oversize_stub_for_tool(tool_name, args, 1_000_000)
            # edit_file already carries the exact old text as grounded context.
            # Requiring a separate read first rejected valid one-shot edits and
            # let small models falsely narrate success after the rejection.
            _edit_sequence_stub = None

            # HARD rewrite-stop: the churn NUDGE (limit 3) only advises — logs
            # showed a model rewrite one file 16x while 25-34 nudges fired.
            # Past 2x the nudge limit, REJECT further full rewrites (see
            # recovery.rewrite_hard_stop). Key on the RAW path like the counter.
            _rewrite_limit_stub = None
            if tool_name in ("write_file", "append_file", "multi_edit") and isinstance(args, dict):
                _rw_path = args.get("path") or args.get("file_path") or ""
                _rewrite_limit_stub = rewrite_hard_stop(_rw_path, _file_write_counts)
            # Execute (timed — wall-clock added to _round_tool_exec_ms
            # so round_end can show the model what fraction of its
            # round time was spent waiting on tools vs LLM).
            _tool_started_at = time.monotonic()
            if _rewrite_limit_stub is not None:
                from ..tools import ToolResult as _ToolResult
                _tool_result_obj = _ToolResult(text=_rewrite_limit_stub, ok=False, facts={"tool": tool_name, "ok": False, "repeated_failed_call": True, "rewrite_hard_stop": True})
            elif _edit_sequence_stub is not None:
                from ..tools import ToolResult as _ToolResult
                _tool_result_obj = _ToolResult(text=_edit_sequence_stub, ok=False, facts={"tool": tool_name, "ok": False, "edit_sequence": "missing_context"})
            elif _oversize_stub is not None:
                from ..tools import ToolResult as _ToolResult
                _tool_result_obj = _ToolResult(text=_oversize_stub, ok=False, facts={"tool": tool_name, "ok": False, "oversize": True})
            elif _dedup_stub is not None:
                from ..tools import ToolResult as _ToolResult
                # Match BOTH the soft "REJECTED:" and the hard
                # "REJECTED — HARD STOP:" stubs (same as tool_result_is_error).
                # The old colon-only check mislabeled hard-stop rejections as
                # ok=True / repeated_failed_call=False, corrupting the regression
                # telemetry used to tune these thresholds.
                _dedup_is_error = str(_dedup_stub).startswith("REJECTED")
                _tool_result_obj = _ToolResult(
                    text=_dedup_stub,
                    ok=not _dedup_is_error,
                    facts={
                        "tool": tool_name,
                        "ok": not _dedup_is_error,
                        "dedup": not _dedup_is_error,
                        "repeated_failed_call": _dedup_is_error,
                    },
                )
            elif _tc_idx in _parallel_futures:
                # Already running in the prefetch ThreadPool above; .result()
                # blocks until done. By the time we reach the Nth tool in
                # the serial loop, the first N-1 have usually finished.
                try:
                    _tool_result_obj = _parallel_futures[_tc_idx].result()
                except Exception as _e:
                    from ..tools import ToolResult as _ToolResult
                    _tool_result_obj = _ToolResult(
                        text=f"Error in {tool_name}: {type(_e).__name__}: {_e}",
                        ok=False,
                        facts={"tool": tool_name, "ok": False, "error_type": type(_e).__name__},
                    )
            else:
                _tool_result_obj = _execute_tool_result(app, tool_name, args, out)
            tool_result = hook_after_tool(tool_name, args, str(_tool_result_obj), _hook_state)
            _tool_facts = dict(getattr(_tool_result_obj, "facts", {}) or {})
            if tool_result != str(_tool_result_obj):
                _tool_facts = extract_tool_facts(tool_name, args, str(tool_result))
            _hook_state.evidence.add_tool_result(
                tool_name,
                args,
                str(tool_result),
                _tool_facts,
            )
            if tool_name in {"read_file", "grep", "glob", "list_files"} and not tool_result_is_error(str(tool_result)):
                _edit_context_seen = True
            try:
                from ..events import emit as _emit_tool_facts
                _emit_tool_facts(
                    "tool_facts",
                    turn_id=_turn_id,
                    name=tool_name,
                    facts=_tool_facts,
                )
            except Exception:
                pass
            try:
                from ..events import emit as _emit_tool_result

                _result_text_for_event = str(tool_result)
                _emit_tool_result(
                    "tool_result",
                    name=tool_name,
                    error=str(
                        bool(not _tool_facts.get("ok", True))
                        or tool_result_is_error(_result_text_for_event)
                    ).lower(),
                    chars=len(_result_text_for_event),
                    preview=_result_text_for_event[:160].replace("\n", " "),
                )
            except Exception:
                pass
            _round_tool_exec_ms += int(
                (time.monotonic() - _tool_started_at) * 1000
            )
            _tool_succeeded = bool(_tool_facts.get("ok", True)) and not tool_result_is_error(str(tool_result))
            if _goal_state.goal_type in {"build_app", "edit_existing"}:
                from .state_machine import event_for_tool, transition
                _stage_event = event_for_tool(tool_name, succeeded=_tool_succeeded)
                if _stage_event is not None:
                    _stage_transition = transition(_current_task_stage_for_thinking(), _stage_event)
                    if _stage_transition.changed:
                        _announce_task_stage(_stage_transition.after.value)
            if tool_name == "bash":
                _bash_cmd = str(args.get("command", ""))
                bash_history.append((_bash_cmd, str(tool_result)))
                from ..execution_policy import assess_shell_execution
                _execution = assess_shell_execution(_bash_cmd, str(tool_result), int(_tool_facts.get("exit_code", 0 if _tool_succeeded else 1)))
                if not _execution.task_succeeded:
                    app._last_failed_tool_name = tool_name
                else:
                    app._last_failed_tool_name = ""
                if _goal_state.goal_type in {"build_app", "edit_existing"} and ran_build_or_test([(_bash_cmd, str(tool_result))]):
                    from pathlib import Path as _EvidencePath
                    from ..evidence import EvidenceRequirement
                    _verification_files = tuple(
                        _EvidencePath(path) if _EvidencePath(path).is_absolute()
                        else _EvidencePath(app.repo_root) / path
                        for path in changed_files
                    )
                    _hook_state.verification_registry.require(EvidenceRequirement(
                        "relevant-verification", _verification_files, _bash_cmd,
                        ("PATH", "NODE_ENV", "PYTHONPATH"),
                    ))
                    _hook_state.verification_registry.record(
                        "relevant-verification", environment=os.environ,
                        passed=_execution.task_succeeded, output=str(tool_result),
                    )
                    from .state_machine import TaskEvent, transition
                    _verify_event = TaskEvent.VERIFICATION_PASSED if _execution.task_succeeded else TaskEvent.VERIFICATION_FAILED
                    _verify_transition = transition(_current_task_stage_for_thinking(), _verify_event)
                    if _verify_transition.changed:
                        _announce_task_stage(_verify_transition.after.value)
                if _goal_state.goal_type in {"build_app", "run_or_launch"}:
                    port = extract_port(f"{args.get('command', '')}\n{tool_result}")
                    if port:
                        try:
                            if hasattr(app, "store") and getattr(app, "session", None) is not None:
                                app.store.update_task(
                                    app.session,
                                    active_port=port,
                                )
                        except Exception:
                            pass
            if (
                tool_name == "launch_app"
                and _goal_state.goal_type in {"build_app", "run_or_launch"}
                and "App launched and verified." in str(tool_result)
            ):
                _launch_url = ""
                _launch_port = 0
                _url_match = re.search(r"URL:\s*(http://[^\s]+)", str(tool_result))
                if _url_match:
                    _launch_url = _url_match.group(1).strip()
                    _launch_port = extract_port(_launch_url)
                if _launch_port:
                    try:
                        if hasattr(app, "store") and getattr(app, "session", None) is not None:
                            app.store.update_task(
                                app.session,
                                active_port=_launch_port,
                                current_stage="verified",
                            )
                    except Exception:
                        pass
                _verified_launch_summary = (
                    f"The app is running and verified.\n\n"
                    f"URL: {_launch_url or 'verified by launch_app'}"
                )
                if _goal_state.goal_type == "run_or_launch":
                    _loop_exit_reason = "verified_run_or_launch"
            tools_called.append(tool_name)
            try:
                _recent = list(getattr(app, "_recent_tool_names", []) or [])
                if tool_name in _recent:
                    _recent.remove(tool_name)
                _recent.insert(0, tool_name)
                app._recent_tool_names = _recent[:8]
            except Exception:
                pass
            track_tool_result(
                tool_name=tool_name,
                args=args,
                tool_result=str(tool_result),
                round_num=round_num,
                state=_tool_exec_state,
                dedup_stub=_dedup_stub,
            )
            # ── Semantic-churn counters ──
            # Count EVERY write/edit to a path (content-agnostic) and
            # every FAILED command by its family token. These feed
            # recovery.detect_churn after the round's tools all run.
            if tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}:
                _churn_path = args.get("path") or args.get("file_path") or ""
                if isinstance(_churn_path, str) and _churn_path:
                    _file_write_counts[_churn_path] = _file_write_counts.get(_churn_path, 0) + 1
            elif tool_name == "bash" and tool_result_is_error(str(tool_result)):
                _cmd_tok = command_token(str(args.get("command", "")))
                if _cmd_tok:
                    _command_fail_counts[_cmd_tok] = _command_fail_counts.get(_cmd_tok, 0) + 1
            # Cross-round repeated-call tracking. A write/edit to a path resets
            # that path's read counts (read-after-edit is legitimate, not a
            # repeat). Idempotent read-only tools + bash/web are counted.
            if tool_name in {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}:
                _wp = args.get("path") or args.get("file_path")
                if _wp:
                    for _k in [k for k in _turn_call_sigs if k[0] == "read_file" and str(_wp) in k[1]]:
                        del _turn_call_sigs[_k]
            if tool_name in {"read_file", "grep", "glob", "list_files", "bash", "web_fetch", "web_search"}:
                _sig = (tool_name, canonical_args(args))
                _turn_call_sigs[_sig] = _turn_call_sigs.get(_sig, 0) + 1
            _failure_count = 0
            if tool_result_is_error(str(tool_result)):
                # For bash use the whitespace-normalized cmd_key so this
                # count agrees with the breaker (`dedup_stub_for_tool` keys
                # on `bash_failures`). Keying on raw `failed_calls` here let
                # whitespace-varying re-emissions slip the steering nudge
                # while the breaker still hard-stopped — F2.
                if tool_name == "bash":
                    _failure_count = _tool_exec_state.bash_failures.get(
                        bash_cmd_key(args),
                        0,
                    )
                else:
                    _failure_count = _tool_exec_state.failed_calls.get(
                        (tool_name, canonical_args(args)),
                        0,
                    )
                # Hard-stop nudge: once an identical failing call crosses the
                # backstop threshold, inject ONE strong corrective that bypasses
                # the soft nudge cap. Without this, the cap (2) is exhausted and
                # the loop emits the same REJECTED stub with no steering — the
                # exact 6→11× spin observed 2026-06-19.
                if (
                    _failure_count >= _HARD_STOP_THRESHOLD
                    and tool_name in {"bash", "write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}
                    and not _hard_stop_nudge_fired
                ):
                    _hard_stop_nudge_fired = True
                    messages.append({
                        "role": "user",
                        "content": (
                            f"SYSTEM — HARD STOP: you have sent the identical {tool_name} "
                            f"call {_failure_count} times this turn and it is now permanently "
                            "blocked. It will NOT run again. STOP narrating that you will retry. "
                            "Pick a concrete DIFFERENT action right now: fix the actual cause "
                            "named in the rejection (e.g. create the missing directory, fix "
                            "quoting), or use a different tool. If a project directory does not "
                            "exist, create files with write_file (it makes parent directories) "
                            "instead of cd-ing into a path that isn't there."
                        ),
                    })
                    _ephemeral_nudge_indices.append(len(messages) - 1)
                if _failure_count >= 2 and tool_name in {
                    "write_file",
                    "append_file",
                    "edit_file",
                    "multi_edit",
                    "edit_diff",
                    "bash",
                }:
                    if _generic_correction_nudges < _MAX_CONSECUTIVE_CORRECTIONS:
                        _generic_correction_nudges += 1
                        messages.append({
                            "role": "user",
                            "content": (
                                f"SYSTEM: The last {tool_name} call failed repeatedly. "
                                "Try a materially different approach now. Do not retry "
                                "the same tool arguments."
                            ),
                        })
                        _ephemeral_nudge_indices.append(len(messages) - 1)
                    try:
                        from ..events import emit as _emit_repeat_failure

                        _emit_repeat_failure(
                            "auto_nudge",
                            stall_mode="repeated_failed_tool_call",
                            retry=_failure_count,
                            failed_tool=tool_name,
                        )
                    except Exception:
                        pass
            if tool_name == "read_file":
                _read_file_chars_this_turn += len(str(tool_result))
            if tool_name in {"edit_file", "multi_edit", "edit_diff"} and (
                "old_string not found" in str(tool_result)
                or "applied 0/" in str(tool_result).lower()
                or str(tool_result).startswith("Error:")
                or str(tool_result).startswith("REJECTED:")
                or bool(_tool_facts.get("reverted"))
            ):
                _edit_failures_this_turn += 1
            if tool_name == "write_file" and "already exists" in str(tool_result):
                _write_existing_rejections_this_turn += 1
            # Show result to user — send full result for write/edit so TUI can render diff
            is_error = tool_result_is_error(tool_result)
            # Match BOTH "REJECTED:" and "REJECTED — HARD STOP:" (em-dash) — the
            # latter used to slip past this colon-only check and render as a raw
            # red protocol line to the user.
            is_rejected = tool_result.startswith("REJECTED")
            if is_rejected:
                # Internal tool-routing redirect — the model needs the full
                # REJECTED payload (kept in `messages` below), but the user sees
                # only a short neutral note, not a red error block.
                out.tool_result(_brief_result(tool_name, tool_result), error=False, idx=idx)
            elif tool_name in ("write_file", "append_file", "edit_file") and not is_error:
                out.tool_result(tool_result, error=False, idx=idx)
            else:
                out.tool_result(_brief_result(tool_name, tool_result), error=is_error, idx=idx)

            # Truncate per-tool
            _facts_note = facts_suffix(_tool_facts)
            if _facts_note and _facts_note not in tool_result:
                tool_result = f"{tool_result}{_facts_note}"
            tool_result = _truncate_result(tool_result, tool_name, ctx_tokens=_ctx_tokens_turn)
            aggregate_size += len(tool_result)

            # Aggregate budget. The post-budget clamp scales with the window too:
            # once a turn's cumulative tool output crosses the (dynamic) budget,
            # a big machine still keeps a proportionally larger stub than a 16 GB
            # one instead of a brutal fixed 500 chars.
            if aggregate_size > _aggregate_budget:
                _stub_chars = max(500, int(_ctx_tokens_turn * 3.5 * 0.01)) if _ctx_tokens_turn else 500
                tool_result = tool_result[:_stub_chars] + "\n[Truncated — context budget exceeded this turn]"

            # ── Todo-close verification nudge (claude-code TodoWriteTool) ──
            # When the model just marked a 3+ item todo list fully done and none
            # of the items was a verification step, append a one-line "verify
            # before your final summary" reminder that RIDES ON this tool result
            # — no extra round. Appended after truncation so the short reminder
            # can't be clipped. General + cheap (pure fn of the todos it sent).
            if tool_name == "todo_write" and isinstance(args, dict):
                try:
                    _todo_suffix = todo_close_verification_suffix(args.get("todos"))
                except Exception:
                    _todo_suffix = ""
                if _todo_suffix:
                    tool_result = f"{tool_result}{_todo_suffix}"
                    try:
                        from ..events import emit as _emit_todo_verify
                        _emit_todo_verify(
                            "todo_close_verify_nudge",
                            turn_id=_turn_id,
                            round_idx=round_num,
                            item_count=len(args.get("todos") or []),
                        )
                    except Exception:
                        pass

            # Add to history
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tc.get("id", ""),
            })
            if (
                tool_name in {"edit_file", "multi_edit", "edit_diff"}
                and (not bool(_tool_facts.get("ok", True)) or bool(_tool_facts.get("reverted")))
                and _edit_recovery_nudges < 2
            ):
                _edit_recovery_nudges += 1
                path_hint = args.get("path") or args.get("file_path") or "the target file"
                messages.append({
                    "role": "user",
                    "content": (
                        "SYSTEM: Edit recovery required. Do not retry the same large or fuzzy edit. "
                        f"Read a focused range around {path_hint!r}, copy a smaller exact old_string "
                        "from that read output, then use edit_file/multi_edit with the minimal targeted "
                        "change. If the file is syntactically broken, fix the nearest syntax error first "
                        "and run the smallest verification command."
                    ),
                })
                _ephemeral_nudge_indices.append(len(messages) - 1)
            if _loop_exit_reason == "verified_run_or_launch":
                break

        # Tear down the prefetch pool. By this point every future has
        # been .result()'d in the serial loop above (or the loop broke
        # early on cancel — orphaned futures are allowed to drain
        # because their underlying file/network ops are short-lived).
        if _parallel_pool is not None:
            _parallel_pool.shutdown(wait=False)

        # Content was already streamed live via out.stream(); just record it
        # for the return value (don't re-render — that produced duplicate text
        # below tool results in CLI mode).
        if content:
            full_response.append(content)

        if _loop_exit_reason == "verified_run_or_launch":
            if _verified_launch_summary:
                _render_markdown(
                    _verified_launch_summary,
                    app.console if hasattr(app, 'console') else None,
                )
                full_response.append(_verified_launch_summary)
            break

        # NOTE: there used to be a "run_or_launch + a port opened ⇒ the task is
        # done" break here. It was an anti-pattern — none of the reference
        # harnesses (codex, opencode, pi, claude-code) infer completion from a
        # detected port, a launched process, or output patterns. All of them end
        # a turn on ONE structural signal: the model returned a message with no
        # tool calls. That block let `npm run dev` opening a socket count as a
        # finished app at ~10% of the work. Removed for good. Completion is now
        # model-driven (the no-tool-calls exit below), backstopped only by the
        # open-todo gate for the small local models that stop early — the harness
        # nudges AGAINST stopping, it never invents a "done" signal. If a run
        # task launches a server, the model reports the URL in its own reply,
        # exactly like the reference tools do.

        # ── Cross-round repeated-call nudge ──
        # The dominant waste in the logs: the model calls the SAME (tool, args)
        # over and over across rounds (read_file same path 53x; pkill->curl->read
        # spins). The in-round breaker misses it and these calls SUCCEED so the
        # failure breakers don't trip. NUDGE only — the model already has every
        # result, so we never withhold content (no read-dedup-stub starvation).
        if (
            not _xround_repeat_nudge_done
            and not _churn_nudge_done
            and not _spin_nudge_done
            and _is_enabled(Feature.AUTO_NUDGE_RECOVERY)
        ):
            _hot = [(k, c) for k, c in _turn_call_sigs.items() if c >= CROSS_ROUND_REPEAT_LIMIT]
            if _hot:
                _hk, _hc = max(_hot, key=lambda kc: kc[1])
                _xround_repeat_nudge_done = True
                try:
                    from ..events import emit as _emit_xr
                    _emit_xr("auto_nudge", signal="cross_round_repeat",
                             tool=_hk[0], count=_hc, round_idx=round_num)
                except Exception:
                    pass
                out.print_info(
                    f"Detected a cross-round loop ({_hk[0]} called {_hc}x with the "
                    "same args) — nudging it to use the result it has."
                )
                messages.append({"role": "user", "content": (
                    f"SYSTEM: You've already called {_hk[0]} with the SAME arguments "
                    f"{_hc} times this turn — its result is unchanged and already "
                    "above. Do NOT call it again. Use the result you have and take "
                    "the next concrete step (make an edit, run the build, or finish). "
                    "If a server/port check keeps failing, the server is on a "
                    "different port or not running — read its startup log instead of "
                    "re-checking."
                )})
                _ephemeral_nudge_indices.append(len(messages) - 1)

        # ── Semantic-churn nudge ──
        # After the round's tools all ran, check whether the turn-so-far
        # is thrashing: same file rewritten N times, same command failing
        # N times, or a long pure read-only spin. The byte-identical-call
        # breakers miss these (different content / different output each
        # time). ONE nudge per turn — if the model ignores it, the normal
        # exit paths (repeated-failure guard, stall, completion gate)
        # still apply. Gated on the same recovery feature flag as stalls.
        # Update the planning-without-progress streak for THIS round. Disjoint
        # from the read-only-spin signal: a pure read-only round feeds
        # `_readonly_streak` (INVESTIGATION_SPIN) and is deliberately NOT
        # counted here, so this signal targets exactly what that one misses —
        # rounds that re-derive the plan via thinking/narration without
        # reading or doing anything. A round makes "progress" if it changed a
        # new file or ran a build/verify this round; either resets the streak.
        # Otherwise, a non-read-only round that produced thinking/narration
        # counts as another re-planning round. Quiet rounds (no progress, no
        # narration) leave the streak unchanged.
        _round_changed_new_file = len(changed_files) > _changed_files_at_round_start
        _round_bash = bash_history[_bash_history_at_round_start:]
        _round_ran_build = ran_build_or_test(_round_bash)
        # Successful setup/scaffolding is concrete progress too. The live trace
        # incorrectly called mkdir + npm create "re-planning without progress".
        _round_ran_successful_bash = any(
            result and not str(result).startswith(("[exit code ", "Error:", "REJECTED:"))
            for _cmd, result in _round_bash
        )
        _round_had_reasoning = bool(
            (content and content.strip()) or _round_reasoning_chars > 0
        )
        if _round_changed_new_file or _round_ran_build or _round_ran_successful_bash:
            _planning_streak = 0
        elif _round_had_reasoning and not _round_was_readonly:
            _planning_streak += 1

        if not _churn_nudge_done and not _spin_nudge_done and _is_enabled(Feature.AUTO_NUDGE_RECOVERY):
            _churn = detect_churn(
                file_write_counts=_file_write_counts,
                command_fail_counts=_command_fail_counts,
                readonly_streak=_readonly_streak,
                planning_streak=_planning_streak,
            )
            if _churn is not None:
                _churn_nudge_done = True
                _last_churn_mode = _churn.mode.value
                try:
                    from ..events import emit as _emit_churn
                    _emit_churn(
                        "auto_nudge",
                        signal="semantic_churn",
                        churn_mode=_churn.mode.value,
                        subject=_churn.subject,
                        count=_churn.count,
                        round_idx=round_num,
                    )
                except Exception:
                    pass
                out.print_info(
                    f"Detected churn ({_churn.mode.value}) — nudging it to "
                    "stop thrashing and make a targeted fix."
                )
                messages.append({"role": "user", "content": churn_nudge_for(_churn)})
                _ephemeral_nudge_indices.append(len(messages) - 1)
                _last_nudge_kind = "churn"

        # ── Reject → re-read → reject recovery rung (bounded, single-shot) ──
        # The remaining log loop the other detectors miss: a mutation keeps
        # coming back REJECTED (dedup / "already modified" / old_string-not-found)
        # and the model reacts by re-reading the file and retrying the same
        # shape. detect_reject_reread_loop RECOMPUTES the signal from the
        # transcript tail (survives compaction). We give ONE actionable redirect
        # per turn, then stop nagging (the guard boolean). Composes with the
        # existing breakers: gated so it won't fire in a round another nudge
        # already fired, and `_append_nudge` blocks the same nag twice running.
        if (
            not _reject_reread_nudge_done
            and not _churn_nudge_done
            and not _spin_nudge_done
            and not _xround_repeat_nudge_done
            and not _hard_stop_nudge_fired
            and _is_enabled(Feature.AUTO_NUDGE_RECOVERY)
        ):
            _rr_count = detect_reject_reread_loop(messages)
            if _rr_count is not None:
                if _append_nudge(reject_reread_nudge(), kind="reject_reread"):
                    _reject_reread_nudge_done = True
                    out.print_info(
                        "Detected a reject → re-read loop — giving it one "
                        "targeted redirect."
                    )
                    try:
                        from ..events import emit as _emit_rr
                        _emit_rr("auto_nudge", signal="reject_reread_loop",
                                 count=_rr_count, round_idx=round_num)
                    except Exception:
                        pass

        # Break outer loop if loop was detected
        if loop_detected:
            break

        # Compaction is handled ONCE per round at the TOP of the loop by the
        # window-aware `should_compact`/`compact` path (see ~line 612), whose
        # threshold scales with the model's real context window and the user's
        # RAM. The old fixed-12K/27K second pass that used to live here fired at
        # <5% of a 256K window, crushed all but 8 messages into a 5-line note,
        # and made the model lose provenance of its own writes → the re-read /
        # "fix systematically" churn loop. Removed: one dynamic system only.

    else:
        # `for…else` only fires when the loop exhausted naturally
        # (i.e. `range(MAX_ROUNDS)` was consumed without a `break`).
        # When MAX_ROUNDS<=0 we use `itertools.count()` which never
        # exhausts, so this branch only fires for a positive cap.
        if MAX_ROUNDS > 0:
            out.print_info(f"Reached max rounds ({MAX_ROUNDS})")
            _loop_exit_reason = "max_rounds"

    # Fallback: if we somehow exited without setting a reason, mark
    # so it's visible in telemetry rather than blank. Covers any
    # break path I haven't yet annotated.
    if not _loop_exit_reason:
        _loop_exit_reason = "unknown"

    # Strip ephemeral nudges from `messages` BEFORE persisting the
    # turn. These were synthetic SYSTEM-prefixed user messages we
    # injected mid-loop to push the model out of a spin or stall;
    # they served their purpose during this turn. Leaving them in
    # `messages` poisons future turns — the model continues honoring
    # "STOP investigating, do NOT call read_file…" instructions long
    # after the spin is over. Pop in reverse so earlier indices stay
    # valid.
    if _ephemeral_nudge_indices:
        strip_ephemeral_nudges(messages, _ephemeral_nudge_indices)

    # End-of-turn telemetry. Captures the complete picture: full
    # user input was in turn_start; full final assistant response
    # goes here along with tool counts, duration, and success flag.
    # Together turn_start + turn_end form a replayable record of
    # the turn for offline debugging.
    final_text = "".join(full_response).strip()
    try:
        from ..events import emit as _emit_edit_quality
        _emit_edit_quality(
            "edit_quality",
            turn_id=_turn_id,
            read_file_chars=_read_file_chars_this_turn,
            edit_failures=_edit_failures_this_turn,
            write_existing_rejections=_write_existing_rejections_this_turn,
            changed_files_count=len(changed_files),
            changed_files=changed_files[:20],
        )
    except Exception:
        pass
    _final_task_stage = getattr(_task_state, "current_stage", "")
    finalize_turn(
        app=app,
        turn_id=_turn_id,
        task_state=_task_state,
        goal_state=_goal_state,
        prompt_result=prompt_result,
        final_text=final_text,
        loop_exit_reason=_loop_exit_reason,
        final_task_stage=_final_task_stage,
        started_mono=_turn_started_mono,
        time_module=_time_mod,
        tools_called=tools_called,
        round_num=round_num if "round_num" in dir() else None,
        tokens_in=_turn_prompt_tokens,
        tokens_out=_turn_completion_tokens,
        tokens_total=_turn_total_tokens,
    )
    return final_text
