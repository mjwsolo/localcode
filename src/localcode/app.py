from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import logging
from pathlib import Path
import re

from rich.console import Console
from rich.panel import Panel

from .approvals import ApprovalItem
from .compact import compact_messages
from .composer import compose_messages
# .browser + .mcp removed during T0.9 — browser automation and MCP client
# loading were dead code (no user configured an MCP server; Playwright
# browser path was unreachable from the agent loop).
from .config import AppConfig, ensure_home_dirs, save_config
from .context import build_context_block, find_repo_root
from .indexer import build_index, load_index, search_index
from .models import GEMMA_PROFILES, get_runtime_model, resolve_profile
from .patching import build_diff, parse_diff
from .permissions import PermissionStore
from .runtime import LocalCodeRuntimeGateway, RuntimeErrorWithContext
from .session import SessionStore
from .tool_router import route_tools
from .cache import SpeculativeExecutor, ToolResultCache
from .permissions_v2 import PermissionManager
try:
    from .display import ThinkingIndicator, ToolCallDisplay, ResponseDisplay, SessionStats
except ImportError:
    ThinkingIndicator = ToolCallDisplay = ResponseDisplay = SessionStats = None
from .output import OutputManager
from .skills import resolve_referenced_skills
from .toolkit import LocalCodeToolkit
from .embeddings import EmbeddingSearch
from .history import HistoryDB
from .verification import guess_verify_command, run_verification
from .auto_compact import compact_if_needed
from .autonomy import AutonomyLevel, apply_autonomy_to_permissions, get_policy
from .hooks import HookRunner
from .snapshots import SnapshotStore, create_snapshot
from .turn_diff import TurnDiffTracker, print_turn_diff
from .agent.goal import GoalState, infer_goal_state
from .performance import detect_machine_profile, benchmark_report, apply_preset, should_promote_legacy_default_to_laptop_26b


_ONLINE_CACHE: tuple[float, bool] | None = None
_ONLINE_CACHE_TTL = 30.0  # seconds


_FOLLOWUP_CHAT_ONLY = {
    "thanks",
    "thank you",
    "ok",
    "okay",
    "cool",
    "nice",
}


_CODING_TASK_KINDS = {"new_app", "existing_app_edit", "run_or_launch"}


def _looks_like_task_followup(user_text: str, current_task: object | None) -> bool:
    """Return true when a short correction should continue the task.

    This is deliberately conservative: it only attaches to a coding task
    and only for short correction/addition wording. The actual feature
    remains model/tool-driven; LocalCode does not classify domain terms.
    """
    if current_task is None:
        return False
    if getattr(current_task, "task_kind", "") not in _CODING_TASK_KINDS:
        return False
    text = (user_text or "").strip().lower()
    if not text:
        return False
    first = next(iter(re.findall(r"[a-z0-9]+", text)), "")
    if text in _FOLLOWUP_CHAT_ONLY:
        return False
    if len(text) <= 80 and "?" not in text and first not in {"what", "why", "how", "where", "when", "who"}:
        return True
    return any(
        phrase in text
        for phrase in (
            "i meant",
            "you forgot",
            "missing",
            "add ",
            "also add",
            "it should",
            "should have",
            "doesn't have",
            "didn't add",
        )
    )


def _continue_goal_for_task_followup(goal_state: GoalState, user_text: str, current_task: object) -> GoalState:
    criteria = list(getattr(current_task, "success_criteria", []) or goal_state.success_criteria)
    criteria.extend([
        "Follow-up correction is implemented in the existing task context",
        "Assistant does not stop with a permission question when the requested implementation is local and feasible",
    ])
    current_kind = getattr(current_task, "task_kind", goal_state.task_kind)
    if current_kind in {"new_app", "existing_app_edit"}:
        goal_type = "edit_existing"
        task_kind = "existing_app_edit"
    else:
        goal_type = getattr(current_task, "goal_type", goal_state.goal_type)
        task_kind = current_kind
    # Preserve task identity; this turn is a correction on existing work,
    # not a fresh task name.
    return replace(
        goal_state,
        goal_type=goal_type,
        task_kind=task_kind,
        task_slug=getattr(current_task, "task_slug", goal_state.task_slug),
        goal_summary=f"{getattr(current_task, 'goal_summary', '')}\nFollow-up: {user_text}".strip()[:240],
        success_criteria=tuple(dict.fromkeys(criteria)),
        allows_blocking_question=False,
    )


def _canonical_project_dir_has_files(repo_root: Path, slug: str) -> bool:
    """True when a creation task's canonical project directory already exists.

    Repeated identical app-build prompts should resume/repair the same project
    identity, not invent sibling names. This helper lets the controller expose
    read/edit tools immediately instead of wasting rounds on write_file
    rejections against existing files.
    """
    clean_slug = (slug or "").strip()
    if not clean_slug:
        return False
    project_dir = repo_root / clean_slug
    if not project_dir.is_dir():
        return False
    try:
        return any(project_dir.iterdir())
    except OSError:
        return False


def is_online() -> bool:
    """True when the machine appears to have working internet.

    Checks a single TCP-connect to a well-known host, cached for ~30 s
    so we don't hit the network every prompt. The model's system-prompt
    network_status block reads this to decide whether to call
    web_search / web_fetch / pip / curl.

    Previously this was hardcoded `return True`, which misled the model
    into attempting downloads on offline machines. Now we actually
    probe.
    """
    global _ONLINE_CACHE
    import socket
    import time
    now = time.time()
    if _ONLINE_CACHE is not None:
        cached_at, verdict = _ONLINE_CACHE
        if now - cached_at < _ONLINE_CACHE_TTL:
            return verdict
    try:
        # Cloudflare public DNS on HTTPS — minimal connect handshake
        # proves we have an internet route. No content transferred.
        sock = socket.create_connection(("1.1.1.1", 443), timeout=1.5)
        sock.close()
        verdict = True
    except (OSError, socket.timeout):
        verdict = False
    _ONLINE_CACHE = (now, verdict)
    return verdict


def notify_if_slow(_app_name: str, _message: str, _start_time: float) -> None:
    """Stub: no-op after notify module removal."""
    pass

import os
import time


class LocalCodeApp:
    def __init__(
        self,
        config: AppConfig,
        session_id: str | None = None,
        cwd: Path | None = None,
        profile_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.console = Console()
        self.log = logging.getLogger("localcode")
        self.config = config
        machine = detect_machine_profile()
        if should_promote_legacy_default_to_laptop_26b(self.config, machine):
            _, preset = benchmark_report(self.config, self.config.runtime.mode)
            apply_preset(self.config, preset, model=self.config.runtime.model)
        self.repo_root = find_repo_root(cwd or Path.cwd())
        self.store = SessionStore()
        if session_id:
            self.session = self.store.load(session_id)
        else:
            self.session = self.store.create(
                self.repo_root,
                profile=profile_name or config.runtime.profile,
                model=model_name or config.runtime.model,
            )
        if profile_name:
            self.session.profile = profile_name
        if model_name is not None:
            self.session.model = model_name
        self.profile = resolve_profile(self.session.profile, self.session.model)
        self.runtime_model = get_runtime_model(self.profile, self.session.model)
        self.session.profile = self.profile.key
        self.session.model = self.runtime_model
        self.config.runtime.profile = self.profile.key
        self.config.runtime.model = self.runtime_model
        self.engine = LocalCodeRuntimeGateway(config.runtime)
        self.toolkit = LocalCodeToolkit(self.repo_root, config, app=self)
        self.permissions = PermissionStore(self.repo_root)
        self.store.save(self.session)
        self.session_allows: set[str] = set()
        # Per-session "always allow" bash first-tokens (git, pip, etc.) —
        # populated when the user picks option 2 on the approval prompt.
        # Cleared on next process start.
        self._session_allow: set[str] = set()
        self._thinking_tick = 0
        # Removed unused pending modality fields (T0.9 purge — no
        # UI path ever fed them, and the agent.ask() args were never
        # passed from any live caller).
        self.stats = SessionStats()
        self.tool_cache = ToolResultCache(self.repo_root)
        self._vim_mode = False
        # Plan-mode state (see src/localcode/plans.py and ARCHITECTURE.md).
        # This is now front-end workflow metadata and a plan-file handle;
        # the core task/turn runtime no longer branches on it.
        self.plan_mode: bool = False
        self.plan_slug: str | None = None
        # User-requested cancel: flipped to True by the TUI when the
        # user types "stop" / "cancel" / "abort" during a running turn.
        # The agent loop polls this between rounds and before each tool
        # execution, breaking out cleanly when set. Reset at each turn
        # start by the chat screen's _start_turn.
        self.cancel_requested: bool = False
        self._output_style: str = ""
        self.out = OutputManager()  # centralized output
        self.out.set_event_callback(self._record_exec_event)
        self.logger = None  # SessionLogger removed
        self.history = HistoryDB()  # unified SQLite history
        self.embedding_search = EmbeddingSearch(str(self.repo_root))
        self.perms = PermissionManager()  # tool permissions
        self.hooks = HookRunner(str(self.repo_root), self.session.session_id, self.runtime_model)
        self.snapshot_store = SnapshotStore()
        self.turn_tracker = TurnDiffTracker(self.repo_root)
        # Apply autonomy mode
        autonomy_level = os.environ.get("LOCALCODE_AUTONOMY", "auto_edit")
        self._autonomy = AutonomyLevel(autonomy_level) if autonomy_level in ("suggest", "auto_edit", "full_auto") else AutonomyLevel.AUTO_EDIT
        apply_autonomy_to_permissions(self.perms, get_policy(self._autonomy))
        # Run session start hook
        self.hooks.on_session_start()
        self._spec_executor = SpeculativeExecutor(self.toolkit.execute_tool_calls)
        self._memory = self._load_memory()
        # Per-session notebook directory — a disk-backed "working memory"
        # area the model can use for drafts, plans, and intermediate data
        # without polluting the user's project tree or the in-context
        # conversation history. See src/localcode/notebook.py for the
        # rationale and directory layout.
        from .notebook import notebook_dir_for, gc_old_sessions
        gc_old_sessions()
        self.notebook_dir: Path = notebook_dir_for(self.session.session_id)

        # One-shot session-info event — machine specs + active config.
        # Fires exactly once per process (idempotent). Gives us the
        # anchor for correlating turn/tool/server events with the
        # hardware and model mix they ran against. See events.py for
        # the MECE bucket taxonomy.
        try:
            from .events import emit_session_info_once
            emit_session_info_once(
                model=str(self.config.runtime.model or ""),
                runtime_mode=str(self.config.runtime.laptop_26b_runtime_mode or ""),
                context_window=int(
                    getattr(self.config.runtime, "llama_cpp_ctx_size", 0) or 0
                ),
                autonomy=str(getattr(self, "_autonomy", "")),
                session_id=self.session.session_id,
                project_root=str(self.repo_root),
            )
        except Exception:
            pass

    def _record_exec_event(self, event_type: str, payload: dict) -> None:
        detail = payload.get("stage") or payload.get("name") or payload.get("message") or payload.get("chunk") or event_type
        detail_text = str(detail).replace("\n", " ")[:200]
        self.store.append_event(
            self.session,
            f"exec:{event_type}",
            detail_text,
            **{k: str(v)[:240] for k, v in payload.items() if v is not None},
        )

    def _load_memory(self) -> dict:
        p = ensure_home_dirs() / "memory.json"
        if p.exists():
            try:
                import json
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    def _save_memory(self) -> None:
        import json
        p = ensure_home_dirs() / "memory.json"
        p.write_text(json.dumps(self._memory, indent=2))

    def _record_runtime_sample(self, *, first_token_s: float | None, total_s: float | None) -> None:
        if self.profile.key != "gemma4-26b-laptop":
            return
        mode = "fit" if self.config.runtime.provider == "llama_cpp" else "speed"
        bucket = self._memory.setdefault("runtime_telemetry", {}).setdefault("gemma4-26b-laptop", {}).setdefault(mode, {})
        samples = int(bucket.get("samples", 0)) + 1
        bucket["samples"] = samples
        alpha = 0.35
        if first_token_s is not None:
            prev = float(bucket.get("ema_first_token_s", first_token_s))
            bucket["ema_first_token_s"] = round(prev * (1 - alpha) + first_token_s * alpha, 3)
        if total_s is not None:
            prev_total = float(bucket.get("ema_total_s", total_s))
            bucket["ema_total_s"] = round(prev_total * (1 - alpha) + total_s * alpha, 3)
        bucket["provider"] = self.config.runtime.provider
        bucket["model"] = self.runtime_model
        bucket["updated_at"] = int(time.time())
        self._save_memory()

    def _on_project_files_changed(self, modified: list[str], created: list[str], deleted: list[str]) -> None:
        """Called by ProjectWatcher when files change externally."""
        # Invalidate tool cache for modified/deleted files
        for f in modified + deleted:
            self.tool_cache.invalidate(f)
        # Update embedding index incrementally if it exists
        if hasattr(self, 'embedding_search') and self.embedding_search.is_indexed():
            all_changed = modified + created + deleted
            if all_changed:
                try:
                    self.embedding_search.update_files(all_changed)
                except Exception:
                    pass


    def _handle_command(self, raw: str) -> bool:
        """Minimal command handler -- TUI only uses /undo."""
        name, _, arg = raw.partition(" ")
        arg = arg.strip()
        if name == "/undo":
            if arg == "all":
                messages = self.toolkit.changes.undo_all()
                for msg in messages:
                    self.console.print(f"  {msg}")
                self.console.print(f"Undid {len(messages)} changes.")
            else:
                ok, msg = self.toolkit.changes.undo_last()
                self.console.print(msg)
            return True
        return True

    def ask(self, user_text: str, stream: bool = True) -> str:
        # Reset per-turn state
        self.perms.new_turn()
        goal_state = infer_goal_state(user_text)
        _continuing_task = False
        _current_task = getattr(self.session, "current_task", None)
        # Followup classifier disabled — was false-matching plain English
        # phrases ("should have") in fresh build requests and forcing
        # goal_type=edit_existing, which then triggered the loop.py:1310
        # "edit_existing workflow requires context before patching"
        # rejection on write_file calls into brand-new project dirs.
        # Generalist mode: every turn runs through the same path.

        # Avoid legacy context leakage across unrelated app/edit tasks.
        # Project facts are reloaded below via context/retrieval; old chat
        # transcripts from a different task mostly add stale paths, stale
        # ports, and stale implementation assumptions.
        if (
            not _continuing_task
            and _current_task is not None
            and goal_state.goal_type in {"build_app", "edit_existing"}
        ):
            prev_slug = str(getattr(_current_task, "task_slug", "") or "")
            next_slug = str(getattr(goal_state, "task_slug", "") or "")
            prev_status = str(getattr(_current_task, "status", "") or "")
            if prev_slug and next_slug and prev_slug != next_slug and prev_status in {"completed", "blocked", "failed", "cancelled"}:
                self.session.messages.clear()

        # ── Dynamic task name (like agent) ──
        # Keep task_slug internal. Showing it here made users think LocalCode
        # was forcing weird deterministic folder names.
        task_name = self._generate_task_name(user_text)
        if stream and task_name:
            self.out.set_stage(task_name)

        # Start centralized output — thinking indicator appears immediately
        if stream:
            self.out.start_thinking()

        # Two-tier routing: 26B stays loaded always, e2b handles simple queries.
        # With OLLAMA_MAX_LOADED_MODELS=2, both coexist — no swap penalty.
        # Only DE-escalate (26B→e2b for speed), never escalate (avoids 3min swap).
        self._maybe_use_fast_model(user_text)

        self._adapt_to_prompt(user_text)
        self._apply_cache_policy()
        history_cap = self._effective_context_chars()
        if goal_state.goal_type == "build_app":
            history_cap = min(history_cap, 12_000)
        elif goal_state.goal_type == "edit_existing":
            history_cap = min(history_cap, 16_000)
        self.session.messages = compact_messages(self.session.messages, max_chars=max(5_000, history_cap))

        # -- Context gathering — skip heavy stuff for simple creation tasks --
        ctx_chars = self._effective_context_chars()
        context_result = ""
        retrieval_result = ""
        skill_result = ""
        plan_result = None
        draft_result = ""

        try:
            # Always gather context — model needs to know what files exist
            with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(build_context_block, self.repo_root, self.session.pinned_files, ctx_chars): "context",
                        pool.submit(self._retrieval_context, user_text): "retrieval",
                        pool.submit(resolve_referenced_skills, self.repo_root, user_text): "skills",
                        pool.submit(self.plan_for_task, user_text): "plan",
                    }
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            result = future.result()
                            if key == "context":
                                context_result = result
                            elif key == "retrieval":
                                retrieval_result = result
                            elif key == "skills":
                                skill_result = "\n\n".join(f"Skill {name}:\n{content}" for name, content in result)
                            elif key == "plan":
                                plan_result = result
                        except Exception:
                            pass
        except KeyboardInterrupt:
            self.out.done()
            self.console.print("\n  [dim]interrupted[/]")
            return ""

        context = context_result
        if retrieval_result:
            context = f"{context}\n\nIndexed code matches:\n{retrieval_result}"

        # No system prompt here — agent_loop sets task-specific ones
        system_prompt = ""

        # Unused multipart paths removed during T0.9 purge.
        # No UI path ever fed them; the branches just kept compose_messages
        # aware of modalities we do not support. Text-only is the only
        # supported input today. If multimodal lands for real, reintroduce
        # behind an explicit Feature flag.
        composed_messages = compose_messages(
            self.profile,
            system_prompt,
            context,
            self.session.messages,
            user_text,
            provider=self.config.runtime.provider,
        )

        # Lifecycle hook: user prompt submit
        hook_result = self.hooks.on_user_prompt_submit(user_text)
        if hook_result.blocked:
            self.console.print(f"[dim]  Hook blocked prompt: {hook_result.error or hook_result.output}[/]")
            return
        # Start turn tracking
        self.turn_tracker.start_turn(watched_files=self.session.pinned_files)
        _turn_start = time.time()

        self.session.messages.append({"role": "user", "content": user_text})
        self.store.append_event(self.session, "user", user_text[:160])

        # Pre-2026-04-29 we ran a model call (`extract_feature_criteria`)
        # at task intake to decompose the user's request into a
        # verifiable checklist that the goal block could carry. Removed
        # because (a) it added 3-10 s of latency per first turn, (b)
        # mainstream agentic CLIs don't do this, (c) the resulting
        # criteria didn't actually prevent the loop / repeat /
        # over-verification failures it was added to mitigate. We now
        # use only the static `goal_state.success_criteria` from the
        # rule-based classifier; the goal block consumer downstream
        # filters out the generic placeholders, so an unset list is fine.
        merged_criteria = list(goal_state.success_criteria)

        if _continuing_task:
            task_state = self.store.continue_task(
                self.session,
                user_request=user_text,
                goal_type=goal_state.goal_type,
                goal_summary=goal_state.goal_summary,
                success_criteria=merged_criteria,
            )
            if task_state is None:
                task_state = self.store.create_task(
                    self.session,
                    user_request=user_text,
                    goal_type=goal_state.goal_type,
                    task_kind=goal_state.task_kind,
                    task_slug=goal_state.task_slug,
                    goal_summary=goal_state.goal_summary,
                    success_criteria=merged_criteria,
                )
        else:
            task_state = self.store.create_task(
                self.session,
                user_request=user_text,
                goal_type=goal_state.goal_type,
                task_kind=goal_state.task_kind,
                task_slug=goal_state.task_slug,
                goal_summary=goal_state.goal_summary,
                success_criteria=merged_criteria,
            )
        if (
            not _continuing_task
            and goal_state.goal_type == "build_app"
            and goal_state.task_kind == "new_app"
            and _canonical_project_dir_has_files(self.repo_root, goal_state.task_slug)
        ):
            task_state.current_stage = "scaffolding"
            task_state.goal_summary = (
                f"{task_state.goal_summary}\n"
                "Existing canonical project directory found; continue inside it."
            ).strip()[:240]
            self.store.save(self.session)
        self.history.record_user_prompt(
            self.session.session_id, str(self.repo_root),
            user_text, model=self.runtime_model,
            task_id=task_state.task_id,
            task_status=task_state.status,
            task_stage=task_state.current_stage,
            goal_type=goal_state.goal_type,
            goal_summary=goal_state.goal_summary,
            task_kind=goal_state.task_kind,
            task_slug=goal_state.task_slug,
        )

        changes_before = self.toolkit.changes.change_count
        assistant_text = ""

        # Pass the ask-level indicator to model methods (ONE indicator for entire flow)
        out = self.out

        try:
            # State machine agent loop: harness controls sequencing, model generates content.
            # Based on agent/terminal coding tools analysis: GATHER → WRITE → VERIFY → FIX → DONE
            from .agent import run_agent_loop
            self._active_goal_state = goal_state.as_dict()
            assistant_text = run_agent_loop(
                self, user_text, composed_messages, self.out,
            )
        except KeyboardInterrupt:
            self.out.done()
            # Auto-save ghost snapshot on interrupt
            create_snapshot(
                self.session.session_id, str(self.repo_root),
                self.session.messages, self.session.pinned_files,
                model=self.runtime_model, reason="interrupt",
                capture_files=self.session.pinned_files,
            )
            self.console.print("\n[dim]  Snapshot saved. Use /snapshot list to see saved points.[/]")
            return ""
        except RuntimeErrorWithContext as exc:
            lowered_exc = str(exc).lower()
            retried = False
            if any(token in lowered_exc for token in ("context", "prompt too long", "token", "overflow")):
                max_ctx = self._effective_context_chars()
                compacted, was_compacted = compact_if_needed(self.session.messages, max_ctx, keep_recent=6)
                if was_compacted:
                    self.session.messages = compacted
                    self.store.append_event(self.session, "exec:reactive_compact", "Compacted context after runtime overflow", error=str(exc)[:200])
                    self.store.save(self.session)
                    try:
                        from .agent import run_agent_loop
                        self._active_goal_state = goal_state.as_dict()
                        assistant_text = run_agent_loop(self, user_text, composed_messages, self.out)
                        retried = True
                    except Exception:
                        retried = False
            if not retried:
                from .errors import format_for_user
                self.out.set_error(format_for_user(exc, fallback_code="E3102"))
                self.log.exception("Runtime error")
                try:
                    simple = [{"role": "user", "content": user_text}]
                    response = self.engine.chat_once(simple)
                    assistant_text = response.get("message", {}).get("content", "")
                    if stream and assistant_text:
                        self.console.print(f"\n{assistant_text}\n")
                except Exception:
                    self.console.print(f"  [red]Failed: {exc}[/]")
                return ""
        except Exception as exc:
            # Route through the error-code system so we get a documented
            # [Eccc] prefix and a fix line. ALSO write the full Python
            # traceback to ~/.localcode/last_error.log so we can pinpoint
            # which line raised it — bare KeyErrors with stripped names
            # have been bouncing around for a while and we need the
            # actual call site, not just the message.
            from .errors import format_for_user
            import traceback as _tb
            try:
                from .paths import last_error_log_path
                log_path = last_error_log_path()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"=== {type(exc).__name__}: {exc!r} ===\n\n")
                    _tb.print_exc(file=f)
                # Show path relative to project root so user knows where
                # to look (the file lives in their current project's
                # `.localcode/`, not in their home dir).
                tb_hint = f" (traceback: .localcode/last_error.log)"
            except Exception:
                tb_hint = ""
            self.out.set_error(format_for_user(exc, fallback_code="E9001") + tb_hint)
            self.log.exception("Chat error")
            return ""

        # Done — clean up output
        self.out.done()

        self.session.last_assistant_text = assistant_text
        # Don't pollute history with garbage — model will mirror it
        is_garbage = (
            not assistant_text.strip()
            or "<unused" in assistant_text
            or len(assistant_text) > 50 and " " not in assistant_text[:100]
        )
        if not is_garbage:
            self.session.messages.append({"role": "assistant", "content": assistant_text})
        self.store.append_event(self.session, "assistant", assistant_text[:160])
        self.store.save(self.session)
        # Turn diff summary
        turn_diff = self.turn_tracker.end_turn()
        if turn_diff.changes:
            print_turn_diff(turn_diff)
        # Desktop notification for slow tasks
        notify_if_slow("localcode", f"Task complete: {user_text[:40]}", _turn_start)
        # Record in unified history
        changed_files = self.toolkit.changes.recent_files(since=changes_before)
        self.history.record_assistant_response(
            self.session.session_id, str(self.repo_root),
            assistant_text, model=self.runtime_model,
            files_changed=changed_files if changed_files else [],
            task_id=task_state.task_id,
            task_status=str(getattr(self.session.current_task, "status", "")),
            task_stage=str(getattr(self.session.current_task, "current_stage", "")),
            blocked_reason=str(getattr(self.session.current_task, "blocked_reason", "")),
            goal_type=goal_state.goal_type,
            goal_summary=goal_state.goal_summary,
            task_kind=goal_state.task_kind,
            task_slug=goal_state.task_slug,
            completion_status=str(getattr(self, "_last_turn_completion_status", "completed" if assistant_text.strip() else "empty")),
        )

        # Show file changes + context budget
        changes_after = self.toolkit.changes.change_count
        if changes_after > changes_before:
            diff = changes_after - changes_before
            self.console.print(f"  [dim]{diff} file(s) changed. Use /verify to run tests, /undo to revert.[/]")

        # Auto-compact if approaching context limit
        max_ctx = self._effective_context_chars()
        self.session.messages, was_compacted = compact_if_needed(self.session.messages, max_ctx)
        if was_compacted:
            self.console.print("[dim]  Context compacted — older messages summarized.[/]")
            self.store.save(self.session)
            # Clear server-side KV cache so stale tokens don't persist
            try:
                import httpx
                httpx.post(f"{self.config.runtime.base_url.rstrip('/')}/v1/chat/completions",
                          json={"model": self.config.runtime.model, "messages": [], "max_tokens": 1,
                                "cache_prompt": False}, timeout=5)
            except Exception:
                pass

        return assistant_text

    # Animated localcode icon — pulsates during thinking
    _SPINNER_ICONS = ["[dim green].[/]", "[green]·[/]", "[bright_green]◆[/]", "[bold bright_green]◆[/]", "[bright_green]◆[/]", "[green]·[/]"]
    _spinner_tick = 0
    _token_count = 0

    def _spinner_icon(self) -> str:
        self._spinner_tick += 1
        return self._SPINNER_ICONS[self._spinner_tick % len(self._SPINNER_ICONS)]

    def _start_status(self) -> None:
        """Pulsating status with typewriter thinking — reveals text gradually."""
        import threading
        import time as _time
        import sys as _sys
        import os as _os
        self._status_running = True
        self._status_start = _time.time()
        self._thinking_buffer = ""  # full thinking text received so far
        self._thinking_revealed = 0  # how many chars we've shown so far
        self._thinking_lock = threading.Lock()

        def _pulse():
            icons = [".", "·", "◆", "◆", "·", "."]
            tick = 0
            lines_printed = 0  # how many full lines we've already printed
            chars_shown_on_printed_lines = 0
            try:
                cols = _os.get_terminal_size().columns
            except Exception:
                cols = 70

            prefix_len = 10  # " ◆ (10s) " roughly
            indent = "    "  # 4 spaces for wrapped lines
            indent_len = len(indent)
            text_width_first = cols - prefix_len - 2
            text_width_wrap = cols - indent_len - 1
            if text_width_first < 20:
                text_width_first = 50
            if text_width_wrap < 20:
                text_width_wrap = 60

            while self._status_running:
                tick += 1
                icon = icons[tick % len(icons)]
                elapsed = _time.time() - self._status_start
                if elapsed < 60:
                    t = f"{elapsed:.0f}s"
                else:
                    t = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

                with self._thinking_lock:
                    buf = self._thinking_buffer.replace("\n", " ").strip()
                    chars_revealed = self._thinking_revealed
                    remaining = len(buf) - chars_revealed
                    if remaining > 0:
                        if chars_revealed < 40:
                            speed = 6
                        elif chars_revealed < 150:
                            speed = 3
                        else:
                            speed = 1
                        self._thinking_revealed = min(len(buf), chars_revealed + speed)
                    visible = buf[:self._thinking_revealed]

                # Text after already-printed lines
                current_text = visible[chars_shown_on_printed_lines:]
                tw = text_width_first if lines_printed == 0 else text_width_wrap

                # Check if we need to wrap to a new line
                if len(current_text) > tw:
                    full_line = current_text[:tw]
                    if lines_printed == 0:
                        pre = f"\033[32m {icon} ({t}) \033[0m"
                    else:
                        pre = indent
                    _sys.stderr.write(f"\r{pre}\033[2m{full_line}\033[0m\033[K\n")
                    _sys.stderr.flush()
                    chars_shown_on_printed_lines += tw
                    lines_printed += 1
                    current_text = visible[chars_shown_on_printed_lines:]
                    tw = text_width_wrap

                # Current partial line
                if lines_printed == 0:
                    pre = f"\033[32m {icon} ({t}) \033[0m"
                else:
                    pre = indent
                display = current_text[:tw]
                _sys.stderr.write(f"\r{pre}\033[2m{display}\033[0m\033[K")
                _sys.stderr.flush()
                _time.sleep(0.04)

        self._status_thread = threading.Thread(target=_pulse, daemon=True)
        self._status_thread.start()

    def _feed_thinking(self, chunk: str) -> None:
        """Feed a new thinking chunk — the pulse thread reveals it gradually."""
        with self._thinking_lock:
            self._thinking_buffer += chunk

    def _stop_status(self) -> None:
        """Stop and clear."""
        import sys as _sys
        self._status_running = False
        if hasattr(self, "_status_thread"):
            self._status_thread.join(timeout=1)
        _sys.stderr.write("\r\033[K")
        _sys.stderr.flush()
        self._thinking_buffer = ""
        self._thinking_revealed = 0

    def _run_tool_loop_streaming(self, composed_messages: list[dict], stream: bool = True, _indicator: ThinkingIndicator | None = None) -> str:
        """Tool-calling loop with clean display."""
        working_messages = composed_messages

        # Route tools based on user message
        user_text = ""
        for msg in reversed(composed_messages):
            if msg.get("role") == "user":
                user_text = str(msg.get("content", ""))
                break
        routing = route_tools(
            user_text,
            self.toolkit.list_tool_names(),
            conversation_history=self.session.messages[-4:],
            online=is_online(),
        )
        if self.logger:
            self.logger.log_user_input(user_text, intent=",".join(routing.intents))
        if not routing.tool_names:
            return self._run_stream_simple(composed_messages, stream)
        if stream:
            route_bits = ", ".join(routing.intents[:3]) or "chat"
            tool_bits = ", ".join(routing.tool_names[:3])
            self.out.print_info(f"route: {route_bits}")
            if tool_bits:
                self.out.print_info(f"tool lane: {tool_bits}")

        ctx_size = self._select_task_context(user_text, routing.intents)
        output_budget = self._select_task_output_budget(user_text, routing.intents)

        # ── Speed optimization: speculative pre-execution ──
        # Start running predicted tools NOW while model thinks
        if hasattr(self, '_spec_executor'):
            self._spec_executor.predict_and_prefetch(user_text, routing.tool_names)

        # DIRECT FILE EDIT: if intent is edit and we can identify the file,
        # skip the model tool-call loop entirely. Read file, ask model for
        # updated content, write it. One shot. No multi-round failures.
        # Also trigger if recent conversation was about editing a file
        recent_edit_file = self._get_recent_edit_file()
        is_question = "?" in user_text
        is_edit_intent = ("file_edit" in routing.intents or "file_write" in routing.intents) and not is_question
        is_followup_edit = (
            bool(recent_edit_file)
            and not is_edit_intent
            and not is_question
            and any(w in user_text.lower() for w in [
                "add", "change", "remove", "update", "make",
                "modify", "replace", "delete", "insert",
                "improve", "fix", "refactor", "optimize", "rewrite",
                "just", "dunno", "better", "clean",
            ])
        )
        if (is_edit_intent or is_followup_edit) and self.profile.feature_variant in ("compact", "balanced"):
            if stream and (recent_edit_file if is_followup_edit else True):
                target = recent_edit_file if is_followup_edit else "resolved from request"
                self.out.print_info(f"edit path: direct file edit ({target})")
            result = self._direct_file_edit(
                user_text, stream,
                force_file=recent_edit_file if is_followup_edit else None,
            )
            if result is not None:
                return result

        use_minimal = self.profile.feature_variant == "compact"
        all_schemas = self.toolkit.schemas(minimal=use_minimal)
        tools = [t for t in all_schemas if t["function"]["name"] in routing.tool_names]
        # Use centralized output manager
        out = self.out

        try:
            consecutive_errors = 0
            last_tool_call = ""
            max_rounds = 20  # safety cap — agent has none but we need one for local models
            for _round in range(max_rounds):
                thinking_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls_found: list[dict] = []
                content_started = False
                round_started = time.time()
                first_token_s: float | None = None

                # ── Thinking mode ──
                # Reasoning mode (/switch): think on every round
                # Fast mode: never think
                from .thinking import should_use_thinking
                goal_type = infer_goal_state(user_text).goal_type
                use_think = should_use_thinking(
                    self.config.runtime.laptop_26b_runtime_mode,
                    self.config.runtime.internal_thinking_mode,
                    goal_type=goal_type,
                    task_stage=getattr(getattr(self.session, "current_task", None), "current_stage", ""),
                    user_text=user_text,
                )
                round_budget = output_budget if _round == 0 else min(800, output_budget)
                for event in self.engine.stream_chat_events(
                    working_messages,
                    tools=tools,
                    think=use_think,
                    num_ctx=ctx_size,
                    num_predict=round_budget,
                ):
                    if event["type"] == "thinking":
                        chunk = str(event["content"])
                        thinking_parts.append(chunk)
                        if stream:
                            out.feed_thinking(chunk)
                            peek = " ".join("".join(thinking_parts).split())[-120:]
                            if peek:
                                out.set_thinking_peek(peek)
                    elif event["type"] == "content":
                        chunk = str(event["content"])
                        # Filter special tokens
                        if "<|" in chunk or "|>" in chunk:
                            from .tool_parsing import parse_tool_calls as _ptc
                            parsed = _ptc(chunk)
                            if parsed.has_tools:
                                tool_calls_found = parsed.to_ollama_format()
                                break
                            import re as _re
                            chunk = _re.sub(r'<\|[^>]*\|>', '', chunk)
                        if not chunk.strip():
                            continue
                        if first_token_s is None:
                            first_token_s = time.time() - round_started
                        content_parts.append(chunk)
                        if stream:
                            out.stream(chunk)
                            content_started = True
                    elif event["type"] == "tool_calls":
                        tool_calls_found = event["tool_calls"]
                        break

                thinking = "".join(thinking_parts)
                content = "".join(content_parts).strip()
                self._record_runtime_sample(first_token_s=first_token_s, total_s=time.time() - round_started)
                self.stats.record(self.engine.last_response_meta)

                # Check if content contains a text-format tool call (model output it as text)
                if not tool_calls_found and content:
                    import re as _re
                    # Match: write_file(path='x', content='...') or similar
                    tc_match = _re.search(r'(write_file|read_file|edit_file|bash)\s*\(', content)
                    if tc_match:
                        from .tool_parsing import parse_tool_calls as _ptc
                        parsed = _ptc(content)
                        if parsed.has_tools:
                            tool_calls_found = parsed.to_ollama_format()
                            content = parsed.content  # strip tool call from display

                if not content_started and stream:
                    out.done()

                if tool_calls_found:
                    # Dedup: don't repeat the same tool call
                    call_sig = str([(t.get("function",{}).get("name",""), t.get("function",{}).get("arguments","")) for t in tool_calls_found])
                    if call_sig == last_tool_call:
                        break
                    last_tool_call = call_sig

                    # Permission check + execute each tool
                    tool_messages = []
                    for tc in tool_calls_found:
                        f = tc.get("function", {})
                        name = f.get("name", "")
                        args = f.get("arguments", {})
                        if isinstance(args, str):
                            import json as _json
                            try: args = _json.loads(args)
                            except: args = {}

                        # Lifecycle hook: pre-tool
                        hook = self.hooks.on_pre_tool_use(name, args)
                        if hook.blocked:
                            if stream:
                                out.log_tool(name, f"BLOCKED by hook: {hook.output or hook.error}")
                            tool_messages.append({"role": "tool", "content": f"Blocked by hook: {hook.output or hook.error}"})
                            continue

                        allowed, reason = self.perms.check(name, args)
                        if not allowed:
                            if stream:
                                out.log_tool(name, f"SKIPPED: {reason}")
                            tool_messages.append({"role": "tool", "content": f"Denied: {reason}"})
                            continue

                        if stream:
                            out.log_tool(name, str(args)[:60])
                        # Check speculative cache first — instant if pre-fetched
                        spec_result = self._spec_executor.get_if_ready(name, args)
                        if spec_result is not None:
                            result = {"role": "tool", "content": spec_result}
                        else:
                            result = self.toolkit._execute_one(tc)
                        is_err = result["content"].startswith("Error") or result["content"].startswith("Tool error")
                        if stream:
                            out.tool_result(result["content"][:120], error=is_err)
                        if self.logger:
                            self.logger.log_tool_call(name, args, result["content"], duration_ms=0)
                        self.hooks.on_post_tool_use(name, args, result["content"], is_err)
                        # Track file changes for turn diff
                        if name in ("write_file", "edit_file") and "path" in args:
                            self.turn_tracker.track_file(str(args["path"]))
                        tool_messages.append(result)

                    # Check for errors — if too many consecutive, break and force-edit
                    all_errors = all(
                        m["content"].startswith("Error") or m["content"].startswith("Tool error")
                        for m in tool_messages
                    )
                    if all_errors:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                    if stream:
                        for item in tool_messages:
                            is_err = item["content"].startswith("Error") or item["content"].startswith("Tool error")
                            out.tool_result(item["content"], error=is_err)

                    # If ANY file tool error, do the edit ourselves immediately
                    if consecutive_errors >= 1 and bool({"file_edit", "file_write"} & set(routing.intents)):
                        if stream:
                            ResponseDisplay.print_info("applying edit directly...")
                        # Find the file being edited
                        import re as _re
                        fpath = None
                        for tc in tool_calls_found:
                            args = tc.get("function", {}).get("arguments", {})
                            if "path" in args:
                                fpath = args["path"]
                                break
                        if not fpath:
                            fpath_match = _re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                            fpath = fpath_match.group(1) if fpath_match else None
                        if fpath and (self.repo_root / fpath).is_file():
                            old = (self.repo_root / fpath).read_text(errors="replace")
                            code_r = self.engine.chat_once([
                                {"role": "user", "content": f"Current {fpath}:\n```\n{old}\n```\n\nApply this change: {user_text}\n\nIMPORTANT: Make MINIMAL changes. Keep ALL existing code. Only modify what was requested.\nReturn the COMPLETE file with the small change applied. No explanation."}
                            ])
                            new = code_r.get("message", {}).get("content", "").strip()
                            new = _re.sub(r'^```\w*\n', '', new)
                            new = _re.sub(r'\n```\s*$', '', new)
                            if new and new != old:
                                self.toolkit.changes.snapshot_before(fpath, "direct_edit")
                                (self.repo_root / fpath).write_text(new)
                                if stream:
                                    out.log_tool("write_file", f"path={fpath}")
                                    out.tool_result(f"Edited {fpath}")
                                return f"I've updated {fpath} with the requested changes."
                        break

                    if stream:
                        out.start_thinking()  # restart indicator for next round

                    working_messages = [
                        *working_messages,
                        {"role": "assistant", "content": content, "tool_calls": tool_calls_found},
                        *tool_messages,
                        {"role": "user", "content": f"Good. Now do the NEXT step. Original task: {user_text}"},
                    ]
                    continue

                # Force tool if model didn't call one but should have
                if _round <= 1 and not tool_calls_found:
                    content_lower = content.lower()
                    # Check if file write/edit was expected but not done
                    file_intents = {"file_write", "file_edit"}
                    file_tools_called = any(
                        t.get("function", {}).get("name", "") in ("write_file", "edit_file")
                        for t in tool_calls_found
                    )
                    should_force = (
                        not content.strip()  # empty response
                        or any(p in content_lower for p in [
                            "i do not have access", "i cannot", "i don't have",
                            "i recommend using", "beyond my", "not able to",
                            "i am unable", "i'm unable", "cannot directly",
                            "i will now", "i will edit", "i will update",  # says it will but didn't
                        ])
                        or (bool(file_intents & set(routing.intents)) and not file_tools_called)  # edit/write intent but no tool
                    )
                    if should_force and thinking:
                        thinking_lower = thinking.lower()
                        # Try each tool — match name or name with underscores replaced
                        forced = False
                        for ts in tools:
                            tname = ts["function"]["name"]
                            readable = tname.replace("_", " ")
                            if tname in thinking_lower or readable in thinking_lower or f"`{tname}`" in thinking_lower:
                                forced_args = self._extract_args_from_thinking(tname, thinking, user_text)
                                if forced_args is None:
                                    import re as _re
                                    if tname in ("write_file", "edit_file"):
                                        # e2b can't call file tools — generate code, write ourselves
                                        path_match = _re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                                        fpath = path_match.group(1) if path_match else "output.py"
                                        existing = (self.repo_root / fpath)
                                        if existing.is_file() and tname == "edit_file":
                                            # Edit: ask model for the UPDATED version of the file
                                            old_content = existing.read_text(errors="replace")
                                            if stream:
                                                ResponseDisplay.print_info(f"generating edit for {fpath}...")
                                            code_r = self.engine.chat_once([
                                                {"role": "user", "content": f"Here is the current content of {fpath}:\n```\n{old_content}\n```\n\nModify it to: {user_text}\n\nReturn ONLY the complete updated file. No explanation."}
                                            ])
                                        else:
                                            if stream:
                                                ResponseDisplay.print_info(f"generating code for {fpath}...")
                                            code_r = self.engine.chat_once([
                                                {"role": "user", "content": f"Write ONLY the code for {fpath}. No explanation, just the code. Task: {user_text}"}
                                            ])
                                        code_content = code_r.get("message", {}).get("content", "").strip()
                                        code_content = _re.sub(r'^```\w*\n', '', code_content)
                                        code_content = _re.sub(r'\n```\s*$', '', code_content)
                                        if code_content:
                                            forced_args = {"path": fpath, "content": code_content}
                                            forced_calls = [{"function": {"name": "write_file", "arguments": forced_args}}]
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    forced_calls = [{"function": {"name": tname, "arguments": forced_args}}]
                                tool_msgs = self.toolkit.execute_tool_calls(forced_calls)
                                if stream:
                                    out.log_tool(tname, str(forced_args)[:60])
                                    for item in tool_msgs:
                                        out.tool_result(item["content"])
                                    out.start_thinking()  # restart indicator for next round
                                working_messages = [
                                    *working_messages,
                                    {"role": "assistant", "content": "", "tool_calls": forced_calls},
                                    *tool_msgs,
                                    {"role": "user", "content": f"Good. Now do the NEXT step. Original task: {user_text}"},
                                ]
                                forced = True
                                break
                        if forced and len(working_messages) > len(composed_messages):
                            continue
                        # Last resort: retry with ultra-short prompt (no system context)
                        if not forced and not content.strip():
                            retry_msgs = [
                                {"role": "system", "content": "Use tools. Always call tools."},
                                {"role": "user", "content": user_text},
                            ]
                            r = self.engine.chat_once(retry_msgs, tools=tools)
                            tc = r.get("message", {}).get("tool_calls", [])
                            if tc:
                                tool_msgs = self.toolkit.execute_tool_calls(tc)
                                if stream:
                                    for t in tc:
                                        f = t.get("function", {})
                                        out.log_tool(f.get("name", ""), str(f.get("arguments", ""))[:60])
                                    for item in tool_msgs:
                                        out.tool_result(item["content"])
                                    out.start_thinking()  # restart indicator for next round
                                working_messages = [
                                    *working_messages,
                                    r["message"],
                                    *tool_msgs,
                                    {"role": "user", "content": f"Answer: {user_text}"},
                                ]
                                continue
                            content = r.get("message", {}).get("content", "")

                # Auto-save: if response contains a code block with a filename, write it
                if content and "```" in content:
                    import re as _re
                    # Look for patterns like "save to pong.py" or "file named pong.py" in the text
                    file_hint = _re.search(r'(?:save|file|named|called|create)\s+(?:it\s+(?:as|to)\s+)?[`*]*(\w+\.(?:py|js|ts|html|css|json|md|sh))[`*]*', content, _re.IGNORECASE)
                    if file_hint:
                        fname = file_hint.group(1)
                        # Extract the code block
                        code_match = _re.search(r'```(?:\w+)?\n(.*?)```', content, _re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            if len(code) > 20:
                                fpath = self.repo_root / fname
                                self.toolkit.changes.snapshot_before(fname, "auto_write")
                                fpath.parent.mkdir(parents=True, exist_ok=True)
                                fpath.write_text(code)
                                if stream:
                                    self.console.print(f"\n  [green]✓[/] Auto-saved to {fname} ({len(code.splitlines())} lines)")

                if stream:
                    pass  # output handled by out.done()
                return content

            return ""
        finally:
            out.done()

    def _get_recent_edit_file(self) -> str | None:
        """Check if we recently edited/read a file (for follow-up edits)."""
        for event in reversed(self.session.events[-10:]):
            detail = event.get("detail", "")
            if "Updated " in detail or "read_file" in detail or "write_file" in detail:
                import re
                m = re.search(r'(\w[\w.-]*\.(?:py|js|ts|json|md|txt|html|css))', detail)
                if m:
                    return m.group(1)
        # Also check undo changelog
        if self.toolkit.changes.snapshots:
            return self.toolkit.changes.snapshots[-1].path
        return None

    def _direct_file_edit(self, user_text: str, stream: bool = True, force_file: str | None = None) -> str | None:
        """Directly edit a file without relying on model tool calls.

        For small models that can't reliably call edit_file:
        1. Find the file path in the user's message
        2. Read the file ourselves
        3. Ask the model to generate updated content
        4. Write it
        """
        import re
        out = self.out
        tool_display = ToolCallDisplay()
        response = ResponseDisplay()

        if not force_file and self._is_quality_sensitive_creation(user_text):
            return None

        # Find file path
        fpath = force_file  # use forced file if provided (follow-up edit)

        if not fpath:
            # 1. Try exact filename with extension
            path_match = re.search(r'(\w[\w.-]*\.(?:py|js|ts|json|md|txt|html|css|yaml|toml|sh))', user_text)
            fpath = path_match.group(1) if path_match else None

        if not fpath:
            # 2. Fuzzy match against actual repo files
            # Priority: exact stem > startswith > contains (longer match wins)
            skip = {"edit", "file", "make", "the", "function", "should", "help",
                    "write", "add", "create", "bro", "you", "need", "that", "use",
                    "any", "libraries", "with", "and", "classifies", "images"}
            words = [w for w in re.findall(r'\b(\w{3,})\b', user_text.lower()) if w not in skip]
            repo_files = [f for f in self.repo_root.iterdir() if f.is_file() and not f.name.startswith(".")]

            best_match = None
            best_score = 0
            for word in words:
                for f in repo_files:
                    stem = f.stem.lower()
                    if stem == word:
                        score = 100  # exact match
                    elif stem.startswith(word) or word.startswith(stem):
                        score = 50 + len(word)  # prefix match, longer = better
                    elif word in stem:
                        score = 20 + len(word)
                    else:
                        continue
                    if score > best_score:
                        best_score = score
                        best_match = f.name
            fpath = best_match

        if not fpath:
            return None

        full_path = self.repo_root / fpath
        if not full_path.is_file():
            return None

        # Read the file
        old_content = full_path.read_text(errors="replace")
        if stream:
            out.log_tool("read_file", f"path={fpath}")
            out.tool_result(f"{fpath} ({len(old_content.splitlines())} lines)")

        # If user just said a filename with no action, ask what to do
        # Otherwise, always proceed — let the model figure out the intent
        words_without_filename = re.sub(r'\b\w+\.\w{2,4}\b', '', user_text).strip()
        if len(words_without_filename) < 5:  # basically just a filename
            if stream:
                self.console.print(f"I've read {fpath}. What changes would you like me to make?")
            return ""

        # Ask model to generate updated content
        if stream:
            out.set_stage(f"editing {fpath}")
            out.start_thinking()

        old_max = self.config.runtime.max_context_chars
        try:
            num_lines = len(old_content.splitlines())
            use_patch_format = num_lines > 40 or len(old_content) > 1800

            if not use_patch_format:
                r = self.engine.chat_once([
                    {"role": "user", "content": (
                        f"Here is {fpath}:\n```\n{old_content}\n```\n\n"
                        f"Apply this change: {user_text}\n\n"
                        f"Return the COMPLETE updated file. Keep ALL existing code. No explanation."
                    )}
                ], think=False, num_predict=min(2200, max(600, len(old_content) // 2)))
            else:
                r = self.engine.chat_once([
                    {"role": "user", "content": (
                        f"Here is {fpath} ({num_lines} lines):\n```\n{old_content}\n```\n\n"
                        f"Apply this change: {user_text}\n\n"
                        f"Return ONLY the changes. For each change write:\n"
                        f"SEARCH:\n```\nexact old lines\n```\n"
                        f"REPLACE:\n```\nnew lines\n```\n\n"
                        f"Use 2-6 lines of context in SEARCH to match uniquely. No explanation."
                    )}
                ], think=False, num_predict=1200)

            # Restore
            self.config.runtime.max_context_chars = old_max
        finally:
            if stream:
                out.done()

        raw_response = r.get("message", {}).get("content", "").strip()

        # For large files, parse SEARCH/REPLACE blocks and apply them
        if use_patch_format and ("SEARCH:" in raw_response or "SEARCH\n" in raw_response):
            new_content = old_content
            # Extract search/replace pairs from the response
            search_blocks = re.split(r'SEARCH:\s*\n```[^\n]*\n', raw_response)
            applied = 0
            for block in search_blocks[1:]:
                # Find the search text (between first ``` pair)
                search_end = block.find("\n```")
                if search_end < 0:
                    continue
                search_text = block[:search_end]
                # Find the replace text
                replace_match = re.search(r'REPLACE:\s*\n```[^\n]*\n(.*?)\n```', block, re.DOTALL)
                if replace_match:
                    replace_text = replace_match.group(1)
                    if search_text.strip() and search_text.strip() in new_content:
                        new_content = new_content.replace(search_text.strip(), replace_text.strip(), 1)
                        applied += 1
            if applied == 0:
                # Fallback: treat entire response as complete file
                new_content = re.sub(r'^```\w*\n', '', raw_response)
                new_content = re.sub(r'\n```\s*$', '', new_content).strip()
            elif stream:
                self.console.print(f"  [dim]{applied} edit(s) applied[/]")
        else:
            new_content = raw_response
        # Strip markdown fences
        new_content = re.sub(r'^```\w*\n', '', new_content)
        new_content = re.sub(r'\n```\s*$', '', new_content)
        new_content = re.sub(r'^```$', '', new_content, flags=re.MULTILINE).strip()

        if not new_content or new_content == old_content:
            self.console.print("No changes generated.")
            return ""

        # Write the file
        self.toolkit.changes.snapshot_before(fpath, "direct_edit")
        full_path.write_text(new_content)

        # Show what changed
        import difflib
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{fpath}", tofile=f"b/{fpath}", lineterm=""))

        self.console.print(f"  [green]✓[/] Updated {fpath} ({len(new_lines)} lines)")

        # Show compact diff summary
        added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        if added or removed:
            self.console.print(f"  [green]+{added}[/] [red]-{removed}[/] lines changed")
            # Show first few changes
            changes_shown = 0
            for line in diff:
                if changes_shown >= 8:
                    self.console.print("  [dim]  ... and more[/]")
                    break
                if line.startswith("+") and not line.startswith("+++"):
                    self.console.print(f"  [green]{line}[/]")
                    changes_shown += 1
                elif line.startswith("-") and not line.startswith("---"):
                    self.console.print(f"  [red]{line}[/]")
                    changes_shown += 1

        # LSP diagnostics — ruff/pyflakes/py_compile
        if fpath.endswith(".py"):
            from .lsp import get_diagnostics
            diags = get_diagnostics(full_path)
            if not diags:
                self.console.print("  [green]✓[/] No issues found")
            else:
                for d in diags[:5]:
                    color = "red" if d.severity == "error" else "yellow"
                    self.console.print(f"  [{color}]{d}[/]")

        if diff:
            changed_chunks = max(1, sum(1 for line in diff if line.startswith("@@")))
            self.console.print(f"\n  [dim]{fpath}: {changed_chunks} changed region(s), +{added}/-{removed} lines[/]")

        return f"Updated {fpath}."

    @staticmethod
    def _extract_args_from_thinking(tool_name: str, thinking: str, user_text: str) -> dict | None:
        """Extract tool arguments from thinking. Returns None if can't extract reliably."""
        import re
        if tool_name == "current_datetime":
            return {}
        if tool_name in ("web_search",):
            return {"query": user_text}
        if tool_name == "write_file":
            # Only force write_file if we found ACTUAL code in thinking
            code_match = re.search(r'```(?:\w+)?\n(.*?)```', thinking, re.DOTALL)
            if code_match:
                path_match = re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
                path = path_match.group(1) if path_match else "output.py"
                return {"path": path, "content": code_match.group(1).strip()}
            # No code found — DON'T force with garbage, return None to skip
            return None
        if tool_name == "read_file":
            path_match = re.search(r'(\w+\.(?:py|js|ts|json|md|txt|html|css))', user_text + " " + thinking)
            return {"path": path_match.group(1)} if path_match else None
        if tool_name == "bash":
            cmd_match = re.search(r'`([^`]+)`', thinking)
            return {"command": cmd_match.group(1)} if cmd_match else None
        if tool_name == "grep":
            return {"pattern": user_text.split()[-1] if user_text.split() else None}
        return None

    def _run_stream_simple(self, composed_messages: list[dict], stream: bool = True) -> str:
        """Simple streaming without tools."""
        chunks: list[str] = []
        started_content = False
        out = self.out
        started_at = time.time()
        first_token_s: float | None = None
        max_tokens = 2048
        if stream:
            out.set_thinking_peek("contacting model and waiting for first tokens")

        try:
            for event in self.engine.stream_chat_events(composed_messages, num_predict=max_tokens):
                if event["type"] == "thinking":
                    if stream:
                        out.feed_thinking(str(event["content"]))
                    continue
                if event["type"] == "content":
                    chunk = str(event["content"])
                    # Filter raw Gemma 4 special tokens
                    if "<|" in chunk or "|>" in chunk:
                        import re as _re
                        chunk = _re.sub(r'<\|[^>]*\|>', '', chunk)
                    if not chunk:
                        continue
                    if first_token_s is None:
                        first_token_s = time.time() - started_at
                        if stream:
                            out.set_thinking_peek("model responded, streaming answer")
                    chunks.append(chunk)
                    if stream:
                        out.stream(chunk)
                        started_content = True
            if stream:
                out.done()
        finally:
            self._record_runtime_sample(first_token_s=first_token_s, total_s=time.time() - started_at)

        return "".join(chunks).strip()

    def _select_task_context(self, user_text: str, intents: list[str]) -> int:
        text = user_text.lower()
        simple_intents = {"time", "chat", "git", "web"}
        if intents and set(intents) <= simple_intents:
            return 4096

        quality_task = self._is_quality_sensitive_creation(text)
        single_file = bool(Path(text.split()[-1]).suffix) or any(ext in text for ext in (".py", ".js", ".ts", ".html", ".css"))
        if quality_task and ("game" in text or "app" in text):
            return 8192 if single_file else 12288

        if set(intents) & {"file_edit", "file_write", "search_code", "quality_create"}:
            return 8192 if single_file else 12288

        return 6144

    def _select_task_output_budget(self, user_text: str, intents: list[str]) -> int:
        text = user_text.lower()
        if set(intents) & {"quality_create"}:
            return 2200
        if set(intents) & {"file_edit"}:
            return 1200
        if set(intents) & {"file_write"}:
            if any(ext in text for ext in (".py", ".js", ".ts", ".html", ".css")):
                return 1600
            return 1200
        if set(intents) & {"search_code", "git"}:
            return 700
        return 1200

    def _maybe_auto_verify(self) -> None:
        """Auto-suggest running tests after file changes in interactive mode."""
        verify_cmd = guess_verify_command(self.repo_root)
        if not verify_cmd:
            return
        change_count = self.toolkit.changes.change_count
        self.console.print(
            f"\n  [dim]{change_count} file(s) changed.[/] "
            f"[yellow]Run tests?[/] [dim]({verify_cmd})[/]"
        )
        try:
            answer = self.prompt.prompt("  [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer in {"y", "yes"}:
            self.console.print()
            output, code = run_verification(self.repo_root, bias=self.profile.verification_bias)
            style = "green" if code == 0 else "red"
            self.console.print(Panel(
                output[-4000:],
                title=f"[{style}]verification exit={code}[/]",
                border_style=style,
            ))
            self.store.append_event(self.session, "auto_verify", f"exit={code}")

    def _render_thinking(self, thinking_text: str) -> None:
        mode = self.config.ui.thinking_mode
        if mode == "hidden" or not thinking_text:
            return
        self._thinking_tick += 1
        # Simple inline spinner — the original `thinking_frame` helper was
        # removed from the codebase but this path still exists in the legacy
        # CLI renderer. Keep behavior functional without the dep.
        _SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = _SPIN[self._thinking_tick % len(_SPIN)]
        label = "thinking..."
        if mode == "summary":
            one_line = " ".join(thinking_text.split())
            preview = one_line[:240] + ("..." if len(one_line) > 240 else "")
            body = f"  {frame}  [dim]{label}[/]\n  [dim]{preview}[/]"
        else:
            # full mode
            body = f"  {frame}  [dim]{label}[/]\n  {thinking_text[-1800:]}"
        self.console.print(Panel(body, title="[dim]thinking[/]", border_style="dim"))

    def _approve_tool_calls(self, summaries: list[str]) -> bool:
        tool_names = [summary.split("(", 1)[0] for summary in summaries]
        decisions = [self._permission_decision(name) for name in tool_names]
        if decisions and all(decision == "allow" for decision in decisions):
            return True
        if decisions and any(decision == "deny" for decision in decisions):
            return False
        decision = self.approvals.review(
            "Tool Calls",
            [ApprovalItem(label=name, detail=summary) for name, summary in zip(tool_names, summaries)],
            allow_repo_option=True,
        )
        if decision == "session-allow":
            self.session_allows.update(tool_names)
            return True
        if decision == "repo-allow":
            for name in tool_names:
                self.permissions.allow(name)
            return True
        return decision == "allow"

    def _review_patch(self, diff_text: str) -> str | None:
        files = parse_diff(diff_text)
        if not files:
            return None
        selected_files = []
        for file in files:
            file_panel = Panel(
                "\n".join(
                    [f"{hunk.header}\n" + "\n".join(hunk.lines[:20]) + ("\n..." if len(hunk.lines) > 20 else "") for hunk in file.hunks]
                )[:3000],
                title=f"{file.new_path} ({len(file.hunks)} hunks)",
            )
            self.console.print(file_panel)
            mode = self.prompt.prompt("Apply [a]ll hunks, [s]elect hunks, or [n]one? ").strip().lower()
            if mode in {"n", "none"}:
                continue
            if mode in {"s", "select"}:
                chosen_hunks = []
                for idx, hunk in enumerate(file.hunks, start=1):
                    preview = hunk.header + "\n" + "\n".join(hunk.lines[:16])
                    self.console.print(Panel(preview, title=f"{file.new_path} hunk {idx}"))
                    if self._confirm("Include this hunk?"):
                        chosen_hunks.append(hunk)
                if chosen_hunks:
                    file.hunks = chosen_hunks
                    selected_files.append(file)
                continue
            selected_files.append(file)
        if not selected_files:
            return None
        reviewed = build_diff(selected_files)
        preview = reviewed[:1200] + ("..." if len(reviewed) > 1200 else "")
        self.console.print(Panel(preview, title="Selected Patch"))
        return reviewed if self._confirm("Keep this reviewed patch?") else None

    def _retrieval_context(self, query: str) -> str:
        if not query.strip():
            return ""
        budget = self.profile.retrieval_budget

        # Leg 1 — lexical chunk index (keyword match, exact identifiers).
        # Lazily built; always available. Failures degrade to empty, not fatal.
        if load_index(self.repo_root) is None:
            try:
                count, _ = build_index(self.repo_root)
                self.store.append_event(self.session, "index_build", f"{count} files")
            except Exception:
                pass
        try:
            lexical = search_index(self.repo_root, query, limit=budget) or []
        except Exception:
            lexical = []

        # Leg 2 — semantic index (embeddings: intent match, not just keywords).
        # The embedding index used to be dead weight: built nowhere, queried
        # nowhere. We now query it when it exists and trigger a one-time
        # BACKGROUND build when it doesn't (build can take 10-30s / pull a
        # model, so never on the chat hot path). First turn is lexical-only;
        # later turns pick up the semantic leg once the index lands on disk.
        semantic: list[dict] = []
        try:
            if self.embedding_search.is_indexed():
                for r in self.embedding_search.search(query, top_k=budget):
                    semantic.append({
                        "path": r.file,
                        "chunk_id": r.start_line,
                        "preview": r.preview,
                    })
            else:
                self._ensure_embedding_index()
        except Exception:
            semantic = []

        # Merge: interleave semantic + lexical, dedup by file path (keeps
        # coverage diverse within a small budget), cap at the budget. Falls
        # back to pure lexical when the semantic leg is empty — byte-identical
        # to the old behaviour in the no-index / deps-missing case.
        from itertools import zip_longest
        merged: list[dict] = []
        seen: set[str] = set()
        for s, l in zip_longest(semantic, lexical):
            for item in (s, l):
                if item is None:
                    continue
                path = item.get("path")
                if path in seen:
                    continue
                seen.add(path)
                merged.append(item)
                if len(merged) >= budget:
                    break
            if len(merged) >= budget:
                break
        if not merged:
            return ""
        lines = [f"{item['path']}#chunk{item['chunk_id']}: {item['preview']}" for item in merged]
        return "\n\n".join(lines)

    def _ensure_embedding_index(self) -> None:
        """Kick off a one-time, non-blocking background build of the semantic
        embedding index. Embedding builds can take 10-30s and may download a
        model on first run, so we never do this on the chat hot path. Guarded
        by a once-flag; failures (missing deps, etc.) are swallowed so the
        agent silently stays on the lexical leg. With `sentence-transformers`
        installed this is true neural retrieval; otherwise it falls back to a
        sklearn TF-IDF index (still better ranking than raw keyword match)."""
        if getattr(self, "_embed_build_started", False):
            return
        self._embed_build_started = True
        import threading

        def _build() -> None:
            try:
                self.embedding_search.build_index()
            except Exception:
                pass

        threading.Thread(target=_build, name="lc-embed-index", daemon=True).start()

    def plan_for_task(self, task: str):
        """Simplified plan — planner module removed."""
        return None

    def planner_checkpoint_hint(self, _checkpoint: str, _task: str, _context_snippet: str = ""):
        """Planner hints removed — always returns None."""
        return None

    def maybe_escalate_for_task(self, task: str) -> None:
        if not self.config.runtime.escalation_enabled:
            return
        note = self.plan_for_task(task)
        if note is None or getattr(note, 'complexity', None) != "high":
            return
        if self.profile.key == "gemma4-e2b":
            target = GEMMA_PROFILES["gemma4-e4b"]
        elif self.profile.key == "gemma4-e4b":
            target = GEMMA_PROFILES["gemma4-12b"]
        elif self.profile.key == "gemma4-12b":
            target = GEMMA_PROFILES["gemma4-26b-moe"]
        else:
            return
        self.profile = target
        self.runtime_model = target.default_model
        self.config.runtime.profile = target.key
        self.config.runtime.model = target.default_model
        self.engine = LocalCodeRuntimeGateway(self.config.runtime)
        save_config(self.config)
        self.store.append_event(self.session, "model_escalation", f"{target.key} for {task[:80]}")
        self.render_agent_state("escalation", f"Switched to {target.display_name} for a broader task")

    def _should_orchestrate(self, user_text: str) -> bool:
        """Detect if a task is complex enough to need multi-agent orchestration.

        Auto-triggers orchestrator for tasks that:
        - Involve creating multi-file projects ("make an app", "build a game")
        - Require multiple distinct steps ("set up X, then configure Y, then...")
        - Are explicitly complex ("full", "complete", "entire", "from scratch")

        Simple edits, questions, and single-file tasks stay in the normal loop.
        """
        text = user_text.lower().strip()

        # Never orchestrate questions or short messages
        if "?" in text or len(text) < 20:
            return False

        # Check for multi-step creation tasks
        creation_words = ("make", "build", "create", "set up", "implement", "develop")
        complex_targets = ("app", "game", "project", "website", "server", "api",
                          "system", "framework", "tool", "cli", "dashboard")
        has_creation = any(w in text for w in creation_words)
        has_complex_target = any(w in text for w in complex_targets)

        # Check for explicit complexity signals
        complexity_signals = ("from scratch", "full", "complete", "entire",
                            "multiple files", "multi-file", "step by step",
                            "and then", "after that", "set up everything")
        has_complexity = any(s in text for s in complexity_signals)

        # Multi-step indicators (commas listing steps, "and" chaining)
        import re
        has_multiple_actions = len(re.findall(r'\b(?:and|then|also|plus)\b', text)) >= 2

        return (has_creation and has_complex_target) or has_complexity or has_multiple_actions or self._is_quality_sensitive_creation(text)

    @staticmethod
    def _is_quality_sensitive_creation(user_text: str) -> bool:
        text = user_text.lower().strip()
        creation_words = ("make", "build", "create", "generate", "implement")
        quality_words = (
            "app", "game", "website", "dashboard", "ui", "clone",
            "polish", "authentic", "look like", "feel like", "sonic",
            "beautiful", "high quality", "fidelity", "playable",
        )
        return any(w in text for w in creation_words) and any(w in text for w in quality_words)

    def _generate_task_name(self, user_text: str) -> str:
        """Generate a short dynamic task name from user input (like agent)."""
        import re
        text = user_text.strip().lower()
        # Quick keyword-based task naming — no LLM call needed
        if any(w in text for w in ("fix", "bug", "error", "broken")):
            f = re.search(r'(\w+\.\w{2,4})', text)
            return f"fixing {f.group(1)}" if f else "fixing bug"
        if any(w in text for w in ("make", "create", "build", "write", "generate")):
            # Try to extract what they're making
            for pattern in [r'(?:make|create|build|write)\s+(?:a\s+)?(.{3,25}?)(?:\s+that|\s+which|\s+for|$)',
                           r'(?:make|create|build)\s+(.{3,20}?)\.?\s*$']:
                m = re.search(pattern, text)
                if m:
                    return f"creating {m.group(1).strip()}"
            return "creating file"
        if any(w in text for w in ("edit", "change", "update", "modify", "refactor")):
            f = re.search(r'(\w+\.\w{2,4})', text)
            return f"editing {f.group(1)}" if f else "editing code"
        if any(w in text for w in ("search", "find", "look", "where")):
            return "searching codebase"
        if any(w in text for w in ("test", "run", "execute")):
            return "running tests"
        if any(w in text for w in ("explain", "what", "how", "why")):
            return "analyzing"
        if any(w in text for w in ("git", "commit", "push", "branch")):
            return "git operation"
        if len(text) > 60:
            return text[:40].rsplit(" ", 1)[0] + "..."
        return text[:40] if len(text) > 5 else ""

    def _maybe_use_fast_model(self, _user_text: str) -> None:
        """Disabled — always use 26B. The 4B draft model can't do tools or thinking."""
        return

    def _maybe_switch_tier(self, user_text: str) -> None:
        """Two-tier model routing: fast model for simple tasks, smart for complex.

        e4b (28 tok/s) handles: simple questions, time, git status, web search
        26B (8 tok/s) handles: code generation, file editing, multi-step tasks
        """
        if not self.config.runtime.escalation_enabled:
            return

        # Check what models are available
        try:
            available = self.engine.list_models()
        except Exception:
            return

        has_26b = any("26b" in m.lower() or "gemma26b" in m.lower() for m in available)
        has_e4b = any("e4b" in m.lower() for m in available)

        if not (has_26b and has_e4b):
            return  # Can't do two-tier without both models

        # Determine complexity from intent
        routing = route_tools(
            user_text,
            self.toolkit.list_tool_names(),
            online=is_online(),
        )

        simple_intents = {"time", "chat", "git", "web", "general"}
        complex_intents = {"file_edit", "file_write", "search_code"}

        is_complex = bool(set(routing.intents) & complex_intents) or len(user_text) > 200

        # Pick the right model
        if is_complex and "e4b" in self.runtime_model:
            # Escalate to 26B for complex tasks
            target_model = next((m for m in available if "26b" in m.lower() or "gemma26b" in m.lower()), None)
            if target_model:
                self.out.print_info(f"↑ escalating to {target_model}")
                self.config.runtime.model = target_model
                self.engine = LocalCodeRuntimeGateway(self.config.runtime)
                self.runtime_model = target_model
        elif not is_complex and "26b" in self.runtime_model.lower():
            # De-escalate to e4b for simple tasks (3x faster)
            target_model = next((m for m in available if "e4b" in m.lower()), None)
            if target_model:
                self.out.print_info(f"↓ fast mode: {target_model}")
                self.config.runtime.model = target_model
                self.engine = LocalCodeRuntimeGateway(self.config.runtime)
                self.runtime_model = target_model

    def _effective_context_chars(self) -> int:
        base = min(self.config.runtime.max_context_chars, self.profile.recommended_context_chars)
        if self.profile.key == "gemma4-26b-laptop":
            base = min(base, 24000)
        if self.config.runtime.mode == "fast":
            return max(6000, min(base, 12000))
        if self.config.runtime.mode == "deep":
            return max(base, min(self.config.runtime.max_context_chars, base + 12000))
        if self.config.runtime.cache_policy == "rolling":
            cap = 14000 if self.profile.key == "gemma4-26b-laptop" else 16000
            return max(8000, min(base, cap))
        if self.config.runtime.low_overhead_mode:
            return max(6000, min(base, 12000))
        return base

    def _adapt_to_prompt(self, prompt: str) -> None:
        if not self.config.runtime.adaptive_execution:
            return
        lowered = prompt.lower()
        if self.config.runtime.mode == "fast" and any(word in lowered for word in ("large", "architecture", "migration", "multi-file", "refactor")):
            self.config.runtime.max_context_chars = max(self.config.runtime.max_context_chars, 18000)
        if len(prompt) > 3000:
            self.config.runtime.max_context_chars = min(self.config.runtime.max_context_chars, self.profile.recommended_context_chars)

    def _apply_cache_policy(self) -> None:
        window = max(8, self.config.runtime.rolling_window_messages)
        if self.config.runtime.cache_policy == "rolling":
            self.session.messages = self.session.messages[-window:]
            return
        if self.config.runtime.cache_policy == "tight":
            self.session.messages = self.session.messages[-max(8, window // 2):]

    def _draft_assist(self, user_text: str) -> str:
        draft_model = self.config.runtime.draft_model
        if not draft_model or draft_model == self.runtime_model:
            return ""
        try:
            draft_runtime = replace(
                self.config.runtime,
                model=draft_model,
                temperature=0.0,
                max_context_chars=min(6000, self._effective_context_chars()),
            )
            draft_engine = LocalCodeRuntimeGateway(draft_runtime)
            response = draft_engine.chat_once(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are LocalCode's draft lane."
                            "Return a short draft of the likely answer shape, files, tools, or checks needed. "
                            "Keep it under 120 words."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ]
            )
            text = response.get("message", {}).get("content", "").strip()
            return text[:500]
        except Exception:
            return ""

    def _permission_decision(self, action: str) -> str | None:
        if action in self.session_allows:
            return "allow"
        return self.permissions.decision_for(action)

    def _approve_action(self, action: str, detail: str) -> bool:
        decision = self._permission_decision(action)
        if decision == "allow":
            return True
        if decision == "deny":
            return False
        result = self.approvals.review(
            action.replace("_", " ").title(),
            [ApprovalItem(label=action, detail=detail)],
            allow_repo_option=True,
        )
        if result == "session-allow":
            self.session_allows.add(action)
            return True
        if result == "repo-allow":
            self.permissions.allow(action)
            return True
        return result == "allow"

    def close(self) -> None:
        if hasattr(self, '_ask_indicator'):
            self._ask_out.done()
        # Stop the output manager indicator thread
        if hasattr(self, 'out'):
            self.out._indicator_running = False
            self.out._stop_indicator()
        self.toolkit.close()
        self.engine.close()
