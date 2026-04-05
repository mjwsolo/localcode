"""Multi-agent orchestrator — Planner/Worker/Reviewer pipeline.

Architecture (inspired by Codex teams, Aider Architect, OpenCode ORCH):

  26B PLANNER → generates task DAG with dependencies
       ↓
  e2b WORKERS → execute steps in parallel waves (independent steps concurrent)
       ↓
  26B REVIEWER → validates result, scores 0-100%, loops if < threshold

Context layers:
  - Global: repo structure, file tree, git status (shared, read-only)
  - Local: each worker sees ONLY files relevant to its task
  - Merge: after each wave, worker outputs merge into global context

Key design decisions:
  - Workers NEVER decide what to do. They execute mechanical edits.
  - 26B does ALL planning and reviewing. Workers are dumb but fast.
  - File-based coordination (like OpenCode ORCH): numbered step files
  - Push-based completion (like OpenClaw): workers announce when done
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import GemApp


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class TaskStep:
    """A single step in the execution plan."""
    id: int
    description: str
    file_targets: list[str] = field(default_factory=list)  # files to read/modify
    depends_on: list[int] = field(default_factory=list)     # step IDs this depends on
    status: str = "pending"   # pending | running | done | error | skipped
    result: str = ""
    error: str = ""
    worker_model: str = ""    # which model executed this


@dataclass
class TaskPlan:
    """A DAG of steps with dependencies."""
    task: str
    steps: list[TaskStep] = field(default_factory=list)
    review_score: int = 0
    review_notes: str = ""

    def get_ready_steps(self) -> list[TaskStep]:
        """Get steps whose dependencies are all done (ready to execute)."""
        done_ids = {s.id for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in done_ids for d in s.depends_on)
        ]

    def get_waves(self) -> list[list[TaskStep]]:
        """Compute execution waves — groups of steps that can run in parallel."""
        waves: list[list[TaskStep]] = []
        done_ids: set[int] = set()
        remaining = [s for s in self.steps if s.status == "pending"]

        while remaining:
            wave = [s for s in remaining if all(d in done_ids for d in s.depends_on)]
            if not wave:
                # Deadlock or circular dependency — force remaining as sequential
                wave = [remaining[0]]
            waves.append(wave)
            done_ids.update(s.id for s in wave)
            remaining = [s for s in remaining if s.id not in done_ids]

        return waves

    @property
    def is_done(self) -> bool:
        return all(s.status in ("done", "error", "skipped") for s in self.steps)

    @property
    def summary(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        total = len(self.steps)
        return f"{done}/{total} steps done, review: {self.review_score}/100"


# ── Context layers ──────────────────────────────────────────────────

@dataclass
class GlobalContext:
    """Shared context available to all agents. Updated after each wave."""
    repo_root: Path
    file_tree: str = ""           # ls-like file listing
    git_status: str = ""          # current git status
    existing_files: dict[str, str] = field(default_factory=dict)  # path → content
    completed_steps: list[str] = field(default_factory=list)      # summaries of done steps

    def snapshot(self) -> str:
        """Create a text snapshot for injection into prompts."""
        parts = [f"Repository: {self.repo_root}"]
        if self.git_status:
            parts.append(f"Git status:\n{self.git_status}")
        if self.completed_steps:
            parts.append("Completed so far:\n" + "\n".join(f"  - {s}" for s in self.completed_steps))
        return "\n\n".join(parts)


@dataclass
class LocalContext:
    """Per-worker context — only the files relevant to this task."""
    step: TaskStep
    file_contents: dict[str, str] = field(default_factory=dict)  # path → content
    global_snapshot: str = ""  # read-only global context


# ── Orchestrator ────────────────────────────────────────────────────

class Orchestrator:
    """Multi-agent orchestrator with parallel execution and layered context.

    Auto-triggered for complex tasks (multi-file creation, multi-step chains).
    Falls back gracefully to single-agent loop on failure.
    """

    REVIEW_THRESHOLD = 80  # minimum score to pass review
    MAX_REVIEW_LOOPS = 2   # max planner→worker→review cycles

    def __init__(self, app: "GemApp") -> None:
        self.app = app
        self.out = app.out

    def run(self, task: str) -> TaskPlan:
        """Execute the full pipeline: plan → workers → review → (loop if needed)."""
        import sys

        # Phase 1: Build global context
        global_ctx = self._build_global_context()

        for attempt in range(self.MAX_REVIEW_LOOPS):
            # Phase 2: Plan (26B generates task DAG)
            self.out.set_stage("planning")
            plan = self._plan(task, global_ctx)

            if not plan.steps:
                self.out.print_info("no steps generated, falling back to direct execution")
                return plan

            # Show the plan to the user
            sys.stdout.write(f"\n\033[1m  Plan ({len(plan.steps)} steps):\033[0m\n")
            waves = plan.get_waves()
            for wave_num, wave in enumerate(waves, 1):
                parallel_tag = f" \033[2m(parallel)\033[0m" if len(wave) > 1 else ""
                sys.stdout.write(f"  \033[2mwave {wave_num}:{parallel_tag}\033[0m\n")
                for step in wave:
                    deps = f" \033[2m← after {step.depends_on}\033[0m" if step.depends_on else ""
                    sys.stdout.write(f"    [{step.id}] {step.description[:70]}{deps}\n")
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Phase 3: Execute waves (workers in parallel)
            for wave_num, wave in enumerate(waves, 1):
                self.out.set_stage(f"wave {wave_num}/{len(waves)}")
                self.out.print_info(f"wave {wave_num}/{len(waves)}: {len(wave)} worker{'s' if len(wave) > 1 else ''}")
                self._execute_wave(wave, global_ctx)
                # Merge: update global context with results
                for step in wave:
                    if step.status == "done":
                        global_ctx.completed_steps.append(
                            f"[{step.id}] {step.description}: {step.result[:100]}"
                        )
                        # Re-read modified files into global context
                        for fpath in step.file_targets:
                            full = global_ctx.repo_root / fpath
                            if full.is_file():
                                try:
                                    global_ctx.existing_files[fpath] = full.read_text(errors="replace")
                                except Exception:
                                    pass

            # Phase 4: Review (26B checks quality)
            self.out.set_stage("reviewing")
            plan = self._review(task, plan, global_ctx)

            if plan.review_score >= self.REVIEW_THRESHOLD:
                self.out.print_info(f"review passed ({plan.review_score}/100)")
                break
            else:
                self.out.print_info(
                    f"review {plan.review_score}/100 — replanning..."
                )
                task = f"{task}\n\nPrevious attempt feedback:\n{plan.review_notes}"

        return plan

    # ── Phase implementations ───────────────────────────────────────

    def _build_global_context(self) -> GlobalContext:
        """Build initial global context from the repo."""
        from .context import git_status, list_repo_files

        ctx = GlobalContext(repo_root=self.app.repo_root)

        try:
            ctx.git_status = git_status(self.app.repo_root)
        except Exception:
            ctx.git_status = ""

        try:
            files = list_repo_files(self.app.repo_root, max_files=100)
            ctx.file_tree = "\n".join(files[:100])
        except Exception:
            ctx.file_tree = ""

        return ctx

    def _plan(self, task: str, global_ctx: GlobalContext) -> TaskPlan:
        """Use 26B to generate a task DAG."""
        prompt = f"""Break this task into 3-5 concrete steps. Keep it simple.

TASK: {task}

EXISTING FILES IN REPO:
{global_ctx.file_tree[:1000]}

Return a JSON array. Each step:
- "id": integer
- "description": what to do (specific: "Create X" or "Add Y to Z")
- "file_targets": file paths to create or modify
- "depends_on": step IDs that must finish first ([] if independent)

IMPORTANT:
- Keep it SIMPLE. 3-5 steps max. Don't over-engineer.
- You CAN edit existing files if the task requires it (e.g. "refactor app.py")
- Do NOT touch config files (.gitignore, pyproject.toml, CLAUDE.md, .env) unless explicitly asked
- For new projects, prefer ONE main file over many small files
- Steps modifying the same file MUST be sequential (add dependency)
- Return ONLY valid JSON, no explanation.

Example for "make a snake game":
[
  {{"id": 1, "description": "Create snake.py with complete snake game using pygame", "file_targets": ["snake.py"], "depends_on": []}},
  {{"id": 2, "description": "Create requirements.txt with pygame dependency", "file_targets": ["requirements.txt"], "depends_on": []}}
]"""

        try:
            response = self._call_planner(prompt)
            steps = self._parse_plan_json(response)
            return TaskPlan(task=task, steps=steps)
        except Exception as exc:
            self.out.print_info(f"plan parse error: {exc}")
            # Fallback: single step
            return TaskPlan(task=task, steps=[
                TaskStep(id=1, description=task, file_targets=[])
            ])

    def _execute_wave(self, wave: list[TaskStep], global_ctx: GlobalContext) -> None:
        """Execute a wave of independent steps in parallel using e2b workers."""
        if len(wave) == 1:
            self._execute_step(wave[0], global_ctx)
            return

        with ThreadPoolExecutor(max_workers=min(4, len(wave))) as pool:
            futures = {
                pool.submit(self._execute_step, step, global_ctx): step
                for step in wave
            }
            for future in as_completed(futures):
                step = futures[future]
                try:
                    future.result(timeout=180)
                except Exception as exc:
                    step.status = "error"
                    step.error = str(exc)

    def _execute_step(self, step: TaskStep, global_ctx: GlobalContext) -> None:
        """Execute a single step — Aider Architect pattern.

        Instead of relying on the worker to call tools (unreliable on small models),
        we ask the worker to GENERATE CODE, then WE write the files.
        """
        import re as _re
        step.status = "running"
        self.out.log_tool("worker", f"[{step.id}] {step.description[:60]}")

        # Build local context — only files this step needs
        file_section = ""
        for fpath in step.file_targets:
            full = global_ctx.repo_root / fpath
            if full.is_file():
                try:
                    content = full.read_text(errors="replace")
                    file_section += f"\n--- {fpath} ---\n{content[:4000]}\n"
                except Exception:
                    pass
            elif fpath in global_ctx.existing_files:
                file_section += f"\n--- {fpath} ---\n{global_ctx.existing_files[fpath][:4000]}\n"

        # Worker prompt: generate COMPLETE code, no truncation
        file_targets_str = ", ".join(step.file_targets) if step.file_targets else "the file"
        if file_section:
            prompt = (
                f"TASK: {step.description}\n\n"
                f"EXISTING FILES:{file_section}\n\n"
                f"Return the COMPLETE updated file. Do NOT truncate or abbreviate.\n"
                f"Write the ENTIRE file from start to finish.\n\n"
                f"FILE: {file_targets_str}\n```\n"
            )
        else:
            prompt = (
                f"TASK: {step.description}\n\n"
                f"Write the COMPLETE file. Do NOT truncate or use '# ... rest of code'.\n"
                f"Write EVERY line from start to finish.\n\n"
                f"FILE: {file_targets_str}\n```\n"
            )

        try:
            response = self._call_worker(prompt, step)

            # Parse FILE: blocks and write them
            files_written = []
            # Match FILE: path followed by code block
            pattern = r'FILE:\s*(\S+)\s*\n```\w*\n(.*?)```'
            matches = _re.findall(pattern, response, _re.DOTALL)

            if not matches:
                # Fallback: try to extract any code block and write to first target
                code_match = _re.search(r'```\w*\n(.*?)```', response, _re.DOTALL)
                if code_match and step.file_targets:
                    matches = [(step.file_targets[0], code_match.group(1))]

            for fpath, content in matches:
                fpath = fpath.strip()
                content = content.strip()
                if not content:
                    continue
                full_path = global_ctx.repo_root / fpath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                # Snapshot for undo
                self.app.toolkit.changes.snapshot_before(fpath, f"orch_step_{step.id}")
                full_path.write_text(content + "\n")
                files_written.append(fpath)
                # Update global context
                global_ctx.existing_files[fpath] = content

            if files_written:
                step.result = f"wrote {', '.join(files_written)}"
                step.status = "done"
                self.out.tool_result(f"[{step.id}] {step.result}")
            else:
                # Worker returned text but no parseable code — still mark done
                step.result = response[:100]
                step.status = "done"
                self.out.tool_result(f"[{step.id}] completed (no files)")

        except Exception as exc:
            step.status = "error"
            step.error = str(exc)
            self.out.tool_result(f"[{step.id}] error: {exc}", error=True)

    def _review(self, task: str, plan: TaskPlan, global_ctx: GlobalContext) -> TaskPlan:
        """Use 26B to review the results and score quality."""
        step_results = "\n".join(
            f"Step {s.id} ({s.status}): {s.description}\n  Result: {s.result[:200]}"
            for s in plan.steps
        )

        # Read current state of modified files
        modified_files = ""
        seen = set()
        for s in plan.steps:
            for fpath in s.file_targets:
                if fpath in seen:
                    continue
                seen.add(fpath)
                full = global_ctx.repo_root / fpath
                if full.is_file():
                    try:
                        content = full.read_text(errors="replace")
                        modified_files += f"\n--- {fpath} ---\n{content[:3000]}\n"
                    except Exception:
                        pass

        prompt = f"""Score this implementation. Be practical, not perfectionist.

TASK: {task}

FILES CREATED:
{modified_files[:6000]}

Score 0-100:
- 80+: Code runs and does what was asked. Minor issues ok.
- 50-79: Partially works but has real bugs (syntax errors, missing imports).
- <50: Fundamentally broken or incomplete.

If all files have valid syntax and the task is addressed, score 80+.

Return JSON: {{"score": <0-100>, "notes": "<one line: what's wrong or 'looks good'>"}}
ONLY the JSON."""

        try:
            response = self._call_planner(prompt)
            # Parse review JSON
            import re
            match = re.search(r'\{[^}]*"score"\s*:\s*(\d+)[^}]*"notes"\s*:\s*"([^"]*)"[^}]*\}', response)
            if match:
                plan.review_score = int(match.group(1))
                plan.review_notes = match.group(2)
            else:
                # Try simpler parse
                score_match = re.search(r'"score"\s*:\s*(\d+)', response)
                plan.review_score = int(score_match.group(1)) if score_match else 50
                plan.review_notes = response[:200]
        except Exception:
            plan.review_score = 50
            plan.review_notes = "review failed"

        return plan

    # ── Model calls ─────────────────────────────────────────────────

    def _call_planner(self, prompt: str) -> str:
        """Call the 26B planner model."""
        response = self.app.engine.chat_once(
            [{"role": "user", "content": prompt}],
            think=False,
        )
        return response.get("message", {}).get("content", "").strip()

    def _call_worker(self, prompt: str, step: TaskStep) -> str:
        """Call a worker model to generate code (Aider pattern — no tool calling).

        Workers generate code text. The orchestrator writes files.
        Reliable even with small models that can't call tools.
        """
        from dataclasses import replace
        from .runtime import GemRuntimeGateway

        # Use draft model (e2b) for workers when available.
        # With OLLAMA_MAX_LOADED_MODELS=2, both stay loaded — no swap penalty.
        draft_model = self.app.config.runtime.draft_model
        if draft_model and draft_model != self.app.runtime_model:
            try:
                worker_config = replace(
                    self.app.config.runtime,
                    model=draft_model,
                    max_context_chars=16000,  # enough for complete files
                )
                worker_engine = GemRuntimeGateway(worker_config)
                step.worker_model = draft_model
            except Exception:
                worker_engine = self.app.engine
                step.worker_model = self.app.runtime_model
        else:
            worker_engine = self.app.engine
            step.worker_model = self.app.runtime_model

        # No tools — worker generates code, orchestrator writes files
        # Use higher num_predict so files don't get truncated
        response = worker_engine.chat_once(
            [
                {"role": "system", "content": "You are a code generator. Output ONLY complete code. Never truncate. Never use comments like '# rest of code here'. Write every line."},
                {"role": "user", "content": prompt},
            ],
            think=False,
        )
        return response.get("message", {}).get("content", "").strip()

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_plan_json(response: str) -> list[TaskStep]:
        """Parse the planner's JSON response into TaskStep objects."""
        import re
        # Extract JSON array from response
        match = re.search(r'\[[\s\S]*\]', response)
        if not match:
            raise ValueError("No JSON array found in planner response")

        data = json.loads(match.group())
        steps = []
        for item in data:
            steps.append(TaskStep(
                id=int(item.get("id", len(steps) + 1)),
                description=str(item.get("description", "")),
                file_targets=list(item.get("file_targets", [])),
                depends_on=list(item.get("depends_on", [])),
            ))
        return steps
