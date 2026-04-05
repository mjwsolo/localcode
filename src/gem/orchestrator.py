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

    Usage:
        orch = Orchestrator(app)
        outcome = orch.run("make a pong app")
    """

    REVIEW_THRESHOLD = 80  # minimum score to pass review
    MAX_REVIEW_LOOPS = 2   # max planner→worker→review cycles

    def __init__(self, app: "GemApp") -> None:
        self.app = app
        self.out = app.out

    def run(self, task: str) -> TaskPlan:
        """Execute the full pipeline: plan → workers → review → (loop if needed)."""
        # Phase 1: Build global context
        global_ctx = self._build_global_context()

        for attempt in range(self.MAX_REVIEW_LOOPS):
            # Phase 2: Plan (26B generates task DAG)
            self.out.print_info(f"planning (attempt {attempt + 1})...")
            plan = self._plan(task, global_ctx)

            if not plan.steps:
                self.out.print_info("planner returned no steps, falling back to direct execution")
                return plan

            # Phase 3: Execute waves (e2b workers in parallel)
            waves = plan.get_waves()
            for wave_num, wave in enumerate(waves, 1):
                self.out.print_info(f"wave {wave_num}/{len(waves)}: {len(wave)} workers")
                self._execute_wave(wave, global_ctx)
                # Merge: update global context with results
                for step in wave:
                    if step.status == "done":
                        global_ctx.completed_steps.append(
                            f"[{step.id}] {step.description}: {step.result[:100]}"
                        )
                        # Re-read modified files
                        for fpath in step.file_targets:
                            full = global_ctx.repo_root / fpath
                            if full.is_file():
                                try:
                                    global_ctx.existing_files[fpath] = full.read_text(errors="replace")
                                except Exception:
                                    pass

            # Phase 4: Review (26B checks quality)
            self.out.print_info("reviewing...")
            plan = self._review(task, plan, global_ctx)

            if plan.review_score >= self.REVIEW_THRESHOLD:
                self.out.print_info(f"review passed: {plan.review_score}/100")
                break
            else:
                self.out.print_info(
                    f"review score {plan.review_score}/100 < {self.REVIEW_THRESHOLD}, "
                    f"replanning with feedback..."
                )
                # Feed review notes back into the next planning cycle
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
        prompt = f"""You are a senior software architect. Break this task into concrete steps.

TASK: {task}

REPOSITORY:
{global_ctx.file_tree[:2000]}

{global_ctx.snapshot()}

Return a JSON array of steps. Each step has:
- "id": integer (1, 2, 3...)
- "description": what to do (be specific: "Create file X with Y" or "Add function Z to file W")
- "file_targets": list of file paths this step reads or modifies
- "depends_on": list of step IDs that must complete first (empty if independent)

Rules:
- Steps that don't share files CAN run in parallel (empty depends_on)
- Steps that modify the same file MUST be sequential (add dependency)
- Each step should be small enough for a junior developer
- Maximum 8 steps
- Return ONLY valid JSON, no explanation

Example:
[
  {{"id": 1, "description": "Create pong.py with pygame boilerplate", "file_targets": ["pong.py"], "depends_on": []}},
  {{"id": 2, "description": "Add Paddle class to pong.py", "file_targets": ["pong.py"], "depends_on": [1]}},
  {{"id": 3, "description": "Add Ball class to pong.py", "file_targets": ["pong.py"], "depends_on": [1]}}
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
        """Execute a single step using an e2b worker."""
        step.status = "running"
        self.out.log_tool("worker", f"[{step.id}] {step.description[:50]}")

        # Build local context — only files this step needs
        local_ctx = LocalContext(
            step=step,
            global_snapshot=global_ctx.snapshot(),
        )
        for fpath in step.file_targets:
            full = global_ctx.repo_root / fpath
            if full.is_file():
                try:
                    local_ctx.file_contents[fpath] = full.read_text(errors="replace")
                except Exception:
                    pass
            elif fpath in global_ctx.existing_files:
                local_ctx.file_contents[fpath] = global_ctx.existing_files[fpath]

        # Build worker prompt
        file_section = ""
        for path, content in local_ctx.file_contents.items():
            file_section += f"\n--- {path} ---\n{content[:4000]}\n"

        prompt = f"""You are a code worker. Execute this ONE task precisely.

TASK: {step.description}

{f"EXISTING FILES:{file_section}" if file_section else "No existing files — create from scratch."}

CONTEXT: {local_ctx.global_snapshot[:1000]}

Instructions:
- Use write_file to create new files
- Use edit_file with old_string/new_string for changes to existing files
- Use bash to run commands if needed
- Do ONLY what the task says, nothing more
- When done, say "DONE" with a one-line summary"""

        try:
            result = self._call_worker(prompt, step)
            step.result = result
            step.status = "done"
            self.out.tool_result(f"[{step.id}] done: {result[:80]}")
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

        prompt = f"""You are a code reviewer. Score the implementation quality.

ORIGINAL TASK: {task}

STEP RESULTS:
{step_results}

CURRENT FILES:
{modified_files[:6000]}

Score the implementation 0-100:
- Does it fulfill the original task?
- Is the code correct and complete?
- Are there bugs or missing pieces?

Return JSON: {{"score": <0-100>, "notes": "<specific issues or 'looks good'>"}}
Return ONLY the JSON."""

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
        from .config import RuntimeConfig
        from .runtime import GemRuntimeGateway
        from dataclasses import replace

        # Use the main model (26B if available, otherwise current)
        response = self.app.engine.chat_once(
            [{"role": "user", "content": prompt}],
            think=False,
        )
        return response.get("message", {}).get("content", "").strip()

    def _call_worker(self, prompt: str, step: TaskStep) -> str:
        """Call an e2b worker model with tool access."""
        from .config import RuntimeConfig
        from .runtime import GemRuntimeGateway
        from dataclasses import replace

        # Try to use draft model (e2b) for workers — faster
        draft_model = self.app.config.runtime.draft_model
        if draft_model and draft_model != self.app.runtime_model:
            try:
                worker_config = replace(
                    self.app.config.runtime,
                    model=draft_model,
                    max_context_chars=8000,
                )
                worker_engine = GemRuntimeGateway(worker_config)
                step.worker_model = draft_model
            except Exception:
                worker_engine = self.app.engine
                step.worker_model = self.app.runtime_model
        else:
            worker_engine = self.app.engine
            step.worker_model = self.app.runtime_model

        # Give worker access to file tools only
        worker_tools = [
            t for t in self.app.toolkit.schemas(minimal=True)
            if t["function"]["name"] in ("read_file", "write_file", "edit_file", "bash", "grep", "glob")
        ]

        response = worker_engine.chat_once(
            [{"role": "user", "content": prompt}],
            tools=worker_tools,
            think=False,
        )
        msg = response.get("message", {})
        content = msg.get("content", "").strip()

        # Execute tool calls if any
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tool_results = self.app.toolkit.execute_tool_calls(tool_calls)
            content += "\n" + "\n".join(r.get("content", "") for r in tool_results)

        return content

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
