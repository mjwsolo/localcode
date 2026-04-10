"""Sub-agent delegation — spawn isolated workers for subtasks.

The coordinator agent can delegate work to sub-agents that:
1. Run in their own context (no shared state)
2. Optionally use a git worktree for file isolation
3. Return a structured result to the coordinator
4. Can run concurrently for independent tasks
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import GemApp


@dataclass
class SubAgentTask:
    prompt: str
    result: str = ""
    status: str = "pending"  # pending | running | done | error
    error: str = ""


@dataclass
class SubAgentResult:
    tasks: list[SubAgentTask]
    summary: str = ""

    @property
    def all_done(self) -> bool:
        return all(t.status in ("done", "error") for t in self.tasks)

    @property
    def success_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == "done")


class SubAgentCoordinator:
    """Coordinate sub-agent workers from a parent agent."""

    def __init__(self, app: "GemApp") -> None:
        self.app = app

    def delegate(self, tasks: list[str], concurrent: bool = True) -> SubAgentResult:
        """Run multiple sub-agent tasks. Returns when all complete."""
        agent_tasks = [SubAgentTask(prompt=p) for p in tasks]
        result = SubAgentResult(tasks=agent_tasks)

        if concurrent and len(tasks) > 1:
            self._run_concurrent(agent_tasks)
        else:
            self._run_serial(agent_tasks)

        # Build summary
        done = [t for t in agent_tasks if t.status == "done"]
        failed = [t for t in agent_tasks if t.status == "error"]
        result.summary = (
            f"{len(done)}/{len(agent_tasks)} tasks completed"
            + (f", {len(failed)} failed" if failed else "")
        )
        return result

    def _run_serial(self, tasks: list[SubAgentTask]) -> None:
        for task in tasks:
            self._execute_task(task)

    def _run_concurrent(self, tasks: list[SubAgentTask]) -> None:
        with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as pool:
            futures: dict[Future, SubAgentTask] = {}
            for task in tasks:
                f = pool.submit(self._execute_task, task)
                futures[f] = task
            for f in futures:
                try:
                    f.result(timeout=120)
                except Exception as exc:
                    futures[f].status = "error"
                    futures[f].error = str(exc)

    def _execute_task(self, task: SubAgentTask) -> None:
        """Run a single sub-agent task in isolated context."""
        task.status = "running"
        try:
            # Create a minimal message for the sub-agent
            from .prompts import build_system_prompt
            from .composer import compose_messages

            system = build_system_prompt(self.app.profile)
            system += "\nYou are a worker agent. Complete the task and return only the result. Be concise."

            composed = compose_messages(
                self.app.profile,
                system,
                "",  # no repo context for workers (isolated)
                [],  # fresh conversation
                task.prompt,
                provider=self.app.config.runtime.provider,
            )

            # Use the engine directly
            response = self.app.engine.chat_once(
                composed,
                tools=self.app.toolkit.schemas(compact=True, minimal=True),
            )
            msg = response.get("message", {})
            content = msg.get("content", "").strip()

            # Handle tool calls if any
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_results = self.app.toolkit.execute_tool_calls(tool_calls)
                # Feed results back for a final answer
                composed2 = [
                    *composed,
                    msg,
                    *tool_results,
                    {"role": "user", "content": f"Summarize the result concisely: {task.prompt}"},
                ]
                r2 = self.app.engine.chat_once(composed2)
                content = r2.get("message", {}).get("content", content).strip()

            task.result = content
            task.status = "done"
        except Exception as exc:
            task.status = "error"
            task.error = str(exc)
