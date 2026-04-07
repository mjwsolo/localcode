from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .live_display import GemLiveDisplay
from .ui_art import agent_progress_bar
from .verification import run_verification

if TYPE_CHECKING:
    from .app import GemApp


@dataclass(slots=True)
class AgentOutcome:
    answer: str
    verification_output: str
    verification_code: int
    steps: list[str]


class AgentRunner:
    def __init__(self, app: GemApp) -> None:
        self.app = app

    def run(self, task: str, auto_verify: bool = True) -> AgentOutcome:
        observations: list[str] = []
        answer = ""
        steps: list[str] = []
        verification_output = ""
        verification_code = 0
        max_steps = self.app.profile.agent_steps

        # Plan phase — build the task list
        plan_note = self.app.plan_for_task(task)
        self.app.store.append_event(self.app.session, "plan", plan_note.summary)
        self.app.maybe_escalate_for_task(task)

        # Build the visible checklist labels
        task_labels = self._build_task_labels(task, max_steps, auto_verify)

        display = GemLiveDisplay(self.app.console, self.app.config.ui.thinking_mode)

        with display:
            # Show the plan
            display.set_agent_tasks(task_labels)
            display.start_thinking(plan_note.summary[:120])

            # Execution loop
            for step_number in range(1, max_steps + 1):
                step_idx = step_number - 1
                display.update_agent_task(step_idx, "running")
                display.state.phase = "thinking"
                display.state.phase_detail = f"step {step_number}/{max_steps}"
                display._refresh()

                prompt = self._build_prompt(task, observations, step_number, max_steps)

                # Run the step through the tool loop (with live display)
                answer = self._run_agent_step(prompt, display)
                steps.append(answer)
                observations.append(answer[:1200])

                display.update_agent_task(step_idx, "done")
                self.app.store.append_event(
                    self.app.session,
                    "agent_step",
                    f"{step_number}/{max_steps} {answer[:160]}",
                )

                if self._should_stop(answer):
                    # Mark remaining steps as skipped (leave as pending)
                    break

            # Verification phase
            if auto_verify and any(self._looks_like_code_change(step) for step in steps):
                verify_idx = len(task_labels) - 1  # last task is verify
                display.update_agent_task(verify_idx, "running")
                display.start_verifying()

                verification_output, verification_code = run_verification(
                    self.app.repo_root,
                    bias=self.app.profile.verification_bias,
                )
                self.app.store.append_event(
                    self.app.session,
                    "verification",
                    f"exit={verification_code} {verification_output[:400]}",
                )

                if verification_code == 0:
                    display.update_agent_task(verify_idx, "done")
                else:
                    display.update_agent_task(verify_idx, "error")

                    # Auto-repair on failure
                    retry_prompt = (
                        "The previous coding attempt failed verification.\n"
                        f"Verification output:\n{verification_output[:4000]}\n\n"
                        "Analyze the errors, fix only the failing issues, and provide the corrected code."
                    )
                    display.state.phase = "thinking"
                    display.state.phase_detail = "repairing after verification failure"
                    display._refresh()

                    answer = self._run_agent_step(retry_prompt, display)
                    steps.append(answer)

                    # Re-verify
                    display.update_agent_task(verify_idx, "running")
                    display.start_verifying()
                    verification_output, verification_code = run_verification(
                        self.app.repo_root,
                        bias=self.app.profile.verification_bias,
                    )
                    display.update_agent_task(
                        verify_idx,
                        "done" if verification_code == 0 else "error",
                    )
                    self.app.store.append_event(
                        self.app.session,
                        "re-verification",
                        f"exit={verification_code} {verification_output[:400]}",
                    )

            display.finish()

        # Print final result outside the live display
        if answer:
            self.app.session.last_assistant_text = answer
            self.app.session.messages.append({"role": "assistant", "content": answer})
            self.app.store.save(self.app.session)

        return AgentOutcome(
            answer=answer,
            verification_output=verification_output,
            verification_code=verification_code,
            steps=steps,
        )

    def _run_agent_step(self, prompt: str, display: GemLiveDisplay | None = None) -> str:
        """Run a single agent step using the app's primary unified execution loop."""
        from .composer import compose_messages
        from .context import build_context_block
        from .prompts import build_system_prompt
        from .project_context import load_project_context
        from .agent_loop import run_agent_loop

        ctx_chars = self.app._effective_context_chars()
        context = build_context_block(
            self.app.repo_root,
            self.app.session.pinned_files,
            ctx_chars,
        )
        system_prompt = build_system_prompt(self.app.profile)
        project_ctx = load_project_context(self.app.repo_root)
        if project_ctx:
            system_prompt = f"{system_prompt}\n\n{project_ctx}"

        composed = compose_messages(
            self.app.profile,
            system_prompt,
            context,
            self.app.session.messages[-6:],  # keep context tight in agent mode
            prompt,
        )

        return run_agent_loop(self.app, prompt, composed, self.app.out)

    @staticmethod
    def _build_task_labels(task: str, max_steps: int, auto_verify: bool) -> list[str]:
        """Build human-readable checklist labels for the agent plan."""
        # Truncate the task for display
        task_short = task[:60] + ("..." if len(task) > 60 else "")
        labels = []
        for i in range(1, max_steps + 1):
            if i == 1:
                labels.append(f"Analyze: {task_short}")
            elif i == max_steps:
                labels.append("Finalize changes")
            else:
                labels.append(f"Step {i}: implement")
        if auto_verify:
            labels.append("Run verification")
        return labels

    @staticmethod
    def _looks_like_code_change(answer: str) -> bool:
        lowered = answer.lower()
        indicators = (
            "```diff", "```python", "```javascript", "```typescript",
            "write_file", "edit_file", "Wrote ", "Edited ", "Created ",
            "apply", "patch", ".py", ".ts", ".js",
        )
        return any(marker in lowered for marker in indicators)

    @staticmethod
    def _should_stop(answer: str) -> bool:
        lowered = answer.lower()
        stop_markers = (
            "final answer",
            "task complete",
            "task is complete",
            "done.",
            "completed.",
            "all changes have been",
            "verification passed",
            "no further changes",
        )
        return any(marker in lowered for marker in stop_markers)

    @staticmethod
    def _build_prompt(task: str, observations: list[str], step_number: int, max_steps: int) -> str:
        if not observations:
            return (
                f"Task: {task}\n\n"
                f"Agent step {step_number}/{max_steps}: "
                "Start by understanding the task. Use tools to read relevant files, search the codebase, "
                "and gather context. Then make the necessary changes. "
                "Use edit_file for surgical changes, write_file for new files, and bash for running tests."
            )
        recent = "\n\n".join(observations[-3:])
        remaining = max_steps - step_number
        urgency = ""
        if remaining <= 1:
            urgency = " This is your last step - finalize and provide the answer."
        elif remaining <= 2:
            urgency = " Wrapping up soon - make remaining changes and verify."
        return (
            f"Original task: {task}\n\n"
            f"Recent observations:\n{recent}\n\n"
            f"Agent step {step_number}/{max_steps}:{urgency} "
            "Continue from the current state. Use tools as needed. "
            "Say 'task complete' when done."
        )
