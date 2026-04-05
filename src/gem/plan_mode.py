"""Plan mode — structured task planning with step-by-step execution.

/plan <task>  — model proposes steps
/plan show    — view current plan
/plan go      — execute all steps
/plan next    — execute next step
/plan cancel  — cancel plan
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    description: str
    status: str = "pending"  # pending | running | done | error | skipped
    result: str = ""


@dataclass
class Plan:
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    current_step: int = 0

    @property
    def is_done(self) -> bool:
        return self.current_step >= len(self.steps)

    @property
    def next_step(self) -> PlanStep | None:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def summary(self) -> str:
        lines = [f"Plan: {self.task}\n"]
        for i, step in enumerate(self.steps):
            icons = {"pending": "○", "running": "◉", "done": "✓", "error": "✗", "skipped": "—"}
            icon = icons.get(step.status, "?")
            marker = "→ " if i == self.current_step else "  "
            lines.append(f"  {marker}{icon} {i + 1}. {step.description}")
            if step.result and step.status in ("done", "error"):
                preview = step.result[:80].replace("\n", " ")
                lines.append(f"       {preview}")
        return "\n".join(lines)


def parse_plan_from_response(response: str) -> list[PlanStep]:
    """Parse numbered steps from model's plan response."""
    import re
    steps = []
    # Match patterns like "1. Do X" or "- Step 1: Do X" or "Step 1: Do X"
    for match in re.finditer(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-*]\s*(?:Step \d+:?\s*)?)(.*)', response):
        text = match.group(1).strip()
        if text and len(text) > 5 and not text.startswith("Plan") and not text.startswith("Here"):
            steps.append(PlanStep(description=text))
    # If no numbered steps found, split by sentences
    if not steps:
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 10]
        steps = [PlanStep(description=s) for s in sentences[:5]]
    return steps
