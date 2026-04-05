from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PlanNote:
    summary: str
    use_small_model: bool
    suggest_search: bool
    suggest_index: bool
    complexity: str


def build_plan_note(task: str) -> PlanNote:
    lowered = task.lower()
    search_words = ("latest", "news", "search", "look up", "documentation", "docs")
    broad_words = ("refactor", "architecture", "large", "multi-file", "migration", "agentic")
    use_small_model = len(task) < 600 and not any(word in lowered for word in broad_words)
    suggest_search = any(word in lowered for word in search_words)
    suggest_index = any(word in lowered for word in ("repo", "codebase", "file", "module", "class", "function"))
    complexity = "high" if any(word in lowered for word in broad_words) else "medium"
    if use_small_model and complexity != "high":
        complexity = "low"
    summary = (
        "Start narrow: inspect local code and use the smallest reasonable model first."
        if use_small_model
        else "Use the main model for synthesis, but rely on retrieval and verification to keep context tight."
    )
    return PlanNote(
        summary=summary,
        use_small_model=use_small_model,
        suggest_search=suggest_search,
        suggest_index=suggest_index,
        complexity=complexity,
    )
