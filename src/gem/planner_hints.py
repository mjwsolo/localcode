from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any


@dataclass(slots=True)
class PlannerHint:
    checkpoint: str
    next_action: str = ""
    likely_file: str = ""
    risk: str = ""
    quality_gap: str = ""
    stop: bool = False
    created_at: float = field(default_factory=time.time)


class PlannerHintState:
    def __init__(self) -> None:
        self.latest: PlannerHint | None = None
        self.history: list[PlannerHint] = []

    def record(self, hint: PlannerHint) -> PlannerHint:
        self.latest = hint
        self.history.append(hint)
        self.history = self.history[-12:]
        return hint


def parse_planner_hint(response: str, checkpoint: str) -> PlannerHint | None:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        data: dict[str, Any] = json.loads(match.group(0))
    except Exception:
        return None
    return PlannerHint(
        checkpoint=checkpoint,
        next_action=str(data.get("next_action", "")).strip()[:160],
        likely_file=str(data.get("likely_file", "")).strip()[:160],
        risk=str(data.get("risk", "")).strip()[:200],
        quality_gap=str(data.get("quality_gap", "")).strip()[:200],
        stop=str(data.get("stop", "")).strip().lower() in {"1", "true", "yes", "stop"},
    )
