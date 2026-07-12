"""Pure policy helpers for deciding whether a shell command achieved its task."""
from __future__ import annotations

from dataclasses import dataclass
import re


_FAILURE_SIGNALS = re.compile(
    r"traceback \(most recent call last\)|modulenotfounderror|module_not_found|"
    r"command not found|no such file or directory|permission denied|"
    r"\b(?:failed|fatal|panic):|\btests? failed\b|\b[1-9]\d* failed\b",
    re.IGNORECASE,
)
_FALLBACK_SIGNALS = re.compile(
    r"\b(?:falling back|fallback to|using fallback|retrying with|skipped because)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExecutionAssessment:
    shell_succeeded: bool
    task_succeeded: bool
    confidence: str
    signals: tuple[str, ...]


def assess_shell_execution(command: str, output: str, returncode: int) -> ExecutionAssessment:
    """Separate process exit status from evidence that the requested task worked."""
    command = command or ""
    output = output or ""
    signals: list[str] = []
    shell_ok = returncode == 0
    if not shell_ok:
        signals.append(f"nonzero-exit:{returncode}")
    if _FAILURE_SIGNALS.search(output):
        signals.append("failure-in-output")
    if _FALLBACK_SIGNALS.search(output):
        signals.append("fallback-used")
    if "|" in command and "pipefail" not in command:
        signals.append("pipeline-without-pipefail")
    if re.search(r"(?:\|\||;\s*true\b|\bor\s+true\b)", command):
        signals.append("failure-masking-control-flow")
    task_ok = shell_ok and not any(
        s in signals for s in ("failure-in-output", "fallback-used", "failure-masking-control-flow")
    )
    confidence = "high" if task_ok and "pipeline-without-pipefail" not in signals else "low"
    return ExecutionAssessment(shell_ok, task_ok, confidence, tuple(signals))
