"""Stable outcome + failure normalization for a parsed run stream.

Turns a `ParseResult` into a `RunOutcome`: one normalized failure category from
the plan's fixed taxonomy (§10), the raw LocalCode reason preserved alongside,
and objective counts (rounds, tool calls, tokens). Both harnesses normalize
here so a regression comparison never depends on two consumers bucketing the
same reason string differently.

Boundary that matters (integration plan §9): this module only classifies the
LocalCode side — setup, model server, agent, protocol. Whether the *task*
passed is the verifier's call, never inferred from `status == "ok"`. The three
verifier-owned categories (`task_passed`, `task_failed`, `verifier_failure`)
are defined here for a shared vocabulary but are assigned by the caller, not by
this parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .jsonl import ParseResult

__all__ = ["FailureCategory", "RunOutcome", "normalize_reason", "redact", "outcome_from_parse"]


class FailureCategory:
    """The fixed failure taxonomy (integration plan §10). String constants so
    they serialize cleanly and compare stably across releases."""

    # LocalCode-side — assignable from the stream alone.
    SETUP_FAILURE = "setup_failure"
    MODEL_SERVER_UNREACHABLE = "model_server_unreachable"
    MODEL_SERVER_FAILURE = "model_server_failure"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_CRASH = "agent_crash"
    AGENT_INCOMPLETE = "agent_incomplete"
    EVENT_PROTOCOL_ERROR = "event_protocol_error"
    TOOL_FAILURE = "tool_failure"
    CONTEXT_EXHAUSTED = "context_exhausted"
    REASONING_LOOP_ABORT = "reasoning_loop_abort"
    # Clean finish (LocalCode's own loop completed). NOT "task passed".
    COMPLETED = "completed"
    # Verifier-owned — assigned by the caller, never by this parser.
    VERIFIER_FAILURE = "verifier_failure"
    TASK_FAILED = "task_failed"
    TASK_PASSED = "task_passed"


# Exact LocalCode `result.reason` / status strings → category. Matched first;
# substring rules below catch the parameterized ones (e.g. "stream_error:...").
_EXACT: dict[str, str] = {
    "completed": FailureCategory.COMPLETED,
    "verified_run_or_launch": FailureCategory.COMPLETED,
    "no model found on disk": FailureCategory.SETUP_FAILURE,
    "could not start the model server": FailureCategory.MODEL_SERVER_FAILURE,
    "model_server_unreachable": FailureCategory.MODEL_SERVER_UNREACHABLE,
    "keyboard interrupt": FailureCategory.AGENT_INCOMPLETE,
    "user_cancel": FailureCategory.AGENT_INCOMPLETE,
    "stream_interrupt": FailureCategory.AGENT_INCOMPLETE,
    "stall_exhausted": FailureCategory.AGENT_INCOMPLETE,
    "max_rounds": FailureCategory.AGENT_INCOMPLETE,
    "max_output_recovery_exhausted": FailureCategory.CONTEXT_EXHAUSTED,
    "context_exhausted": FailureCategory.CONTEXT_EXHAUSTED,
    "thinking_loop_exhausted": FailureCategory.REASONING_LOOP_ABORT,
    "truncated_tool_call_exhausted": FailureCategory.TOOL_FAILURE,
}

# (substring, category) — ordered; first hit wins. For reasons that carry a
# suffix payload like "stream_error:ConnectionError".
_SUBSTRING: tuple[tuple[str, str], ...] = (
    ("model_server", FailureCategory.MODEL_SERVER_FAILURE),
    ("stream_error", FailureCategory.MODEL_SERVER_FAILURE),
    ("reasoning_loop", FailureCategory.REASONING_LOOP_ABORT),
    ("thinking_loop", FailureCategory.REASONING_LOOP_ABORT),
    ("context", FailureCategory.CONTEXT_EXHAUSTED),
    ("tool", FailureCategory.TOOL_FAILURE),
    ("timeout", FailureCategory.AGENT_TIMEOUT),
)


def normalize_reason(status: str, reason: str) -> str:
    """Map a LocalCode terminal (status, reason) to a stable failure category.

    `status` is the coarse `result.status` ("ok"/"error"/"timeout"/
    "interrupted"/"incomplete"); `reason` is the finer machine reason. Status is
    consulted first for the unambiguous coarse cases, then the reason string.
    An unrecognised failing reason falls back to `agent_crash` (something went
    wrong we haven't named) rather than being silently treated as success.
    """
    status = (status or "").strip().lower()
    reason = (reason or "").strip()
    rl = reason.lower()

    if status == "ok":
        return _EXACT.get(rl, FailureCategory.COMPLETED)
    if status == "timeout":
        return FailureCategory.AGENT_TIMEOUT
    if status == "interrupted":
        return FailureCategory.AGENT_INCOMPLETE

    # status in {"error","incomplete",...}: classify by the reason.
    if rl in _EXACT:
        return _EXACT[rl]
    for needle, category in _SUBSTRING:
        if needle in rl:
            return category
    if status == "incomplete":
        return FailureCategory.AGENT_INCOMPLETE
    # An error we couldn't place — name it a crash, never a pass.
    return FailureCategory.AGENT_CRASH


@dataclass
class RunOutcome:
    """Normalized, objective summary of one run stream."""

    status: str
    category: str
    raw_reason: str
    final_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    rounds: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    schema: str = ""
    protocol_ok: bool = True
    problems: list[str] = field(default_factory=list)

    @property
    def clean_finish(self) -> bool:
        """LocalCode's own loop completed without a failure category.

        NOT a statement about task success — the verifier owns that."""
        return self.category == FailureCategory.COMPLETED and self.protocol_ok


def _as_int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def outcome_from_parse(parsed: ParseResult) -> RunOutcome:
    """Build a `RunOutcome` from a parsed stream.

    Token counts come from the terminal `result` event when present (it already
    carries the accumulated totals); rounds and tool calls are counted from the
    event stream by their real type names (`round_end`, `tool_call`). A tool
    result carrying an error status counts toward `tool_failures`.
    """
    terminal = parsed.terminal
    status = str(terminal.get("status") if terminal else "") or ""
    reason = str(terminal.get("reason") if terminal else "") or ""

    # A protocol violation (missing/duplicate terminal) is itself a category —
    # the stream can't be trusted, so we don't pretend to know the LocalCode
    # outcome.
    protocol_problems = [p for p in parsed.problems
                         if p.kind in {"missing_terminal", "duplicate_terminal", "decode_error"}]
    if protocol_problems:
        category = FailureCategory.EVENT_PROTOCOL_ERROR
    else:
        category = normalize_reason(status, reason)

    tokens = terminal.get("tokens", {}) if terminal else {}
    if isinstance(tokens, dict):
        prompt = _as_int(tokens.get("prompt"))
        completion = _as_int(tokens.get("completion"))
        total = _as_int(tokens.get("total")) or (prompt + completion)
    else:
        prompt = completion = total = 0

    tool_calls = 0
    tool_failures = 0
    rounds = 0
    for ev in parsed.events:
        if ev.type == "round_end":
            rounds += 1
        elif ev.type == "tool_call":
            tool_calls += 1
        elif ev.type == "tool_result":
            st = str(ev.get("status") or ev.get("result_status") or "").lower()
            if st in {"error", "failed", "rejected"}:
                tool_failures += 1

    return RunOutcome(
        status=status,
        category=category,
        raw_reason=reason,
        final_text=str(terminal.get("final_text") if terminal else "") or "",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        rounds=rounds,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        schema=str(parsed.schema) if parsed.schema else "",
        protocol_ok=not protocol_problems,
        problems=[f"{p.kind}: {p.detail}" for p in parsed.problems],
    )


# Field names whose values are secrets and must never be persisted in a
# trajectory or archived config. Case-insensitive substring match.
_SECRET_HINTS = ("token", "secret", "password", "api_key", "apikey",
                 "authorization", "auth_header", "bearer", "credential")
_REDACTED = "***REDACTED***"


def redact(obj: Any) -> Any:
    """Recursively redact secret-looking fields from a dict/list before it is
    persisted (integration plan §11). Returns a redacted copy; the input is not
    mutated. Non-container values pass through unchanged."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k).lower()
            if any(hint in key for hint in _SECRET_HINTS):
                out[k] = _REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj
