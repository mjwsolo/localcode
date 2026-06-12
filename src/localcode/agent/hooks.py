"""Lifecycle hooks for LocalCode's agent loop.

The loop is responsible for sequencing; policy lives in small hook
functions here that can be tested and replaced independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
__all__ = [
    "TurnState",
    "EvidenceLedger",
    "QualityVerdict",
    "before_turn",
    "before_model",
    "after_tool",
    "quality_monitor",
    "completion_gate",
]


@dataclass(slots=True)
class TurnState:
    user_text: str
    goal_state: Any
    task_state: Any = None
    changed_files: list[str] = field(default_factory=list)
    bash_history: list[tuple[str, str]] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    evidence: "EvidenceLedger" = field(default_factory=lambda: EvidenceLedger())


@dataclass(slots=True)
class EvidenceLedger:
    """Structured proof gathered during a turn.

    The model may narrate intent, but completion depends on this ledger:
    explicit tool facts, runtime probes, and observed errors.
    """
    verified_urls: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    runtime_errors: list[str] = field(default_factory=list)
    empty_probe_count: int = 0
    zero_data_count: int = 0

    def add_tool_result(self, tool_name: str, args: dict[str, Any], result: str, facts: dict[str, Any]) -> None:
        text = str(result or "").strip()
        lower = text.lower()
        compact = re.sub(r"\s+", "", text)
        command = str(args.get("command", "") or "")
        path = facts.get("path")
        if isinstance(path, str) and path and facts.get("changed"):
            self._add_unique(self.changed_paths, path)
        url = facts.get("url")
        if facts.get("verification_signal") and isinstance(url, str) and url:
            self._add_unique(self.verified_urls, url)
        if "traceback " in lower or "exception" in lower or facts.get("error_type") in {"syntax_error", "verification_failed"}:
            self._add_unique(self.runtime_errors, lower[:200])
            self._add_unique(self.unresolved, "runtime-error")
        if (("curl" in command.lower() or "/api/" in command.lower()) and compact in {"[]", "{}", "null", "\"\"", ""}):
            self.empty_probe_count += 1
            self._add_unique(self.unresolved, "empty-api-response")
        if re.search(r"\b(loaded|found|returned|has)\s+0\b", lower) or re.search(r"\b0\s+(items|records|rows|entries|levels|results)\b", lower):
            self.zero_data_count += 1
            self._add_unique(self.unresolved, "zero-runtime-data")

    def blockers(self) -> list[str]:
        return list(dict.fromkeys(self.unresolved))

    @staticmethod
    def _add_unique(items: list[str], value: str) -> None:
        if value and value not in items:
            items.append(value)


@dataclass(slots=True)
class QualityVerdict:
    ok: bool
    reason: str = ""
    correction: str = ""


def before_turn(state: TurnState) -> None:
    """Reserved hook for per-turn initialization."""
    return None


def before_model(messages: list[dict[str, Any]], state: TurnState) -> list[dict[str, Any]]:
    """Transform messages before each model call."""
    return messages


def after_tool(tool_name: str, args: dict, result: str, state: TurnState) -> str:
    """Post-process a tool result before it is fed back to the model."""
    return result


def quality_monitor(content: str, state: TurnState) -> QualityVerdict:
    """Detect generic bad completions and return a correction if needed."""
    lower = (content or "").lower()
    if not content.strip():
        return QualityVerdict(ok=True)
    if (
        not state.changed_files
        and state.goal_state.goal_type in {"build_app", "edit_existing", "general_task"}
        and any(phrase in lower for phrase in ("would you like me to", "should i ", "i can implement"))
        and not getattr(state.goal_state, "allows_blocking_question", True)
    ):
        return QualityVerdict(
            ok=False,
            reason="permission-question-without-action",
            correction=(
                "SYSTEM: The previous response asked for permission instead of "
                "making the requested local code change. Continue with tools now. "
                "Ask only if a real missing decision blocks implementation."
            ),
        )
    if "http://localhost:[" in lower or "127.0.0.1:[" in lower:
        return QualityVerdict(
            ok=False,
            reason="fake-placeholder-url",
            correction=(
                "SYSTEM: The previous response contained a placeholder URL. "
                "Use tools to determine the real port/URL, then report the exact value."
            ),
        )
    unsupported = _unsupported_final_claims(content, state)
    if unsupported:
        return QualityVerdict(
            ok=False,
            reason="unsupported-final-claims",
            correction=(
                "SYSTEM: The previous completion summary claimed features that are "
                f"not evidenced in changed files: {', '.join(unsupported)}. "
                "Either implement and verify those features now, or remove the "
                "unsupported claims from the final summary. Do not overstate what was built."
            ),
        )
    return QualityVerdict(ok=True)


def completion_gate(
    *,
    repo_root: Path | str,
    state: TurnState,
    build_stage: str,
    has_runtime_verification: bool,
    partial_handoff: bool,
    blocking_question: bool,
) -> str | None:
    """Return a generic completion-block reason, or None if completion is valid."""
    if state.goal_state.goal_type == "edit_existing" and not blocking_question:
        if not state.changed_files:
            return "no-edit-applied"
        if _changed_code_files(state.changed_files) and not has_runtime_verification:
            return "unverified-edit"
    if state.goal_state.goal_type == "build_app" and state.changed_files and not blocking_question:
        if build_stage != "verified":
            return f"stage-{build_stage}"
        if partial_handoff:
            return "partial-handoff"
        if not has_runtime_verification:
            return "unverified-app-build"
        unresolved = state.evidence.blockers() or _recent_unresolved_verification_signals(state)
        if unresolved:
            return f"unresolved-verification:{', '.join(unresolved[:3])}"
        missing_features = _requested_feature_gaps(state)
        if missing_features:
            return f"missing-requested-features:{', '.join(missing_features[:3])}"
        score, deductions = _completion_score(
            state=state,
            has_runtime_verification=has_runtime_verification,
            missing_features=missing_features,
        )
        if score < 4:
            return f"low-completion-score:{score}:{', '.join(deductions[:3])}"
    if (
        not state.changed_files
        and partial_handoff
        and not getattr(state.goal_state, "allows_blocking_question", True)
    ):
        return "action-required-followup"
    return None


def _completion_score(
    *,
    state: TurnState,
    has_runtime_verification: bool,
    missing_features: list[str],
) -> tuple[int, list[str]]:
    score = 0
    deductions: list[str] = []
    if state.changed_files:
        score += 2
    else:
        deductions.append("no_changed_files")
    if has_runtime_verification:
        score += 2
    else:
        deductions.append("no_runtime_verification")
    if not missing_features:
        score += 1
    else:
        deductions.append("missing_requested_features")
    recent_errors = [
        result for _, result in state.bash_history[-3:]
        if str(result).startswith("[exit code ") or "Traceback " in str(result)
    ]
    recent_errors.extend(state.evidence.runtime_errors[-2:])
    if recent_errors:
        score -= 1
        deductions.append("recent_unresolved_errors")
    recent_tools = state.tools_called[-8:]
    if len(recent_tools) >= 6 and len(set(recent_tools)) <= 2:
        score -= 1
        deductions.append("repeated_tool_churn")
    return score, deductions


def _recent_unresolved_verification_signals(state: TurnState) -> list[str]:
    """Detect recent verification probes that prove a build is not actually ready.

    This stays generic: it does not know about a specific app domain. It only
    treats explicit runtime/probe evidence such as empty API responses, zero
    loaded records, or tracebacks as blockers for creation-task completion.
    """
    if state.goal_state.goal_type != "build_app":
        return []
    unresolved: list[str] = []
    for command, result in state.bash_history[-10:]:
        cmd = str(command).lower()
        text = str(result).strip()
        lower = text.lower()
        compact = re.sub(r"\s+", "", text)
        if "traceback " in lower or "exception" in lower:
            unresolved.append("runtime-error")
            continue
        if ("curl" in cmd or "/api/" in cmd) and compact in {"[]", "{}", "null", "\"\"", ""}:
            unresolved.append("empty-api-response")
            continue
        if re.search(r"\b(loaded|found|returned|has)\s+0\b", lower):
            unresolved.append("zero-runtime-data")
            continue
        if re.search(r"\b0\s+(items|records|rows|entries|levels|results)\b", lower):
            unresolved.append("zero-runtime-data")
            continue
    return list(dict.fromkeys(unresolved))


def _changed_code_files(paths: list[str]) -> bool:
    code_suffixes = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cc", ".cpp",
        ".h", ".hpp", ".cs", ".rb", ".php", ".lua", ".dart", ".scala",
        ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html", ".css",
        ".vue", ".svelte", ".astro",
    }
    for path in paths:
        suffix = Path(str(path)).suffix.lower()
        if suffix in code_suffixes:
            return True
    return False


def _unsupported_final_claims(content: str, state: TurnState) -> list[str]:
    """Reject concrete feature claims absent from changed-file evidence.

    This is intentionally generic. It does not try to understand every
    possible app domain; it catches high-signal phrases where claiming a
    feature without code evidence is worse than staying quiet.
    """
    if state.goal_state.goal_type not in {"build_app", "edit_existing"}:
        return []
    corpus = _changed_file_corpus(state.changed_files)
    if not corpus:
        return []

    checks: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("localStorage persistence", ("localstorage",), ("localstorage",)),
        ("browser speech synthesis", ("speechsynthesis", "speech synthesis"), ("speechsynthesis", "speechsynthesisutterance")),
        ("text-to-speech", ("text-to-speech", "tts"), ("speechsynthesis", "speechsynthesisutterance", "gtts", "pyttsx3")),
        ("true/false quiz", ("true/false", "true false"), ("true/false", "true false", "boolean", "is_correct === true")),
        ("fill-in-the-blank quiz", ("fill in the blank", "fill-in-the-blank"), ("fill in the blank", "fill-in-the-blank", "blank")),
        ("OAuth/authentication", ("oauth", "authentication", "login"), ("oauth", "auth", "login", "session")),
        ("database", ("database", "sqlite", "postgres", "mysql"), ("sqlite", "postgres", "mysql", "sqlalchemy", "database")),
    )
    unsupported: list[str] = []
    text = content.lower()
    for label, claim_terms, evidence_terms in checks:
        if any(term in text for term in claim_terms) and not any(term in corpus for term in evidence_terms):
            unsupported.append(label)

    count_claims = re.findall(r"\b(\d+)\s*\+\s+(?:words|items|questions|tests|examples)\b", text)
    if count_claims:
        numbers = [int(n) for n in re.findall(r"\b\d+\b", corpus)]
        max_number = max(numbers) if numbers else 0
        for claim in count_claims:
            if int(claim) > max_number:
                unsupported.append(f"{claim}+ item count")
    return unsupported[:4]


def _requested_feature_gaps(state: TurnState) -> list[str]:
    """High-confidence request/evidence checks for build completion.

    This is deliberately generic. It only gates on common capability words
    where code evidence is straightforward, avoiding domain-specific rules.
    The model can still choose any stack/language; it just cannot declare
    completion after verification if obvious requested capabilities are
    absent from the files it changed.
    """
    if state.goal_state.goal_type != "build_app":
        return []
    request = (state.user_text or "").lower()
    corpus = _changed_file_corpus(state.changed_files)
    if not request or not corpus:
        return []
    checks: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "audio/listening",
            ("audio", "sound", "listen", "pronunciation", "tts", "speech", "voice"),
            ("audio", "speechsynthesis", "speechsynthesisutterance", "tts", "voice", "utterance", "pyttsx3", "gtts", "howler"),
        ),
        (
            "quiz/testing",
            ("quiz", "exam", "multiple choice", "question", "test me"),
            ("quiz", "question", "answer", "score", "choice", "exam", "correct"),
        ),
        (
            "persistence/progress",
            ("save", "saved", "persist", "progress", "remember"),
            ("localstorage", "sessionstorage", "sqlite", "database", "json.dump", "writefile", "save", "progress"),
        ),
        (
            "authentication",
            ("login", "sign in", "signup", "sign up", "auth", "oauth"),
            ("login", "signin", "signup", "auth", "oauth", "session", "jwt", "password"),
        ),
        (
            "search/filtering",
            ("search", "filter", "sort"),
            ("search", "filter", "sort", "query"),
        ),
    )
    missing: list[str] = []
    for label, request_terms, evidence_terms in checks:
        if any(term in request for term in request_terms) and not any(term in corpus for term in evidence_terms):
            missing.append(label)
    return missing


def _changed_file_corpus(paths: list[str], *, max_chars: int = 160_000) -> str:
    chunks: list[str] = []
    total = 0
    for raw in paths[:40]:
        path = Path(str(raw))
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip"}:
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        chunks.append(text[: max(0, max_chars - total)])
        total += len(chunks[-1])
        if total >= max_chars:
            break
    return "\n".join(chunks).lower()
