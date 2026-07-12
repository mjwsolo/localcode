"""Goal-state inference for a single user turn.

LocalCode's earlier loop inferred "done" from surface form:
no tool calls + some assistant text. That is too weak for agentic work.
This module gives each turn a first-class goal object so prompting,
telemetry, history, and completion gates all speak the same language.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re


__all__ = [
    "GoalState",
    "infer_goal_state",
    "extract_feature_criteria",
]


_FEATURE_EXTRACTION_PROMPT = """\
You are a requirements analyst. Read the user's request and output the discrete, \
verifiable features the user wants. ONE per line, prefixed with "- ". \
Output ONLY the checklist — no preamble, no commentary, no numbering, no headings.

Each item must:
- be a single concrete outcome the user wants (not a step or action),
- be specific enough that someone else could verify it later (with a probe, a read, or a test),
- restate ONLY what the user asked for — do NOT invent features the user did not request.

If the request is trivial (e.g. a question, a one-line edit, a quick command), output \
the single line "- (trivial — no feature checklist needed)" and nothing else.

User request:
\"\"\"
{user_text}
\"\"\"

Checklist:
"""


def extract_feature_criteria(runtime, user_text: str, *, max_features: int = 12) -> list[str]:
    """Best-effort extraction of discrete user-requested features.

    Calls the underlying model once at task intake to decompose the user's
    request into a verifiable checklist. The model sees ONLY the user's
    text — no app-specific scaffolding — so the result is general across
    any kind of request (build_app, edit, dashboard, fix, etc.).

    Returns [] on any failure or if the model decides the request is
    trivial. Caller should fall back to the pre-existing generic
    success_criteria in that case.

    Why a model call rather than a heuristic: phrasings like
    'with audio AND text AND quiz' vs 'should support uploading and exporting CSV'
    vs 'a dashboard showing revenue / churn / MRR' don't share a common
    parse tree. Heuristic splitters miss most of them. The model's
    natural-language understanding gets it right cheaply (~3 s of TTFT,
    ~50–150 decoded tokens at task start).
    """
    text = (user_text or "").strip()
    if not text or len(text) < 20:
        return []
    try:
        prompt = _FEATURE_EXTRACTION_PROMPT.format(user_text=text)
        raw = runtime.generate_once(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
        )
    except Exception:
        return []
    if not raw:
        return []
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("- ", "* ", "• ")):
            s = s[2:].strip()
        elif s[:1].isdigit():
            # Strip leading "1. " / "1) " numbering if the model added it
            # despite our instruction.
            m = re.match(r"^\d+[.)]\s*(.*)", s)
            if m:
                s = m.group(1).strip()
        if not s:
            continue
        if "trivial" in s.lower() and "no feature" in s.lower():
            return []
        # Drop obvious meta-commentary the model sometimes emits despite
        # the instruction.
        if s.lower().startswith(("here is", "here's", "checklist", "the user")):
            continue
        if len(s) > 200:
            s = s[:197] + "…"
        if s not in out:
            out.append(s)
        if len(out) >= max_features:
            break
    return out


_SLUG_STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "application",
    "ability",
    "able",
    "all",
    "build",
    "can",
    "create",
    "for",
    "from",
    "have",
    "help",
    "i",
    "in",
    "it",
    "make",
    "me",
    "my",
    "of",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "using",
    "via",
    "with",
}


@dataclass(frozen=True)
class GoalState:
    raw_user_text: str
    goal_type: str
    task_kind: str
    task_slug: str
    goal_summary: str
    success_criteria: tuple[str, ...]
    allows_blocking_question: bool = True

    def as_dict(self) -> dict:
        data = asdict(self)
        data["success_criteria"] = list(self.success_criteria)
        return data


def infer_goal_state(user_text: str) -> GoalState:
    text = (user_text or "").strip()
    lower = text.lower()
    if re.search(r"\b(build|create|make|scaffold|implement)\b.{0,40}\b(app|application|website|dashboard|api|service|project)\b", lower):
        goal_type, task_kind = "build_app", "new_app"
        criteria = ("Requested implementation exists on disk", "Relevant build or tests pass", "Runtime behavior is verified when applicable")
    elif re.search(r"\b(run|launch|start|serve|open)\b.{0,24}\b(app|server|site|project|it)\b", lower):
        goal_type, task_kind = "run_or_launch", "run"
        criteria = ("Target process is running", "Exact access URL or process status is reported")
    elif re.search(r"\b(fix|change|edit|update|refactor|rename|remove|add)\b", lower):
        goal_type, task_kind = "edit_existing", "edit"
        criteria = ("Requested code change exists", "Relevant verification passes")
    elif re.match(r"^(what|why|how|where|when|who|is|are|can|could|should|does|do)\b", lower) or text.endswith("?"):
        goal_type, task_kind = "question", "question"
        criteria = ("Question is answered from grounded evidence",)
    else:
        goal_type, task_kind = "general_task", "general_task"
        criteria = ("Assistant makes concrete progress toward the user's request", "Assistant stops only on completion or a focused blocking question")
    return GoalState(
        raw_user_text=text,
        goal_type=goal_type,
        task_kind=task_kind,
        task_slug=_make_task_slug(text),
        goal_summary=text[:240],
        success_criteria=criteria,
    )


def _make_task_slug(text: str) -> str:
    """Short internal task label, not a forced project directory."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    seen: set[str] = set()
    content_words: list[str] = []
    for word in words:
        if word in _SLUG_STOPWORDS or len(word) <= 1 or word in seen:
            continue
        seen.add(word)
        content_words.append(word)
    selected = content_words[:4] or words[:3] or ["task"]
    clean = "-".join(selected)
    clean = re.sub(r"-+", "-", clean).strip("-") or "task"
    clean = clean[:30].strip("-") or "task"
    return clean
