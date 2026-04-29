"""Plan artifact + persistence — durable planning notes for a task.

Port of the essence of OpenAI agent's plan workflow (see ARCHITECTURE.md §5
for source path and deep-dive). agent ships a much larger system —
permission dialogs, multi-agent coordination, CCR/web sync, feature-gated
V2 semantics. We port only the mechanics we can defend for a local-only
single-user agent:

- The plan lives as a markdown file on disk at ~/.localcode/plans/{slug}.md.
- Slug is a small random word-pair generated once per session.
- `enter_plan_mode` / `exit_plan_mode` tools (in agent.py) flip an in-memory
  flag on `LocalCodeApp.plan_mode` so the UI and telemetry can track whether
  a task is currently using the planning workflow.
- `/plan` slash command in the chat screen lets the user create or inspect a
  plan artifact for the current task. It does not block the main runtime.

No permissions dialog, no multi-agent, no remote sync. Just: keep a markdown
plan file around so planning stays visible and shared with the user.
"""
from __future__ import annotations

import random
from pathlib import Path


_GLOBAL_PLANS_DIR = Path.home() / ".localcode" / "plans"


def _find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward from `start` (default: cwd) looking for a git repo.

    Returns the repo root (the dir containing `.git`) or None if we're
    not inside a repo. Used to place plans alongside the project they
    belong to, rather than globally in ~/.localcode/plans/.
    """
    p = (start or Path.cwd()).resolve()
    for anc in (p, *p.parents):
        if (anc / ".git").exists():
            return anc
    return None


def plans_dir() -> Path:
    """Where plans for the CURRENT working directory should live.

    Project-aware:
      * If cwd is inside a git repo → <repo>/.localcode/plans/
      * Otherwise → ~/.localcode/plans/

    Plans are tied to the project they were created for. Mixing plans
    from 5 different projects together in one global directory was a
    real wart — fixed 2026-04-22.
    """
    repo = _find_project_root()
    if repo is not None:
        return repo / ".localcode" / "plans"
    return _GLOBAL_PLANS_DIR


# Back-compat alias. Old call sites using `PLANS_DIR` directly will
# resolve to the project dir at import time, which is wrong for any
# project switch; prefer `plans_dir()`.
PLANS_DIR = plans_dir()

# Short word list for slug generation. Enough distinct pairs (~40k) that
# collisions are rare; recognizable words so the filename is memorable.
_SLUG_WORDS = [
    "quiet", "bright", "steady", "swift", "calm", "bold", "sharp", "gentle",
    "clever", "nimble", "rugged", "warm", "crisp", "dawn", "dusk", "storm",
    "river", "forest", "mountain", "valley", "cedar", "birch", "maple", "pine",
    "falcon", "otter", "fox", "hare", "owl", "heron", "wren", "sparrow",
    "amber", "jade", "indigo", "coral", "ivory", "slate", "ember", "mist",
    "journey", "anchor", "compass", "harbor", "horizon", "orbit", "meadow", "garden",
]


def _rand_slug(rng: random.Random | None = None) -> str:
    """Generate a 2-word slug like 'swift-falcon'. Caller is responsible for
    collision handling if it matters (our PLANS_DIR probably won't hit 40k)."""
    r = rng or random.Random()
    return f"{r.choice(_SLUG_WORDS)}-{r.choice(_SLUG_WORDS)}"


def plan_path(slug: str) -> Path:
    """Absolute path to the plan file for a given slug.

    Resolves against the CURRENT project root — see `plans_dir()`.
    """
    return plans_dir() / f"{slug}.md"


def ensure_plans_dir() -> None:
    plans_dir().mkdir(parents=True, exist_ok=True)


def read_plan(slug: str) -> str | None:
    """Return the plan contents, or None if no plan has been written yet."""
    p = plan_path(slug)
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def write_plan(slug: str, content: str) -> Path:
    """Persist the plan. Returns the path on disk so callers can surface it."""
    ensure_plans_dir()
    p = plan_path(slug)
    p.write_text(content, encoding="utf-8")
    return p


def new_slug() -> str:
    """Generate a fresh slug, retrying up to 10 times if a file already
    exists (very unlikely but keeps us honest)."""
    for _ in range(10):
        slug = _rand_slug()
        if not plan_path(slug).exists():
            return slug
    return _rand_slug()  # accept collision; writer will overwrite


# ── System-prompt injection ─────────────────────────────────────────────

PLAN_MODE_PROMPT = """\
## PLAN MODE — ACTIVE

You are in plan mode. Your job is to PLAN, not execute.

Allowed tools: read_file, grep, glob, list_files, web_search, web_fetch.
These are for exploration — understand the codebase and the user's request
before proposing changes.

### STEP 1 — Decide whether to ask, based on conversation so far

Before asking ANY clarifying question, re-read the prior turn(s). If
the user has already answered the big choices (platform, language,
scope) in earlier messages, DO NOT re-ask them — restate what they
chose in your plan's Assumptions section instead.

If the user gave you latitude ("up to you", "you decide", "whatever
you think", "just pick", "surprise me"), STOP asking and MAKE
decisions. Pick a reasonable default for each open choice and record
it under Assumptions (marked as "default") so the user can override
it if they disagree. Re-asking after latitude is given is the single
most common plan-mode failure mode — don't do it.

Only ask NEW clarifying questions (in text, no tool call) when there
is ambiguity that the conversation hasn't already resolved and that
a default would be risky to choose silently. Examples where asking
is correct: unclear deployment target that locks you into a stack;
data source that requires paid/API access; target directory that
could clobber existing files.

If you do need to ask, keep it to ≤3 questions and do NOT write the
plan file in the same turn — wait for the user's answer.

### STEP 2 — Once everything is clear, write the plan

Write to the plan file using `write_file` (the ONE write allowed in plan
mode). The plan file path is:

    {plan_path}

Plan format — markdown with this structure:

    # Goal
    <one sentence — what done looks like, with the clarifications
    the user just confirmed>

    # Assumptions (if any)
    - <anything the user explicitly said — paste their exact words>
    - <defaults you're choosing if still unspecified — flag as "default"
      so the user can override>

    # Steps
    1. <small, independently verifiable step>
       verify: <the command or assertion that proves it worked>
    2. <...>
       verify: <...>

    # Risks
    <1-2 things that could go wrong>

Keep the plan to ≤6 steps. If it's longer, the task is probably two
separate tasks — flag that in Risks.

### STEP 3 — Exit

When the plan is ready, call `exit_plan_mode` — that will return control to
the user for approval and then transition you back to normal execution mode.

### NEVER (in plan mode)

- Edit or create any files other than the plan file.
- Run destructive bash commands (rm, git push, migrations, deploys).
- Claim a task is done — you're planning, not doing.
- Commit to a technical choice (terminal/web, framework, storage) without
  either confirming with the user OR flagging it as a default in Assumptions.
"""


def plan_mode_prompt(slug: str) -> str:
    return PLAN_MODE_PROMPT.format(plan_path=plan_path(slug))
