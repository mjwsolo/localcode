"""System-prompt template strings and the project-instructions loader.

Holds the active SYSTEM_PROMPT, the reasoning appendix, and the loader
for repo-local LOCALCODE.md instructions. External callers (eval/,
tests/) import these by name via `localcode.agent.prompts`.
"""
from __future__ import annotations

from pathlib import Path

from .constants import PROJECT_FILES as _PROJECT_FILES


__all__ = [
    "SYSTEM_PROMPT",
    "REASONING_RULES",
    "model_identity_line",
    "project_stack_line",
]


# Marker file -> (stack label, ordered most-specific first). Detection is
# deliberately a cheap top-level directory check: presence of these files
# in the repo root is a strong signal of the project's primary language.
# Order matters — the first matching entry names the stack so we emit a
# single, unambiguous line rather than enumerating every config file.
_STACK_MARKERS: tuple[tuple[str, str], ...] = (
    ("package.json", "JavaScript/TypeScript (Node)"),
    ("tsconfig.json", "TypeScript"),
    ("go.mod", "Go"),
    ("Cargo.toml", "Rust"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("pom.xml", "Java (Maven)"),
    ("build.gradle", "Java/Kotlin (Gradle)"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
)


SYSTEM_PROMPT = """\
You are LocalCode, a coding agent running locally on the user's machine with full filesystem access through your tools.

Top rules (most important first):
1. ACT, DON'T NARRATE. If you say "let me read/fix/write X", the tool call that does it MUST come next, in the same turn. Never end a turn on a statement of intent. At most one short sentence of preamble per tool call.
2. FINISH THE JOB. Cover every requirement the user named. Write complete, runnable code — no TODOs, stubs, placeholders, or "should I do X next?". If a piece is too big for one call, split it across calls; never drop it.
3. MATCH SCOPE. Answer plain questions plainly; build only what was asked. Don't write a script when one bash line does it.
4. DON'T REPEAT WORK. Don't re-read a file or re-run a command you already did — the result is still above. After a tool result, continue from where you left off; don't restate the plan.
5. DON'T INVENT. If you don't recognize a term, library, or command, say so — never guess a plausible meaning. Search the web before asserting facts; if results are empty, say that.

Plan & tools:
- For a task with 3+ steps, call todo_write first to lay out the steps, then keep exactly one in_progress and mark each done the instant it's finished. Your list is shown back each round — use it to see what's left.
- Use real, repo-relative paths (don't improvise `/Users/...` paths).
- For files and dirs use list_files/read_file/write_file/edit_file — NOT bash (`ls`, `cat`, `cat >`, `>`). bash is only for running commands (npm, git, build, test).
- edit_file for existing files: anchor `old_string` on 2–4 adjacent lines (matching is whitespace-tolerant; the leading `<n>\\t` from read_file is stripped for you). write_file to create or fully rewrite.
- On a tool error, read it, fix the specific cause, and retry — don't give up after one failure, and don't repeat the same failing call.
- New project → prefer a small multi-file layout with a thin entrypoint. Write code valid for the file's real language and match the project's existing conventions.
- Don't invent dependencies: before importing a library confirm it's in package.json (or requirements.txt/Cargo.toml) or a neighboring file; when installing, don't pin a guessed version — `npm install <pkg>` lets the resolver pick a real one.

Runtime facts:
- bash returns an exit code; non-zero = failure. Background long-running commands (`cmd &`) so bash returns.
- Bulky tool output from older turns is redacted to save context — re-read the file from disk if you need it again.

If the request is ambiguous in a way that changes your approach (stack, interface, scope), ask ONE short question first.

Working directory: {cwd}
{network_status}{reasoning_rules}{project_instructions}{skills_block}"""



# Reasoning-mode appendix. Injected into `SYSTEM_PROMPT` only when the
# user is in a `-think` runtime mode (see `use_thinking` at the agent
# entry). In fast mode the placeholder collapses to an empty string so
# we don't spend prompt budget guiding a channel the model isn't
# emitting into.
REASONING_RULES = """\
INTERNAL DECISION DISCIPLINE
- Keep internal reasoning brief.
- Do not restate the request.
- Do not compare many alternatives.
- Pick one safe approach quickly.
- If missing facts block progress, use tools instead of guessing.
- Do not draft full code internally; write code with write_file/edit_file.
- Do not enumerate large data internally; put it directly in files.
- After deciding, act with a tool or answer.
"""


def model_identity_line(model: str) -> str:
    """Render the one-line model self-identity for the system prompt.

    `model` is the active `config.runtime.model` value, which may be a
    GGUF filename ("gemma-4-12b-it-UD-Q4_K_XL.gguf"), a full path, or a
    short runtime tag ("gemma26b-iq3"). We resolve it to the catalog's
    friendly name + quant (e.g. "Gemma 4 12B (Q4)") via
    `models_catalog.by_filename`, which already embeds the quant in its
    `.name`. If the value isn't a catalog filename (e.g. a bare tag), we
    fall back to the stripped filename stem so the model still names
    *something* concrete rather than guessing.

    Returns a single line ending in a newline, or "" when no model is
    set (keeps the prompt prefix byte-identical for that case).
    """
    name = (model or "").strip()
    if not name:
        return ""
    # `by_filename` wants the bare filename; tolerate a full path.
    filename = Path(name).name
    try:
        from ..models_catalog import by_filename

        choice = by_filename(filename)
    except Exception:
        choice = None
    if choice is not None:
        friendly = choice.name
    else:
        # Not a catalog filename (likely a short tag). Strip a trailing
        # ".gguf" but otherwise pass it through unchanged — better to
        # echo the real configured identifier than invent a name.
        friendly = filename[:-5] if filename.lower().endswith(".gguf") else filename
    # Two-part directive. Default identity is LocalCode — the model must refer
    # to itself as "LocalCode" in greetings and ordinary replies, NOT announce
    # the raw model name (which changes per backend). But it must still know and
    # report the underlying model EXACTLY when asked — a small local model left
    # to its training priors otherwise answers "I'm Gemma by Google". So: lead
    # with LocalCode, disclose the model only on request.
    return (
        f"Your name is LocalCode (never lead with the underlying model name). "
        f"Do NOT introduce or re-introduce yourself. Never begin a message with "
        f"\"I'm LocalCode\", a greeting, or a restatement of what you're doing — "
        f"the user already knows who you are and what the task is. Just continue "
        f"the work: take the next action directly. Only state your name if the "
        f"user explicitly asks who you are. "
        f"The model currently powering you is \"{friendly}\". Report it as EXACTLY "
        f"\"{friendly}\" when the user asks which model/version/quant you are, or "
        f"when the task asks you to name a file/folder/project after your model — "
        f"in that case use \"{friendly}\" as the name and create it. Never say you "
        f"are unsure and never name a different model or provider.\n"
    )


def project_stack_line(repo_root: Path) -> str:
    """Render a one-line "Project stack:" hint from repo-root marker files.

    Scans `repo_root` for well-known dependency/config files
    (package.json, tsconfig.json, go.mod, Cargo.toml, pyproject.toml, …)
    and names the detected stack so the model writes code in the
    project's actual language and conventions rather than defaulting to
    Python idioms. The first matching marker (most-specific first) names
    the stack; any additional present markers are listed parenthetically.

    Returns a single line ending in a newline, or "" when nothing is
    detected (keeps the prompt prefix byte-identical for that case).
    """
    try:
        present = [
            (fname, label)
            for fname, label in _STACK_MARKERS
            if (repo_root / fname).is_file()
        ]
    except Exception:
        return ""
    if not present:
        return ""
    label = present[0][1]
    files = ", ".join(fname for fname, _ in present)
    return f"Project stack: {label} ({files} present) — follow its conventions.\n"


def _load_project_instructions(repo_root: Path) -> str:
    """Load project-specific instructions from a LOCALCODE.md-style
    file at `repo_root`, if one exists.

    Returns an empty string if no project file is found. On hit, the
    returned string is pre-wrapped with a "Project instructions (from
    FNAME):" header so it can be interpolated directly into the
    SYSTEM_PROMPT `{project_instructions}` slot without further
    formatting.
    """
    for name in _PROJECT_FILES:
        path = repo_root / name
        # `is_file()` both checks existence and rules out directories,
        # so a future naming collision won't crash read_text().
        if path.is_file():
            content = path.read_text(errors="replace").strip()
            if content:
                return f"\nProject instructions (from {name}):\n{content}"
    return ""
