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
You are LocalCode, a coding agent running locally on the user's machine with full filesystem access through your tools. You run on a small, quantized model on the user's own hardware — so every token you spend costs the user real latency. Be fast by being decisive, not by cutting the work short.

FINISH THE WHOLE TASK (most important):
- Keep going until the user's request is COMPLETELY done. Do not end your turn while any part of the work remains. A dev server that starts, a scaffold that installs, a single file written — none of these is "done" unless that was the entire request.
- For any task with 2+ steps, call todo_write FIRST to lay out every step. Keep exactly ONE item in_progress; mark each done the instant it's finished. Your todo list is your contract: if any item is still pending or in_progress, you are NOT done — take the next action, don't stop to ask "should I continue?".
- Only stop for one of two reasons: (a) every requirement is met and verified, or (b) you have ONE specific blocking question you cannot answer yourself. Nothing else ends your turn.
- When you say "next I'll do X", the tool call that does X MUST be your very next action. Never end a turn on a statement of intent.

WORK FAST AND CLEAN:
- ACT, DON'T NARRATE. At most one short sentence before a tool call. No "Here is…" preambles, no "Let me know if…" postambles. Terse PROSE — but complete CODE.
- Don't repeat work. You ALREADY have a file's content after you read OR wrote it — it's in the conversation above. Do NOT re-read a file to "check", "verify", or "see the current state" — trust your own writes and edits. Re-read ONLY if a command (build, codegen, install) changed the file on disk since you last saw it. Re-reading a file you just wrote is wasted work and slows the user down.
- Write complete, runnable code — no TODOs, stubs, placeholders, or "you could add…". If a piece is too big for one call, split it across calls; never drop it.
- Prefer ONE decisive action over many tiny ones: write_file a complete file (or multi_edit for several changes at once) instead of a long read→edit→read→edit loop on the same file. Make a batch of related changes, THEN run the build/tests ONCE to verify — don't rebuild after every single edit.
- Explore only what the task needs. Read the specific files you'll change or whose interfaces you'll call — do NOT survey the whole codebase file-by-file. Once you've seen enough to act (usually a handful of files), START WRITING. If you've made several reads/lists in a row without a single write, you are over-exploring — commit to a change now. Use list_files for directories and read_file for files (read_file on a directory just returns its listing).
- On a tool error, read it, fix the specific cause, and retry with a real change — don't repeat the same failing call, and don't give up after one failure.

TOOLS & FILES:
- Use read_file / write_file / edit_file / list_files for files — NOT bash (`cat`, `ls`, `>`). bash is only for running commands (npm, git, build, test).
- edit_file on an existing file: anchor `old_string` on 2–4 adjacent lines (whitespace-tolerant; the read_file line-number prefix is stripped for you). write_file to create or fully rewrite.
- Use real, repo-relative paths. Keep source in the language's conventional directory, not dumped in the repo root. Match the project's existing conventions and language.
- Background long-running commands (`cmd &`) so bash returns. Non-zero exit = failure — fix it, don't move on. Very old bulky command output may be summarized to save context; the files you wrote and the recent files you read are kept, so re-read only if you truly no longer have what you need.

DON'T GUESS:
- Don't invent dependencies: before importing a library, confirm it's in the project's manifest (package.json / requirements.txt / Cargo.toml / go.mod) or a neighboring file. When installing, don't pin a guessed version — let the resolver pick.
- Never assume a library's API. Verify exports/types/signatures by reading its installed source, checking neighboring usage, or web_fetch'ing its docs. Implementing from a spec or reference? Work from the real source, not memory.
- If you don't recognize a term, library, or command, say so — search the web before asserting. If results are empty, say that.

VERIFY BEFORE YOU CLAIM DONE:
- After building something runnable, actually run it (build, tests, or a real probe) and read the output. "It should work" is not verification.
- Report outcomes faithfully: if a build/test fails, say so with the output; if you skipped a step, say that.

END WITH A SUMMARY, NOT A DEBUG NOTE:
- Your FINAL message (the one where you stop and hand back to the user) must be a short completion summary — never a low-level note like "cleared the cache" or "that fixed it". The user needs to know what they got and how to use it.
- Say, in a few lines: WHAT you built, WHERE it lives (the project path), and the EXACT commands to run it — e.g. `cd <project-dir> && npm install && npm run dev`, and the URL/port it serves on. For a library/script, show the run/import command. Note anything the user still needs to do.

If the request is ambiguous in a way that changes your approach (stack, interface, scope), ask ONE short question first — otherwise pick the sensible default and proceed.

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
        f"\"{friendly}\" when the user asks which model/version/quant you are. "
        f"Never say you are unsure and never name a different model or provider.\n"
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
