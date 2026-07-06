"""write_file — create a new file with stub-code detection."""
from __future__ import annotations

import re

from .base import ToolContext

CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".swift", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".lua", ".dart", ".scala",
    ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html", ".css",
    ".vue", ".svelte", ".astro",
}

SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Create a new file or fully rewrite an existing one (not a directory). "
            "For small in-place changes prefer edit_file or multi_edit; use "
            "write_file when the new content is essentially a full replacement. "
            "To create a DIRECTORY, use the bash tool with `mkdir -p <path>` "
            "— write_file cannot create folders and will error if the path is "
            "an existing directory. Write complete useful file content. Do not "
            "split normal source files into tiny chunks. For truly huge "
            "generated data or repetitive content, prefer compact source code "
            "over streaming a massive literal into this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Complete file content"},
            },
            "required": ["path", "content"],
        },
    },
}


def _detect_stub_code(content: str, path: str) -> str | None:
    """Reject obvious stub/mock/placeholder code. Returns rejection msg or None."""
    if not any(path.endswith(ext) for ext in CODE_SUFFIXES):
        return None
    lines = content.splitlines()
    code_lines = [l for l in lines if l.strip()
                   and not l.strip().startswith("#")
                   and not l.strip().startswith("//")]
    if len(code_lines) < 3:
        return None

    # Two tiers of placeholder phrases:
    #   • UNAMBIGUOUS: phrases that ONLY appear in placeholder/stub code.
    #     A single match auto-rejects.
    #   • SOFT: phrases that COULD appear in real working code (e.g. a
    #     temperature-sensor demo that simulates sensor readings, a test
    #     fixture named `dummy_user`, a clock object called
    #     `simulated_clock`). Reject only if 2+ of these co-occur AND no
    #     unambiguous phrase is present — that combination is much more
    #     likely to be real placeholder code than legit feature code.
    #
    # Previous version put "simulates ", "simulated ", "dummy ", "fake "
    # in the unambiguous tier. That blocked write_file calls for the
    # user's "build me an app that simulates sensor readings" prompt
    # because the app's own functionality literally simulates sensors.
    # See user image 121 in session 2026-04-23.
    unambiguous = [
        "in a real implementation", "in a real scenario",
        "this is a mock", "this is a placeholder", "this is a stub",
        "# todo", "# fixme", "# hack", "# placeholder",
        "// todo", "// fixme", "// hack", "// placeholder",
        "not implemented", "raise notimplementederror",
        "unimplemented!(", "todo!(", "panic!(\"todo",
        "would call ", "would use ", "you would ",
    ]
    soft = ["in production", "simulates ", "simulated ", "dummy ", "fake "]

    lower = content.lower()
    hard_found = [p for p in unambiguous if p in lower]
    soft_found = [p for p in soft if p in lower]

    if hard_found:
        return (
            f"REJECTED: This file contains placeholder/stub code "
            f"({', '.join(hard_found[:3])}). Rewrite {path} with a REAL, "
            f"COMPLETE implementation. No 'this is a mock', no 'TODO', no "
            f"'raise NotImplementedError'. Use real libraries and write "
            f"working logic."
        )
    if len(soft_found) >= 2:
        return (
            f"REJECTED: This file looks like placeholder code — multiple "
            f"hint words found together ({', '.join(soft_found[:3])}). "
            f"If your app legitimately needs to simulate something (e.g. "
            f"a sensor demo without real hardware), that's fine — just "
            f"avoid combining it with phrases like 'in production' / "
            f"'dummy' / 'fake' that suggest the code isn't real. Write "
            f"the actual working logic."
        )

    func_count = len(re.findall(r"^\s*def ", content, re.MULTILINE))
    trivial = len(re.findall(
        r"def \w+[^:]*:\s*\n\s*(pass|return None|return \[\]|return \{\}|return \"\"|\.\.\.|return 0)\s*$",
        content, re.MULTILINE,
    ))
    if func_count >= 3 and trivial / func_count > 0.5:
        return (
            f"REJECTED: {trivial}/{func_count} functions in {path} have trivial bodies "
            f"(pass/return None). Write REAL implementations for every function."
        )
    return None


def execute(ctx: ToolContext, args: dict) -> str:
    if "path" not in args:
        return "Error: 'path' argument is required for write_file."
    path = ctx.repo / args["path"]
    # Guard: path must not already be a directory. Without this, the
    # model sometimes invokes write_file to "create a folder", which
    # then crashes with [Errno 21] Is a directory. Redirect it to the
    # right tool instead of crashing.
    if path.is_dir():
        from ..errors import LocalCodeError, by_code
        code = by_code("E4101")
        detail = (
            f"{args['path']} is a directory. Use `bash mkdir -p {args['path']}` "
            f"to create a folder; write_file only writes files."
        )
        if code is not None:
            return str(LocalCodeError(code=code, detail=detail))
        return f"[E4101] {detail}"
    content = args.get("content", "")

    # Defense-in-depth: reject the in-memory redaction stub before it
    # touches disk. Earlier versions of `_redact_old_write_args` put a
    # human-readable "[REDACTED from history — file updated at … (N
    # chars, N lines). Call read_file …]" string into the redacted
    # tool_call's `content` arg. Qwen3.6 IQ2_M would, on the next round,
    # copy that string verbatim into a new write_file call — overwriting
    # the real file with the stub itself. Redaction now
    # drops the `content` key instead of stubbing it, but a guard here
    # catches any future regression.
    if isinstance(content, str):
        stripped = content.lstrip()
        if (stripped.startswith("[REDACTED from history")
                or stripped.startswith("[REDACTED — edit applied")
                or stripped.startswith("[REDACTED edit ")):
            return (
                f"REJECTED: write_file content for {args['path']} is the "
                f"in-memory redaction stub, not real file content. The "
                f"actual file is on disk; call read_file with "
                f"path={args['path']!r} to load its real content, then "
                f"either edit_file the parts you want to change or "
                f"write_file with the genuinely new content."
            )

    stub = _detect_stub_code(content, args["path"])
    if stub:
        return stub

    path.parent.mkdir(parents=True, exist_ok=True)

    # write_file is the create-or-full-rewrite tool. Rule 9 in the
    # system prompt nudges the model toward edit_file/multi_edit for
    # in-place changes, but we don't enforce it with a REJECT — that
    # forced the model into wasteful read+plan+edit cycles every time
    # it wanted to rewrite a stub it had just scaffolded (observed
    # 2026-04-29: 21 minutes of churn on chinese-learning-app/, with
    # write→REJECT→read→edit_file rounds for hsk_data.json,
    # base.html, and app.py). The redaction-stub guard above still
    # catches the data.py-style "model copies in-memory stub back into
    # write_file" disaster, which was the one case where rejecting
    # was load-bearing.
    # Log the EXACT content the model sent (post-JSON-parse) so a "the file is
    # corrupted / write_file isn't writing what I sent" report is provable from
    # logs in one grep, instead of a mystery. Best-effort, capped, append-only.
    try:
        from ..paths import global_state_dir
        _wl = global_state_dir() / "write_content.log"
        with open(_wl, "a", encoding="utf-8", errors="replace") as _fh:
            _fh.write(f"\n===== write_file {args['path']} ({len(content)} chars) =====\n")
            _fh.write(content[:20000])
            _fh.write("\n")
    except Exception:
        pass

    existed = path.is_file()
    # Capture the prior content so an in-place rewrite renders as a diff card
    # (like edit_file/multi_edit), not a bare "Rewrote (N lines)" one-liner.
    old_content = ""
    if existed:
        try:
            old_content = path.read_text(errors="replace")
        except Exception:
            old_content = ""
    path.write_text(content)
    lines = content.count("\n") + 1
    verb = "Rewrote" if existed else "Created"

    # Emit a unified diff for rewrites of existing files so the TUI draws the
    # diff card. New files (or unchanged rewrites) keep the concise summary —
    # a full new-file dump as "+" lines is noise, and the summary reads fine.
    if existed and old_content != content:
        import difflib
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=args["path"], tofile=args["path"], lineterm="",
        ))
        if diff:
            return "\n".join(diff[:120])
    return f"{verb} {args['path']} ({lines} lines)"
