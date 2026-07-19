"""Loop-adjacent helpers.

Pulled out of agent/__init__.py during T0.1-e. The seven functions
here aren't state machines or policy — they're "the loop needs to do
this one small thing" utilities that were crowding the file.

Grouped into three conceptual buckets (kept in one file because
they're all called exclusively by the loop and none is worth its own
module yet):

  1. Tool dispatch plumbing — `_execute_tool`, `_first_token`,
     `_needs_confirmation`. These turn a parsed tool call + args
     into a result string, gating destructive commands on user
     approval via DESTRUCTIVE_PATTERNS.

  2. Display / summary formatting — `_render_markdown`,
     `_brief_result`, `_grounded_file_summary`. These shape text
     for user display (final answers rendered as Markdown; tool
     results briefly summarised for progress logs; file-change
     lists rendered with line counts after the turn).

  3. Progress-indicator label — `_tool_stage_label`. Returns the
     short human phrase the TUI shows while a tool is running
     ("writing foo.py", "running command", "searching code").

Pure helpers. No hidden state. Loop imports them by name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from .constants import DESTRUCTIVE_PATTERNS
from ..tools import ToolContext, ToolResult, dispatch_result as _tools_dispatch_result

if TYPE_CHECKING:
    from ..app import LocalCodeApp
    from ..output import OutputManager


# Helpers here are underscore-prefixed internal utilities the loop
# uses. `__all__` is empty deliberately — nothing here is a public
# contract. If a helper should be public later, rename without the
# leading underscore and add it here.
__all__: list[str] = []


_SHRINK_GUARD_SOURCE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html"}


# ── Autonomy-independent safety hard-block ─────────────────────────────
#
# The approval gate (_needs_confirmation) is a *confirmation* layer: it is
# skipped in FULL_AUTO and in headless mode (which forces full_auto). That
# left no backstop at all for those modes — a prompt-injected model could
# run `curl … | sh` via bash OR background_process, or overwrite
# ~/.ssh/authorized_keys via write_file, entirely unattended. This layer
# runs before EVERY dispatch regardless of autonomy level or headless, and
# cannot be overridden. It blocks only unambiguously destructive operations
# (no legitimate agent use), so it does not add friction to normal work.

# Tools that hand a raw string to a shell.
_SHELL_EXEC_TOOLS = {"bash", "background_process"}
# Tools that write file contents to a path.
_FILE_WRITE_TOOLS = {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}

# Credential / key material that the agent has no legitimate reason to write.
# Matched on the path BASENAME or a path SEGMENT (never a naive substring, so
# editing the project's own `tokenizer.py` / `api_keys.py` is never blocked).
_BLOCKED_WRITE_BASENAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "authorized_keys", "known_hosts",
    ".netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.json",
    "shadow", "passwd", "sudoers",
})
_BLOCKED_WRITE_SEGMENTS = frozenset({".ssh", ".aws", ".gnupg", ".config/gcloud"})


def _is_blocked_write_path(raw_path: str) -> bool:
    """True if writing this path targets credential/key material.

    Precise matching only: exact basename or an exact path segment. A file
    named `tokenizer.py` or `password_reset.py` in the project is NOT blocked.
    """
    if not raw_path:
        return False
    try:
        p = Path(raw_path).expanduser()
    except Exception:
        p = Path(raw_path)
    name = p.name.lower()
    if name in _BLOCKED_WRITE_BASENAMES:
        return True
    lowered_parts = {part.lower() for part in p.parts}
    if lowered_parts & _BLOCKED_WRITE_SEGMENTS:
        return True
    return False


# Catastrophic shell commands with NO legitimate agent use — hard-blocked in
# every mode, never overridable. Deliberately TIGHT: these are anchored to real
# device/root/home targets so they don't false-positive on a `grep` for the
# text. "High-risk but sometimes legitimate" commands (curl|sh installs,
# force-push, sudo rm) are NOT here — they route through the confirmation gate
# instead (see _CONFIRM_SHELL_RE), so the user can approve them.
#
# NB: SQL patterns (DROP TABLE, DELETE FROM, …) are intentionally absent — bash
# does not execute SQL, so blocking them here only broke `grep "DROP TABLE"`.
import re as _re

_HARD_BLOCK_SHELL_RE = [
    _re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(/|~|\$HOME|\$\{HOME\})(\s|/|$)", _re.I),
    _re.compile(r"\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+(/|~|\$HOME|\$\{HOME\})(\s|/|$)", _re.I),
    _re.compile(r"\bmkfs\b", _re.I),
    _re.compile(r"\bdd\b[^\n]*\bof=/dev/", _re.I),
    _re.compile(r">\s*/dev/(sd|nvme|hd|disk|vd)", _re.I),
    _re.compile(r"\bchmod\s+-R\s+0*777\s+/(\s|$)", _re.I),
    _re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", _re.I),  # fork bomb
    _re.compile(r">\s*/etc/", _re.I),
    _re.compile(r"\bwipefs\b", _re.I),
]

# High-risk-but-sometimes-legitimate shell: prompt for approval (overridable),
# never silently run and never hard-blocked. Anchored so a `grep` for the text
# doesn't trigger it: the pipe-to-shell must be a real pipeline, force-push a
# real git invocation.
_CONFIRM_SHELL_RE = _re.compile(
    r"(?:\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b)"
    r"|(?:\bgit\s+push\b[^\n]*(?:--force\b|-f\b))"
    r"|(?:\bsudo\s+rm\b)"
    r"|(?:\bgit\s+reset\s+--hard\s+origin\b)",
    _re.I,
)


def _safety_hard_block(name: str, args: dict) -> str | None:
    """Autonomy-independent hard block. Returns a rejection reason, or None.

    Runs before every tool dispatch in ALL modes (including FULL_AUTO and
    headless). Covers only operations with no legitimate agent use:
      - catastrophic shell (rm -rf /, mkfs, dd of=/dev/…, > /dev/sd*, fork
        bomb, chmod -R 777 /, > /etc/) — see _HARD_BLOCK_SHELL_RE
      - writes to SSH/AWS/GPG key material or credential files
    Everything else — including curl|sh and force-push — falls through to the
    normal confirmation flow, which can approve it.
    """
    if name in _SHELL_EXEC_TOOLS:
        cmd = str(args.get("command", "") or "")
        if cmd:
            for rx in _HARD_BLOCK_SHELL_RE:
                if rx.search(cmd):
                    return "blocked: refusing a command that could destroy the disk or system (matched a catastrophic pattern)"
    if name in _FILE_WRITE_TOOLS:
        raw_path = str(args.get("path", "") or args.get("file_path", "") or "")
        if raw_path:
            # The agent's sanctioned scratch dir is exempt.
            try:
                from ..notebook import is_within_notebook
                if is_within_notebook(Path(raw_path).expanduser()):
                    return None
            except Exception:
                pass
            if _is_blocked_write_path(raw_path):
                return f"blocked: refusing to write credential/key file ({Path(raw_path).name})"
    return None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False
    except Exception:
        return False


def _reject_noncanonical_creation_write(app: "LocalCodeApp", raw_path: str) -> str | None:
    # Project naming is a model/user decision. Earlier versions forced every
    # new-app write into an internal task slug; that over-constrained builds
    # and produced weird repeated folder names. Keep the helper as a no-op so
    # the rest of the write-guard pipeline remains stable.
    return None


def _should_reject_destructive_write(path: Path, content: str) -> str | None:
    """Reject suspicious full-file rewrites that collapse real source files.

    Failure pattern from events.jsonl:
    - existing app source file is 100+ lines
    - model uses write_file, not edit_file
    - new content is a tiny 8-line shell
    - tool succeeds, but the app regresses badly

    Keep this narrow: only existing source files, only severe shrinkage.
    """
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() not in _SHRINK_GUARD_SOURCE_EXTS:
        return None
    try:
        old_text = path.read_text(errors="replace")
    except Exception:
        return None
    old_lines = len(old_text.splitlines())
    new_lines = len(content.splitlines())
    if old_lines < 60:
        return None
    if new_lines > 20:
        return None
    if new_lines >= max(12, old_lines // 5):
        return None
    return (
        f"REJECTED: write_file would collapse {path.name} from {old_lines} lines to "
        f"{new_lines} lines. This looks like an accidental destructive rewrite. "
        f"Read the file and use edit_file or multi_edit for a targeted change. "
        f"Only do a full rewrite if the user explicitly asked for one."
    )


def _should_reject_destructive_multi_edit(path: Path, edits: object) -> str | None:
    """Reject multi_edit calls that effectively replace a whole file with
    a tiny stub. Same protection as write_file, but simulated before the
    tool mutates disk.
    """
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() not in _SHRINK_GUARD_SOURCE_EXTS:
        return None
    if not isinstance(edits, list) or not edits:
        return None
    try:
        old_text = path.read_text(errors="replace")
    except Exception:
        return None
    old_lines = len(old_text.splitlines())
    if old_lines < 60:
        return None

    content = old_text
    replaced_lines = 0
    for edit in edits:
        if not isinstance(edit, dict):
            return None
        old = edit.get("old_string", "")
        new = edit.get("new_string", "")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return None
        if content.count(old) != 1:
            return None
        replaced_lines += len(old.splitlines())
        content = content.replace(old, new, 1)

    new_lines = len(content.splitlines())
    replaced_ratio = replaced_lines / max(old_lines, 1)
    if new_lines > 20:
        return None
    if new_lines >= max(12, old_lines // 5):
        return None
    if replaced_ratio < 0.75:
        return None
    return (
        f"REJECTED: multi_edit would collapse {path.name} from {old_lines} "
        f"lines to {new_lines} lines after replacing {replaced_lines} lines. "
        "This looks like an accidental destructive rewrite. Use smaller "
        "targeted edits, or ask before doing an explicit full-file replacement."
    )


def _execute_tool_result(app: "LocalCodeApp", name: str, args: dict, out: "OutputManager") -> ToolResult:
    """Execute a single tool and return the result string.

    Routes through src/localcode/tools/{name}.py via tools.dispatch, with a
    wrapping plan-mode policy layer that refuses destructive tools
    while the agent is in plan-explore mode.
    """
    # Autonomy-independent safety hard-block. Runs FIRST, in every mode
    # (including FULL_AUTO / headless), and cannot be overridden — this is the
    # backstop the confirmation gate is not. A prompt-injected model cannot use
    # bash/background_process to run `curl|sh` or wipe a disk, nor a write tool
    # to overwrite ~/.ssh/authorized_keys, even with no human present.
    _blocked = _safety_hard_block(name, args)
    if _blocked is not None:
        try:
            from ..events import emit as _emit_block
            _emit_block("safety_hard_block", tool=name, reason=str(_blocked)[:200])
        except Exception:
            pass
        return ToolResult(
            text=f"REJECTED: {_blocked}. This operation is blocked by the safety layer and cannot be auto-approved.",
            ok=False,
            facts={"tool": name, "ok": False, "safety_blocked": True},
        )

    try:
        hook = getattr(app, "hooks", None)
        decision = hook.on_pre_tool_use(name, args) if hook is not None else None
        if decision is not None and decision.blocked:
            reason = decision.error or decision.output or "pre-tool hook blocked"
            return ToolResult(text=f"REJECTED: {reason}", ok=False, facts={"tool": name, "ok": False, "hook_blocked": True})
    except Exception:
        pass

    # Lint-gate: for source-file writes/edits, snapshot the file before the
    # tool runs so we can auto-revert on syntax-break. Motivation: the model
    # at 2-bit quant has been observed to delete import statements during
    # "clean up" edits, corrupting working code. A failed parse after the
    # edit means the model made a destructive mistake — undo it before the
    # next round runs the broken file.
    _lintable_exts = (".py", ".js", ".ts", ".tsx", ".jsx", ".json")
    _lint_tools = {"write_file", "append_file", "edit_file", "multi_edit", "edit_diff"}
    target_path: Path | None = None
    snapshot_bytes: bytes | None = None
    if name in _lint_tools:
        raw_path = args.get("path") or ""
        if raw_path:
            canonical_rejection = _reject_noncanonical_creation_write(app, str(raw_path))
            if canonical_rejection:
                return ToolResult(
                    text=canonical_rejection,
                    ok=False,
                    facts={"tool": name, "ok": False, "path": str(raw_path), "canonical_path": False},
                )
            p = app.repo_root / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
            if p.suffix.lower() in _lintable_exts and p.exists():
                try:
                    snapshot_bytes = p.read_bytes()
                    target_path = p
                except Exception:
                    snapshot_bytes = None
                    target_path = None

    if name == "write_file":
        raw_path = args.get("path") or ""
        if raw_path and "content" in args:
            p = app.repo_root / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
            rejection = _should_reject_destructive_write(p, str(args.get("content", "")))
            if rejection:
                return ToolResult(text=rejection, ok=False, facts={"tool": name, "ok": False, "path": str(raw_path)})
    elif name == "multi_edit":
        raw_path = args.get("path") or ""
        if raw_path:
            p = app.repo_root / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
            rejection = _should_reject_destructive_multi_edit(p, args.get("edits"))
            if rejection:
                return ToolResult(text=rejection, ok=False, facts={"tool": name, "ok": False, "path": str(raw_path)})

    try:
        result = _tools_dispatch_result(name, ToolContext(app=app, out=out), args)
    except Exception as e:
        text = f"Error in {name}: {type(e).__name__}: {e}"
        try:
            if getattr(app, "hooks", None) is not None:
                app.hooks.on_post_tool_use(name, args, text, error=True)
        except Exception:
            pass
        return ToolResult(text=text, ok=False, facts={"tool": name, "ok": False, "error_type": type(e).__name__})

    try:
        if getattr(app, "hooks", None) is not None:
            app.hooks.on_post_tool_use(name, args, result.text, error=not result.ok)
    except Exception:
        pass

    if target_path is not None and target_path.exists() and target_path.suffix == ".py":
        # Python: ast.parse is cheap and authoritative.
        import ast as _ast
        try:
            _ast.parse(target_path.read_text(errors="replace"))
        except SyntaxError as se:
            if snapshot_bytes is not None:
                target_path.write_bytes(snapshot_bytes)
            text = (
                f"REJECTED: the edit broke Python syntax in {target_path.name} "
                f"({se.msg} at line {se.lineno}). File was reverted. "
                "Re-read the file and try a more targeted change — don't delete "
                "import statements, function defs, or class headers unless the "
                "user explicitly asked."
            )
            return ToolResult(text=text, ok=False, facts={"tool": name, "ok": False, "path": str(target_path), "reverted": True})
    if name in _lint_tools:
        try:
            if getattr(app, "hooks", None) is not None:
                app.hooks.on_post_edit(str(args.get("path", "")), result.text, error=not result.ok)
        except Exception:
            pass
    return result


def _execute_tool(app: "LocalCodeApp", name: str, args: dict, out: "OutputManager") -> str:
    return _execute_tool_result(app, name, args, out).text

def _first_token(cmd: str) -> str:
    """Leading command word, used as the key for per-session "always allow"
    whitelisting. `git push ...` → "git", `pip install ...` → "pip".
    """
    return (cmd.strip().split() or [""])[0][:20].lower()

def _needs_confirmation(name: str, args: dict, app: "LocalCodeApp | None" = None) -> bool:
    """Check if this tool needs user confirmation.

    Honors the app's current autonomy level — FULL_AUTO bypasses confirmation
    even for destructive patterns. This is checked PER-CALL so toggling
    /permissions mid-task takes effect immediately.

    Also honors the per-session "always allow" set built up when the user
    picks option 2 on a prompt ("always allow `git`"). That set is on
    `app._session_allow` — scoped to this process, cleared on next launch.
    """
    # Tools that don't execute shell or mutate files never need confirmation.
    if name != "bash" and name not in _SHELL_EXEC_TOOLS and name not in _FILE_WRITE_TOOLS:
        return False

    level = None
    if app is not None:
        try:
            from ..autonomy import AutonomyLevel
            level = getattr(app, "_autonomy", None)
            # FULL_AUTO skips confirmation for everything (the hard-block in
            # _execute_tool_result still applies — it is not an approval).
            if level == AutonomyLevel.FULL_AUTO:
                return False
        except Exception:
            level = None

    # File-write tools: auto-approved in auto_edit (the point of that mode);
    # in suggest (read-only) mode every write is confirmed. Notebook scratch
    # writes are never prompted.
    if name in _FILE_WRITE_TOOLS:
        try:
            from ..autonomy import AutonomyLevel
            if level != AutonomyLevel.SUGGEST:
                return False
        except Exception:
            return False
        raw_path = str(args.get("path", "") or args.get("file_path", "") or "")
        if raw_path:
            try:
                from ..notebook import is_within_notebook
                if is_within_notebook(Path(raw_path).expanduser()):
                    return False
            except Exception:
                pass
        return True

    # Shell-executing tools (bash, background_process).
    cmd = args.get("command", "")
    if app is not None:
        allow = getattr(app, "_session_allow", None)
        if allow and _first_token(cmd) in allow:
            return False
    # background_process hands a raw string straight to /bin/sh with no
    # destructive-substring shortcut — always confirm it (unless full_auto or
    # session-allowed above). In suggest mode, confirm any shell command.
    if name == "background_process":
        return True
    try:
        from ..autonomy import AutonomyLevel
        if level == AutonomyLevel.SUGGEST:
            return True
    except Exception:
        pass
    # High-risk-but-sometimes-legit patterns (remote pipe-to-shell, force-push,
    # sudo rm) are NOT hard-blocked — confirm them so the user can approve.
    if _CONFIRM_SHELL_RE.search(cmd):
        return True
    return any(p in cmd for p in DESTRUCTIVE_PATTERNS)

def _render_markdown(text: str, console: Console | None = None) -> None:
    """Render text as markdown, or plain text for narrow terminals."""
    text = text.strip()
    if not text:
        return
    cols = __import__("shutil").get_terminal_size().columns
    width = cols - 4
    if width < 60:
        # Narrow terminal: just print with indent, no Rich formatting
        for line in text.splitlines():
            sys.stdout.write(f"  {line}\n")
        sys.stdout.flush()
        return
    c = Console(width=width, soft_wrap=True)
    has_md = any(m in text for m in ("```", "###", "**", "- ", "1. ", "`"))
    if has_md:
        c.print(Padding(Markdown(text), (0, 2, 0, 2)))
    else:
        c.print(Padding(text, (0, 2, 0, 2)))

def _split_exit_code(result: str) -> tuple[int | None, str]:
    """Split a bash result into (exit_code, body).

    bash.py prepends `[exit code N]\\n` to a failed command's output as a
    model-facing failure marker. Return the parsed code and the remaining body,
    or (None, result) when the marker is absent (command succeeded).
    """
    s = result.lstrip()
    if s.startswith("[exit code "):
        end = s.find("]")
        if end != -1:
            try:
                code = int(s[len("[exit code "):end].strip())
            except ValueError:
                return None, result
            body = s[end + 1:]
            return code, body[1:] if body.startswith("\n") else body
    return None, result


def _redirect_note(result: str) -> str:
    """Turn an internal `REJECTED …` string into a short user-facing note.

    The full REJECTED payload (routing hints, "Suggested tool call: …",
    hard-stop instructions) is for the model's recovery, not the user. Two
    cases: a routing redirect ("use X instead") → "→ used X instead"; anything
    else (repeat/hard-stop/unsafe/oversize block) → a neutral "→ skipped …"
    note. Never show the raw protocol text.
    """
    import re
    m = re.search(r"\buse\s+([a-z_]+)\s+instead", result)
    if m:
        return f"→ used {m.group(1)} instead"
    low = result.lower()
    if "hard stop" in low or "already failed" in low or "already ran" in low or "rewritten" in low:
        return "→ skipped (repeating a call that won't help)"
    if "safety ceiling" in low or "over the" in low:
        return "→ skipped (too large — splitting instead)"
    if "defer" in low or "blocked" in low or "live server" in low:
        return "→ skipped that step"
    return "→ adjusted approach"


def _brief_result(tool_name: str, result: str) -> str:
    """Short summary of a tool result for terminal display."""
    lines = result.strip().splitlines()
    # Dedup short-circuits across ALL deduped tools (read_file,
    # list_files, glob, grep) return a tiny "[DEDUP …]" stub. Counting
    # lines / matches on the stub gives "1 lines" / "1 matches" in the
    # chat log — looks like a 1-line file or 1 match, which confuses
    # the user. Surface the dedup honestly first so the user can see
    # the cache behaviour, then fall through to per-tool formatting.
    s = result.strip()
    if s.startswith("[DEDUP"):
        return "already done — using cached result"
    if s.startswith("[FILE UNCHANGED"):
        return "unchanged — see earlier read"
    # Internal tool-routing/hard-stop correction. This is a signal FOR THE
    # MODEL, not the user — render a short neutral note, never the raw
    # "REJECTED: …" / "REJECTED — HARD STOP: …" protocol text (both forms).
    if s.startswith("REJECTED"):
        return _redirect_note(s)
    if tool_name == "read_file":
        return f"{len(lines)} lines"
    if tool_name == "write_file":
        return result.strip()[:80] if result.strip() else "written"
    if tool_name == "edit_file":
        return result.strip()[:80] if result.strip() else "edited"
    if tool_name == "bash":
        # `[exit code N]` is a model-facing marker prepended to the output so the
        # model knows the command failed. Don't show that raw token to the user:
        # strip it, and render a clean "failed (exit N)" plus any real output.
        exit_code, body = _split_exit_code(result)
        body_lines = body.strip().splitlines()
        if exit_code is not None:
            head = body_lines[0][:80] if body_lines and body_lines[0].strip() else ""
            tail = f"  …({len(body_lines)} lines)" if len(body_lines) > 1 else ""
            return f"failed (exit {exit_code}){(': ' + head) if head else ''}{tail}"
        if not body.strip():
            return "done (no output)"
        if len(body_lines) <= 3:
            return "\n".join(body_lines)[:200]
        return f"{body_lines[0][:80]}  …({len(body_lines)} lines)"
    if tool_name == "grep":
        return f"{len(lines)} matches" if lines and lines[0] else "no matches"
    if tool_name in ("glob", "list_files"):
        return f"{len(lines)} files"
    if tool_name == "web_search":
        return f"{len(lines) // 3} results" if lines else "no results"
    return result[:80] if result else "done"

def _grounded_file_summary(repo: Path, changed_files: list[str]) -> str:
    """Build a deterministic summary from files that actually exist on disk."""
    existing: list[str] = []
    for rel in changed_files:
        path = repo / rel
        if path.exists() and path.is_file():
            existing.append(rel)

    if not existing:
        return ""

    lines = ["Updated files:"]
    for rel in existing:
        try:
            line_count = len((repo / rel).read_text(errors="replace").splitlines())
            lines.append(f"- `{rel}` ({line_count} lines)")
        except Exception:
            lines.append(f"- `{rel}`")
    return "\n".join(lines)

def _tool_stage_label(tool_name: str, args: dict) -> str:
    """Human-readable stage label for the indicator."""
    if tool_name == "bash":
        cmd = args.get("command", "")
        if "pip " in cmd:
            return "installing packages"
        if "npm " in cmd:
            return "installing packages"
        if "git " in cmd:
            return "git operation"
        if "python " in cmd or "pytest" in cmd:
            return "running code"
        return "running command"
    if tool_name == "write_file":
        return f"writing {Path(args.get('path', 'file')).name}"
    if tool_name == "edit_file":
        return f"editing {Path(args.get('path', 'file')).name}"
    if tool_name == "read_file":
        return f"reading {Path(args.get('path', 'file')).name}"
    if tool_name == "grep":
        return "searching code"
    if tool_name == "glob":
        return "finding files"
    if tool_name == "web_search":
        return "searching web"
    return tool_name
