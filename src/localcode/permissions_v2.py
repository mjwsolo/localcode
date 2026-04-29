"""Permission + safety system — controls what tools can do.

Three layers:
1. SafetyLayer — blocks dangerous commands and sensitive file writes (regex-based)
2. PermissionManager — session-scoped tool approvals (ask once, remember)
3. Dedup — prevents same tool call twice in one turn

Based on terminal coding tools's session-cache pattern + terminal coding tools's safety approach.
"""
from __future__ import annotations

import os
import re
import sys


# ── Safety Layer (always enforced, no override) ────────────────────

class SafetyLayer:
    """Block dangerous operations. Cannot be overridden by user approval."""

    BLOCKED_COMMANDS = [
        r"rm\s+(-rf?|--recursive)\s+[/~]",
        r"rm\s+-[a-zA-Z]*f[a-zA-Z]*\s+[/~]",
        r"mkfs\.",
        r"dd\s+if=",
        r">\s*/dev/sd",
        r"chmod\s+-R\s+777\s+/",
        r"curl.*\|\s*(bash|sh|zsh)",
        r"wget.*\|\s*(bash|sh|zsh)",
        r"sudo\s+rm",
        r":\(\)\{.*\}",
        r"git\s+push\s+.*--force\s+.*main",
        r"git\s+push\s+.*--force\s+.*master",
        r"git\s+reset\s+--hard\s+origin",
        r"drop\s+table",
        r"drop\s+database",
        r"truncate\s+table",
        r">(\\s)*/etc/",
        r"python.*-c.*import\s+os.*remove",
        r"eval\s*\(",
    ]

    CONFIRM_COMMANDS = [
        r"rm\s+",
        r"git\s+push",
        r"git\s+checkout\s+--",
        r"git\s+reset",
        r"pip\s+install",
        r"pip\s+uninstall",
        r"npm\s+install",
        r"npm\s+uninstall",
        r"mv\s+",
        r"docker\s+",
        r"kubectl\s+",
    ]

    SENSITIVE_FILES = [
        ".env", ".env.local", ".env.production",
        "id_rsa", "id_ed25519", "id_ecdsa",
        "credentials", "credentials.json",
        ".ssh/", "shadow", "passwd",
        ".git/config", ".gitconfig",
        "secrets.yaml", "secrets.json",
        ".aws/credentials", ".npmrc",
        "token", "api_key",
    ]

    @classmethod
    def check_command(cls, command: str) -> dict:
        """Check if a bash command is safe to run.

        Returns: {"allowed": True/False/"confirm", "reason": str}
        """
        for pattern in cls.BLOCKED_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return {"allowed": False, "reason": "Blocked: dangerous command pattern"}

        for pattern in cls.CONFIRM_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return {"allowed": "confirm", "reason": f"Needs approval: {command[:60]}"}

        return {"allowed": True}

    @classmethod
    def check_file_write(cls, file_path: str, project_root: str = "") -> dict:
        """Check if writing to a file is safe.

        Returns: {"allowed": True/False, "reason": str}
        """
        # Check sensitive files
        for pattern in cls.SENSITIVE_FILES:
            if pattern in file_path.lower():
                return {"allowed": False, "reason": f"Sensitive file: {file_path}"}

        # Check project boundary
        if project_root:
            abs_file = os.path.abspath(file_path)
            abs_root = os.path.abspath(project_root)
            if not abs_file.startswith(abs_root):
                return {"allowed": False, "reason": f"Outside project: {file_path}"}

        return {"allowed": True}

    @classmethod
    def check_edit(cls, file_path: str, old_string: str, new_string: str,
                   project_root: str = "") -> dict:
        """Check if an edit is safe."""
        write_check = cls.check_file_write(file_path, project_root)
        if not write_check["allowed"]:
            return write_check

        # Warn if deleting significantly more than adding
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        if old_lines > 30 and new_lines < old_lines * 0.3:
            return {
                "allowed": "confirm",
                "reason": f"Large deletion: removing {old_lines - new_lines} lines from {file_path}",
            }

        return {"allowed": True}


# ── Tool categories ────────────────────────────────────────────────

# Tools that NEVER need permission
ALWAYS_ALLOW = {
    "read_file", "current_datetime", "git_status", "git_diff",
    "git_log", "grep", "glob", "list_files", "search_code",
    "web_search", "web_fetch", "tool_search", "repl",
}

# Tools that need permission on first use, then cached
NEEDS_APPROVAL = {
    "bash", "write_file", "append_file", "edit_file", "multi_edit",
    "git_commit", "codemod", "delegate", "security_scan",
    "run_tests",
}

# Safe bash command prefixes (auto-approve)
SAFE_BASH = [
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg",
    "git status", "git diff", "git log", "git branch", "git show",
    "python -m py_compile", "python3 -m py_compile",
    "pip list", "pip show", "pip freeze",
    "echo", "date", "pwd", "which", "env", "whoami",
    "node --check", "npx tsc --noEmit",
    "ruff check", "pyflakes", "mypy",
    "cargo check", "go vet",
    "test -f", "test -d", "file ",
]


# ── Permission Manager ─────────────────────────────────────────────

class PermissionManager:
    """Session-scoped permission manager with integrated safety checks.

    Three checks per tool call:
    1. SafetyLayer — hard block on dangerous ops (can't override)
    2. Dedup — prevent same call twice in one turn
    3. Permission — auto-approve safe tools, ask for risky ones, cache answer
    """

    def __init__(self, project_root: str = "") -> None:
        self._session_approved: set[str] = set()
        self._turn_calls: set[str] = set()
        self._project_root = project_root

    def new_turn(self) -> None:
        """Reset per-turn state (call at start of each user message)."""
        self._turn_calls.clear()

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Check if a tool call is allowed.

        Returns (allowed: bool, reason: str).
        """
        # 0. Notebook bypass — writes/edits inside the per-session
        # notebook directory are always allowed, regardless of autonomy
        # level, safety layer, or session approvals. The notebook is
        # explicitly the agent's "scratch space" — gated by the system
        # prompt to NOT contain final user-facing artefacts, never under
        # the user's project tree, and the directory is wiped between
        # sessions. There is no scenario where prompting the user to
        # approve a notebook write adds value — it just creates UI
        # friction in `suggest` mode. `is_within_notebook` resolves
        # symlinks and rejects path traversal (`../..`) attempts, so
        # this can't be abused to escape into the project tree.
        if tool_name in ("write_file", "append_file", "edit_file", "multi_edit"):
            try:
                from pathlib import Path
                from .notebook import is_within_notebook
                raw_path = str(args.get("path", ""))
                if raw_path:
                    if is_within_notebook(Path(raw_path).expanduser()):
                        return True, "auto-approved (notebook scratch)"
            except Exception:
                # Notebook check is a fast-path optimization; on any
                # failure (import error, exotic path) fall through to
                # the normal safety / permission flow.
                pass

        # 1. Safety layer — hard blocks
        if tool_name == "bash":
            command = str(args.get("command", ""))
            safety = SafetyLayer.check_command(command)
            if safety["allowed"] is False:
                return False, safety["reason"]

        if tool_name in ("write_file", "append_file", "edit_file"):
            path = str(args.get("path", ""))
            safety = SafetyLayer.check_file_write(path, self._project_root)
            if safety["allowed"] is False:
                return False, safety["reason"]

        if tool_name == "edit_file":
            path = str(args.get("path", ""))
            old = str(args.get("old_string", ""))
            new = str(args.get("new_string", ""))
            safety = SafetyLayer.check_edit(path, old, new, self._project_root)
            if safety["allowed"] is False:
                return False, safety["reason"]
            if safety["allowed"] == "confirm" and tool_name not in self._session_approved:
                return self._ask_user(tool_name, safety["reason"])

        # 2. Dedup — same tool+args already called this turn
        call_key = f"{tool_name}:{sorted(args.items())}"
        if call_key in self._turn_calls:
            return False, "duplicate call (already executed this turn)"
        self._turn_calls.add(call_key)

        # 3. Permission check
        if tool_name in ALWAYS_ALLOW:
            return True, "auto-approved (safe tool)"

        # Bash: check safe command list
        if tool_name == "bash":
            command = str(args.get("command", "")).strip()
            for safe in SAFE_BASH:
                if command.startswith(safe) or command.lstrip().startswith(safe):
                    return True, "auto-approved (safe command)"

            # Check if safety layer flagged for confirmation
            safety = SafetyLayer.check_command(command)
            if safety["allowed"] == "confirm" and "bash" not in self._session_approved:
                return self._ask_user(tool_name, f"$ {command[:60]}")

            # Check session cache
            if "bash" in self._session_approved:
                return True, "session-approved"

            return self._ask_user(tool_name, f"$ {command[:60]}")

        # Other tools: check session cache
        if tool_name in self._session_approved:
            return True, "session-approved"

        if tool_name in NEEDS_APPROVAL:
            args_preview = str(args)[:60]
            return self._ask_user(tool_name, args_preview)

        # Unknown tool — allow by default (might be MCP/plugin)
        return True, "allowed (unknown tool)"

    def _ask_user(self, tool_name: str, detail: str) -> tuple[bool, str]:
        """Single-keypress permission prompt.

        y = yes (this time)
        n = no
        a = always (this session)
        """
        sys.stdout.write(f"\033[2;33m  ? {tool_name}: {detail}  [y/n/a]\033[0m ")
        sys.stdout.flush()
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write(f"{ch}\n")
            sys.stdout.flush()
        except Exception:
            ch = input().strip().lower()[:1] or "y"

        if ch in ("y", "\r", "\n"):
            return True, "approved"
        if ch == "a":
            self._session_approved.add(tool_name)
            return True, "always-approved (session)"
        return False, "denied by user"

    def approve_all(self) -> None:
        """Auto-approve everything (for non-interactive/agent mode)."""
        self._session_approved.update(NEEDS_APPROVAL)
        self._session_approved.add("bash")

    def is_approved(self, tool_name: str) -> bool:
        """Check if a tool is already approved (without prompting)."""
        return tool_name in ALWAYS_ALLOW or tool_name in self._session_approved
