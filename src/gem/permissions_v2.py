"""Permission system — controls what tools can do, prevents duplicates.

Based on OpenCode's session-cache pattern:
- Auto-approve safe tools (read_file, current_datetime, grep, glob)
- Ask once for risky tools (bash, write_file, edit_file), cache per session
- Block dangerous commands (rm -rf, etc.)
- Prevent duplicate tool calls in the same turn
"""
from __future__ import annotations

import sys


# Tools that NEVER need permission
ALWAYS_ALLOW = {
    "read_file", "current_datetime", "git_status", "git_diff",
    "git_log", "grep", "glob", "list_files", "search_code",
    "web_search", "web_fetch", "tool_search", "repl",
}

# Tools that need permission on first use, then cached
NEEDS_APPROVAL = {
    "bash", "write_file", "edit_file", "multi_edit",
    "git_commit", "codemod", "delegate", "security_scan",
    "run_tests",
}

# Bash commands that are always blocked
BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev",
    ":(){:|:&};:", "sudo rm", "chmod -R 777 /",
]

# Bash commands that auto-approve (safe)
SAFE_BASH = [
    "ls", "cat", "head", "tail", "wc", "find", "grep",
    "git status", "git diff", "git log", "git branch",
    "python -m py_compile", "pip list", "pip show",
    "echo", "date", "pwd", "which", "env",
]


class PermissionManager:
    """Session-scoped permission manager."""

    def __init__(self) -> None:
        self._session_approved: set[str] = set()  # tool names approved this session
        self._turn_calls: set[str] = set()  # tool calls this turn (for dedup)

    def new_turn(self) -> None:
        """Reset per-turn state (call at start of each user message)."""
        self._turn_calls.clear()

    def check(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Check if a tool call is allowed.

        Returns (allowed, reason).
        """
        # Dedup: same tool+args already called this turn
        call_key = f"{tool_name}:{sorted(args.items())}"
        if call_key in self._turn_calls:
            return False, "duplicate call (already executed this turn)"
        self._turn_calls.add(call_key)

        # Always allow safe tools
        if tool_name in ALWAYS_ALLOW:
            return True, "auto-approved (safe tool)"

        # Check bash commands specifically
        if tool_name == "bash":
            command = str(args.get("command", "")).strip().lower()
            # Block dangerous
            for blocked in BLOCKED_COMMANDS:
                if command.startswith(blocked):
                    return False, f"blocked (dangerous: {blocked})"
            # Auto-approve safe
            for safe in SAFE_BASH:
                if command.startswith(safe):
                    return True, "auto-approved (safe command)"
            # Check session cache
            if "bash" in self._session_approved:
                return True, "session-approved"
            # Ask user
            return self._ask_user(tool_name, f"$ {command}")

        # Check session cache for other tools
        if tool_name in self._session_approved:
            return True, "session-approved"

        # Ask user for approval
        args_preview = str(args)[:60]
        return self._ask_user(tool_name, args_preview)

    def _ask_user(self, tool_name: str, detail: str) -> tuple[bool, str]:
        """Single-keypress permission prompt."""
        sys.stdout.write(f"\033[2;33m  ? {tool_name}: {detail}  [y/n/a]\033[0m ")
        sys.stdout.flush()
        try:
            import tty, termios
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
            # Fallback to regular input
            ch = input().strip().lower()[:1] or "y"

        if ch in ("y", "\r", "\n"):
            return True, "approved"
        if ch == "a":
            self._session_approved.add(tool_name)
            return True, "always-approved"
        return False, "denied"
