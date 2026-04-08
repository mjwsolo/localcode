from __future__ import annotations

import time
from typing import Iterator


# ── LocalCode logo: simple, clean, no Rich markup breakage ───────────────────
# Plain text only — color is applied by the caller via Panel border/title style

GEM_LOGO_PLAIN = "◆  L O C A L  code"

# ── Logo variants ────────────────────────────────────────────────────────────

LOGO_A = """\
 █       ███████  █████     █    █          ◆
 █       █     █ █     █   █ █   █         ◆◆◆
 █       █     █ █        █   █  █          ◆
 █       █     █ █       █     █ █
 █       █     █ █       ███████ █
 █       █     █ █     █ █     █ █
 ███████ ███████  █████  █     █ ███████
  ████   ████  █████  ██████
 █    █ █    █ █    █ █
 █      █    █ █    █ █████
 █      █    █ █    █ █
 █    █ █    █ █    █ █
  ████   ████  █████  ██████"""

LOGO_B = """\
 ██        ███████   ██████     ███    ██          ◆
 ██       ██     ██ ██    ██   ██ ██   ██         ◆◆◆
 ██       ██     ██ ██        ██   ██  ██          ◆
 ██       ██     ██ ██       ██     ██ ██
 ██       ██     ██ ██       █████████ ██
 ██       ██     ██ ██    ██ ██     ██ ██
 ████████  ███████   ██████  ██     ██ ████████
  ██████   ███████  ████████  ████████
 ██    ██ ██     ██ ██     ██ ██
 ██       ██     ██ ██     ██ ██
 ██       ██     ██ ██     ██ ██████
 ██       ██     ██ ██     ██ ██
 ██    ██ ██     ██ ██     ██ ██
  ██████   ███████  ████████  ████████"""

LOGO_C = """\
  _     ___   ____    _    _        ◆
 | |   / _ \\ / ___|  / \\  | |      ◆◆◆
 | |  | | | | |     / _ \\ | |       ◆
 | |__| |_| | |___ / ___ \\| |___
 |_____\\___/ \\____/_/   \\_\\_____|
                _
   ___ ___   __| | ___
  / __/ _ \\ / _` |/ _ \\
 | (_| (_) | (_| |  __/
  \\___\\___/ \\__,_|\\___|"""

LOGO_D = """\
  _      ____   _____          _         ◆
 | |    / __ \\ / ____|   /\\   | |       ◆◆◆
 | |   | |  | | |       /  \\  | |        ◆
 | |   | |  | | |      / /\\ \\ | |
 | |___| |__| | |____ / ____ \\| |____
 |______\\____/ \\_____/_/    \\_\\______|
                _
   ___ ___   __| | ___
  / __/ _ \\ / _` |/ _ \\
 | (_| (_) | (_| |  __/
  \\___\\___/ \\__,_|\\___|"""

LOGO_E = """\
        ◆
       ◆◆◆
        ◆

  L O C A L  c o d e"""

LOGO_F = """\
 █       ███████  █████     █    █
 █       █     █ █     █   █ █   █        ◆
 █       █     █ █        █   █  █       ◆◆◆
 █       █     █ █       █     █ █        ◆
 ███████ ███████  █████  █     █ ███████
                  code"""

LOGO_VARIANTS = {
    "banner":   LOGO_A,
    "banner3":  LOGO_B,
    "standard": LOGO_C,
    "big":      LOGO_D,
    "clean":    LOGO_E,
    "compact":  LOGO_F,
}

GEM_BANNER = GEM_LOGO_PLAIN


def format_banner(
    repo_root: str,
    session_id: str,
    profile_name: str,
    model_name: str,
) -> str:
    """Build the welcome banner as plain text (no Rich markup in the logo)."""
    return (
        f"  path:  {repo_root}\n"
        f"  model: {model_name}"
    )


# ── Thinking spinner: plain text frames ──────────────────────────────────────

SPINNER_FRAMES = [
    "    *   ",
    "   * *  ",
    "  * * * ",
    " * * * *",
    "  * * * ",
    "   * *  ",
]

THINKING_LABELS = [
    "scanning code",
    "reasoning",
    "planning approach",
    "synthesizing",
    "evaluating options",
    "building solution",
]

# ── Tool execution indicators (plain text) ───────────────────────────────────

TOOL_ICONS = {
    "bash": "$",
    "grep": "grep",
    "glob": "glob",
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "replace_in_file": "edit",
    "multi_edit": "edit",
    "web_search": "web",
    "web_fetch": "fetch",
    "list_files": "ls",
    "search_code": "idx",
    "git_diff": "diff",
    "git_status": "git",
    "git_log": "log",
    "git_commit": "commit",
}


# ── Progress bar for agent steps ─────────────────────────────────────────────

def agent_progress_bar(step: int, total: int) -> str:
    filled = step
    empty = total - step
    bar = "=" * filled + "-" * empty
    return f"  [{bar}] step {step}/{total}"


# ── Public API ───────────────────────────────────────────────────────────────

def thinking_frame(step: int) -> tuple[str, str]:
    index = step % len(SPINNER_FRAMES)
    return SPINNER_FRAMES[index], THINKING_LABELS[index % len(THINKING_LABELS)]


def spinner_frame(step: int) -> str:
    return SPINNER_FRAMES[step % len(SPINNER_FRAMES)]


def snake_frame(step: int) -> str:
    """Compatibility alias."""
    return SPINNER_FRAMES[step % len(SPINNER_FRAMES)]


def tool_icon(tool_name: str) -> str:
    return TOOL_ICONS.get(tool_name, tool_name)


def center_ascii_block(block: str, width: int = 34) -> str:
    lines = [line.rstrip() for line in block.strip("\n").splitlines()]
    return "\n".join(line.center(width) for line in lines)


def format_tool_call(name: str, args_preview: str) -> str:
    icon = tool_icon(name)
    return f"  [{icon}] {name}  {args_preview}"
