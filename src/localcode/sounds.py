"""Notification sounds for turn completion + approval prompts.

Macos-only via `afplay`. Non-blocking (Popen + detach). Silent no-op
on other platforms or when the system sound files don't exist.

Two events get sounds:
  • completion — turn finished, ready for next user input
  • approval   — agent needs the user to allow/deny a destructive cmd

Both are toggled by `ui.sounds_enabled` (default False — opt-in so
users in libraries / open offices aren't surprised).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_COMPLETION_PATH = "/System/Library/Sounds/Glass.aiff"
_APPROVAL_PATH = "/System/Library/Sounds/Tink.aiff"


def _afplay(path: str) -> None:
    """Fire-and-forget play of an AIFF via afplay. Returns immediately;
    the spawned process detaches and finishes on its own ~0.5s later."""
    if sys.platform != "darwin":
        return
    if not Path(path).exists():
        return
    try:
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def play_completion(enabled: bool) -> None:
    if enabled:
        _afplay(_COMPLETION_PATH)


def play_approval(enabled: bool) -> None:
    if enabled:
        _afplay(_APPROVAL_PATH)
