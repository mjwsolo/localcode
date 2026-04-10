"""Desktop notifications — alert user when long-running tasks finish.

Uses macOS osascript (no dependencies). Falls back to terminal bell.
Only notifies if task took longer than a threshold (default 10s).
"""
from __future__ import annotations

import subprocess
import time


# Minimum task duration (seconds) before sending notification
NOTIFY_THRESHOLD = 10.0


def notify(title: str, message: str, sound: bool = True) -> bool:
    """Send an OS-level desktop notification. Returns True if sent.

    macOS: uses osascript (always available)
    Linux: tries notify-send
    Fallback: terminal bell
    """
    import sys

    if sys.platform == "darwin":
        return _notify_macos(title, message, sound)
    elif sys.platform == "linux":
        return _notify_linux(title, message)
    else:
        _bell()
        return False


def notify_if_slow(title: str, message: str, start_time: float,
                   threshold: float = NOTIFY_THRESHOLD) -> bool:
    """Only notify if elapsed time exceeds threshold."""
    elapsed = time.time() - start_time
    if elapsed >= threshold:
        return notify(title, f"{message} ({elapsed:.0f}s)")
    return False


def _notify_macos(title: str, message: str, sound: bool = True) -> bool:
    """macOS notification via osascript."""
    # Escape quotes for AppleScript
    title = title.replace('"', '\\"').replace("'", "\\'")
    message = message.replace('"', '\\"').replace("'", "\\'")

    sound_clause = ' sound name "Glass"' if sound else ""
    script = f'display notification "{message}" with title "{title}"{sound_clause}'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, timeout=5,
        )
        return True
    except Exception:
        _bell()
        return False


def _notify_linux(title: str, message: str) -> bool:
    """Linux notification via notify-send."""
    try:
        subprocess.run(
            ["notify-send", "--app-name=gem", title, message],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _bell()
        return False


def _bell() -> None:
    """Terminal bell — universal fallback."""
    import sys
    sys.stdout.write("\a")
    sys.stdout.flush()


class TaskNotifier:
    """Context manager that notifies when a block takes too long.

    Usage:
        with TaskNotifier("Agent task"):
            run_complex_task()
        # → sends notification if it took > 10s
    """

    def __init__(self, task_name: str, threshold: float = NOTIFY_THRESHOLD) -> None:
        self.task_name = task_name
        self.threshold = threshold
        self.start_time = 0.0

    def __enter__(self) -> "TaskNotifier":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.time() - self.start_time
        if elapsed >= self.threshold:
            status = "completed" if args[0] is None else "failed"
            notify("gem", f"{self.task_name} {status} ({elapsed:.0f}s)")

    def start(self) -> None:
        """Manual start (non-context-manager usage)."""
        self.start_time = time.time()

    def finish(self, success: bool = True) -> None:
        """Manual finish."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.threshold:
            status = "completed" if success else "failed"
            notify("gem", f"{self.task_name} {status} ({elapsed:.0f}s)")
