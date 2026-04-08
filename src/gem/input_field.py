"""Custom terminal input field with type-ahead queuing.

Input is always active. If user submits while model is busy,
the message is queued and auto-submitted when model finishes.
Based on Claude Code's input architecture.
"""
from __future__ import annotations

import os
import sys
import tty
import termios
import threading
import select
from pathlib import Path


def _get_cols() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _rule() -> str:
    w = max(24, min(96, _get_cols() - 4))
    return "  " + ("─" * w)


class InputField:
    """Terminal input with type-ahead queuing during model work."""

    def __init__(self, history_file: Path | None = None):
        self._history: list[str] = []
        if history_file and history_file.exists():
            try:
                lines = history_file.read_text(errors="replace").splitlines()
                self._history = [l[1:] for l in lines if l.startswith("+")]
                self._history = self._history[-200:]
            except Exception:
                pass
        self._history_file = history_file
        # Queue for type-ahead
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._busy = False

    @property
    def has_queued(self) -> bool:
        with self._lock:
            return len(self._queue) > 0

    def dequeue(self) -> str | None:
        """Pop the next queued message, or None."""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None

    def set_busy(self, busy: bool) -> None:
        with self._lock:
            self._busy = busy

    def read(self, status_line: str = "") -> str:
        """Show input field, return user input. Checks queue first."""
        # If there's a queued message from type-ahead, return it immediately
        queued = self.dequeue()
        if queued:
            rule = _rule()
            # Show the queued message as if user just typed it
            sys.stdout.write(f"\n\033[2m{rule}\033[0m\n")
            sys.stdout.write(f"  › {queued}\n")
            sys.stdout.write(f"\033[2m{rule}\033[0m\n")
            sys.stdout.flush()
            return queued

        rule = _rule()

        # Draw: top rule, save pos at start of input line, draw input + bottom
        sys.stdout.write(f"\n\033[2m{rule}\033[0m\n")
        sys.stdout.write(f"\033[s")  # save at START of input line
        sys.stdout.write(f"  › ")
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status_line:
            sys.stdout.write(f"\n\033[2m  {status_line}\033[0m")
        # Move cursor back to input position (after "  › ")
        sys.stdout.write(f"\033[u\033[4C")
        sys.stdout.flush()

        buf = []
        cursor = 0
        saved_for_history = ""
        hist_idx = len(self._history)

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)

                if ch == "\r" or ch == "\n":
                    break
                elif ch == "\x03":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    raise KeyboardInterrupt
                elif ch == "\x04":
                    if not buf:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        raise EOFError
                elif ch == "\x7f" or ch == "\x08":
                    if cursor > 0:
                        buf.pop(cursor - 1)
                        cursor -= 1
                        self._redraw(buf, cursor, rule, status_line)
                elif ch == "\x1b":
                    seq1 = sys.stdin.read(1)
                    if seq1 == "[":
                        seq2 = sys.stdin.read(1)
                        if seq2 == "D" and cursor > 0:
                            cursor -= 1
                            sys.stdout.write("\033[D")
                            sys.stdout.flush()
                        elif seq2 == "C" and cursor < len(buf):
                            cursor += 1
                            sys.stdout.write("\033[C")
                            sys.stdout.flush()
                        elif seq2 == "A" and hist_idx > 0:
                            if hist_idx == len(self._history):
                                saved_for_history = "".join(buf)
                            hist_idx -= 1
                            buf = list(self._history[hist_idx])
                            cursor = len(buf)
                            self._redraw(buf, cursor, rule, status_line)
                        elif seq2 == "B" and hist_idx < len(self._history):
                            hist_idx += 1
                            buf = list(saved_for_history) if hist_idx == len(self._history) else list(self._history[hist_idx])
                            cursor = len(buf)
                            self._redraw(buf, cursor, rule, status_line)
                        elif seq2 == "3":
                            sys.stdin.read(1)  # consume ~
                            if cursor < len(buf):
                                buf.pop(cursor)
                                self._redraw(buf, cursor, rule, status_line)
                    elif seq1 in ("\r", "\n"):
                        break
                elif ch == "\x15":  # Ctrl+U
                    buf.clear()
                    cursor = 0
                    self._redraw(buf, cursor, rule, status_line)
                elif ch == "\x01":  # Ctrl+A
                    cursor = 0
                    self._redraw(buf, cursor, rule, status_line)
                elif ch == "\x05":  # Ctrl+E
                    cursor = len(buf)
                    self._redraw(buf, cursor, rule, status_line)
                elif ch == "\x17":  # Ctrl+W
                    while cursor > 0 and buf[cursor - 1] == " ":
                        buf.pop(cursor - 1)
                        cursor -= 1
                    while cursor > 0 and buf[cursor - 1] != " ":
                        buf.pop(cursor - 1)
                        cursor -= 1
                    self._redraw(buf, cursor, rule, status_line)
                elif ord(ch) >= 32:
                    buf.insert(cursor, ch)
                    cursor += 1
                    if cursor == len(buf):
                        sys.stdout.write(ch)
                        sys.stdout.flush()
                    else:
                        self._redraw(buf, cursor, rule, status_line)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        text = "".join(buf).strip()
        self._save_history(text)

        # After submit: restore to line start, clear, show final state
        sys.stdout.write(f"\033[u\033[J")
        display = text if len(text) <= 200 else text[:200] + "..."
        sys.stdout.write(f"  › {display}\n")
        sys.stdout.write(f"\033[2m{rule}\033[0m\n")
        sys.stdout.flush()

        return text

    def collect_typeahead(self) -> None:
        """Non-blocking: collect keystrokes into queue while model works.

        Called by indicator thread. Sets raw mode once, reads all available
        chars, then restores. The indicator thread redraws the input field
        showing the buffer text.
        """
        buf = getattr(self, '_typeahead_buf', [])
        fd = sys.stdin.fileno()
        try:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return  # nothing to read
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)  # cbreak: chars available immediately, no echo
                while select.select([sys.stdin], [], [], 0)[0]:
                    ch = os.read(fd, 1).decode("utf-8", errors="replace")
                    if ch in ("\r", "\n"):
                        text = "".join(buf).strip()
                        if text:
                            with self._lock:
                                self._queue.append(text)
                            self._save_history(text)
                        buf.clear()
                    elif ch == "\x7f" or ch == "\x08":
                        if buf:
                            buf.pop()
                    elif ch == "\x03":
                        buf.clear()
                    elif ch == "\x15":  # Ctrl+U
                        buf.clear()
                    elif len(ch) == 1 and ord(ch) >= 32:
                        buf.append(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        self._typeahead_buf = buf

    def get_typeahead_text(self) -> str:
        """Return current typeahead buffer text (for display in indicator)."""
        buf = getattr(self, '_typeahead_buf', [])
        return "".join(buf)

    def _save_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
            if self._history_file:
                try:
                    with open(self._history_file, "a") as f:
                        f.write(f"+{text}\n")
                except Exception:
                    pass

    def _redraw(self, buf: list[str], cursor: int, rule: str, status: str) -> None:
        """Redraw entire input area from saved line start position."""
        text = "".join(buf)
        # Restore to start of input line, clear everything below
        sys.stdout.write(f"\033[u\033[J")
        # Redraw: prefix + text + bottom
        sys.stdout.write(f"  › {text}")
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status:
            sys.stdout.write(f"\n\033[2m  {status}\033[0m")
        # Move cursor to correct position: restore to line start, move right
        sys.stdout.write(f"\033[u\033[{4 + cursor}C")
        sys.stdout.flush()
