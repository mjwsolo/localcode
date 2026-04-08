"""Custom terminal input field with full layout control.

Simple approach: draws input field with rules when needed.
No scroll regions — just redraws the field after model output.
"""
from __future__ import annotations

import os
import sys
import tty
import termios
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
    """Terminal input with rules above/below and history."""

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

    def draw_busy(self, status: str = "") -> None:
        """Draw a dim input field showing model is working."""
        rule = _rule()
        sys.stdout.write(f"\n\033[2m{rule}\033[0m\n")
        sys.stdout.write(f"\033[2m  ›  \033[0m\n")
        sys.stdout.write(f"\033[2m{rule}\033[0m\n")
        if status:
            sys.stdout.write(f"\033[2m  {status}\033[0m\n")
        sys.stdout.flush()

    def read(self, status_line: str = "") -> str:
        """Show input field with rules, return user input."""
        rule = _rule()

        # Draw: top rule, input line (save cursor), bottom rule, status
        sys.stdout.write(f"\n\033[2m{rule}\033[0m\n")
        sys.stdout.write(f"  › \033[s")  # save cursor at input position
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status_line:
            sys.stdout.write(f"\n\033[2m  {status_line}\033[0m")
        sys.stdout.write(f"\033[u")  # restore to input position
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

        # Save to history
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
            if self._history_file:
                try:
                    with open(self._history_file, "a") as f:
                        f.write(f"+{text}\n")
                except Exception:
                    pass

        # After submit: clear the field area below input, redraw with final text
        sys.stdout.write(f"\033[u\033[J")  # restore to input pos, clear below
        display = text if len(text) <= 200 else text[:200] + "..."
        sys.stdout.write(f"{display}\n")
        sys.stdout.write(f"\033[2m{rule}\033[0m\n")
        sys.stdout.flush()

        return text

    def _redraw(self, buf: list[str], cursor: int, rule: str, status: str) -> None:
        """Redraw input text + bottom area from saved cursor position."""
        text = "".join(buf)
        sys.stdout.write(f"\033[u\033[J")  # restore pos, clear below
        sys.stdout.write(text)
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status:
            sys.stdout.write(f"\n\033[2m  {status}\033[0m")
        # Move cursor back to correct position
        sys.stdout.write(f"\033[u")
        if cursor > 0:
            sys.stdout.write(f"\033[{cursor}C")
        sys.stdout.flush()
