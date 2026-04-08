"""Custom terminal input field with full layout control.

Draws rules above and below the input area, handles editing,
history, and paste. No prompt_toolkit — raw terminal I/O via
the standard library.
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
        self._history_pos: int = -1
        if history_file and history_file.exists():
            try:
                lines = history_file.read_text(errors="replace").splitlines()
                # prompt_toolkit history format: lines starting with +
                self._history = [l[1:] for l in lines if l.startswith("+")]
                # Keep last 200
                self._history = self._history[-200:]
            except Exception:
                pass
        self._history_file = history_file

    def read(self, status_line: str = "") -> str:
        """Show input field with rules, return user input."""
        rule = _rule()

        # Draw: top rule, input line, bottom rule, status
        sys.stdout.write(f"\n\033[2m{rule}\033[0m\n")
        sys.stdout.write(f"  › ")
        # Save the row where input starts
        sys.stdout.write(f"\033[s")  # save cursor
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status_line:
            sys.stdout.write(f"\n\033[2m  {status_line}\033[0m")
        # Move back to input line
        sys.stdout.write(f"\033[u")  # restore cursor
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
                    # Submit
                    break

                elif ch == "\x03":
                    # Ctrl+C
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    raise KeyboardInterrupt

                elif ch == "\x04":
                    # Ctrl+D
                    if not buf:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        raise EOFError
                    # Ignore if buffer not empty

                elif ch == "\x7f" or ch == "\x08":
                    # Backspace
                    if cursor > 0:
                        buf.pop(cursor - 1)
                        cursor -= 1
                        self._redraw_input(buf, cursor, rule, status_line)

                elif ch == "\x1b":
                    # Escape sequence
                    seq1 = sys.stdin.read(1)
                    if seq1 == "[":
                        seq2 = sys.stdin.read(1)
                        if seq2 == "D":  # Left
                            if cursor > 0:
                                cursor -= 1
                                sys.stdout.write("\033[D")
                                sys.stdout.flush()
                        elif seq2 == "C":  # Right
                            if cursor < len(buf):
                                cursor += 1
                                sys.stdout.write("\033[C")
                                sys.stdout.flush()
                        elif seq2 == "A":  # Up — history
                            if hist_idx > 0:
                                if hist_idx == len(self._history):
                                    saved_for_history = "".join(buf)
                                hist_idx -= 1
                                buf = list(self._history[hist_idx])
                                cursor = len(buf)
                                self._redraw_input(buf, cursor, rule, status_line)
                        elif seq2 == "B":  # Down — history
                            if hist_idx < len(self._history):
                                hist_idx += 1
                                if hist_idx == len(self._history):
                                    buf = list(saved_for_history)
                                else:
                                    buf = list(self._history[hist_idx])
                                cursor = len(buf)
                                self._redraw_input(buf, cursor, rule, status_line)
                        elif seq2 == "H":  # Home
                            cursor = 0
                            self._redraw_input(buf, cursor, rule, status_line)
                        elif seq2 == "F":  # End
                            cursor = len(buf)
                            self._redraw_input(buf, cursor, rule, status_line)
                        elif seq2 == "3":  # Delete key (3~)
                            next_ch = sys.stdin.read(1)  # consume ~
                            if cursor < len(buf):
                                buf.pop(cursor)
                                self._redraw_input(buf, cursor, rule, status_line)
                    elif seq1 == "\r" or seq1 == "\n":
                        # Alt+Enter — add newline (just submit for now)
                        break
                    # else: ignore unknown escape

                elif ch == "\x15":
                    # Ctrl+U — clear line
                    buf.clear()
                    cursor = 0
                    self._redraw_input(buf, cursor, rule, status_line)

                elif ch == "\x0b":
                    # Ctrl+K — kill to end of line
                    buf = buf[:cursor]
                    self._redraw_input(buf, cursor, rule, status_line)

                elif ch == "\x01":
                    # Ctrl+A — home
                    cursor = 0
                    self._redraw_input(buf, cursor, rule, status_line)

                elif ch == "\x05":
                    # Ctrl+E — end
                    cursor = len(buf)
                    self._redraw_input(buf, cursor, rule, status_line)

                elif ch == "\x17":
                    # Ctrl+W — delete word back
                    while cursor > 0 and buf[cursor - 1] == " ":
                        buf.pop(cursor - 1)
                        cursor -= 1
                    while cursor > 0 and buf[cursor - 1] != " ":
                        buf.pop(cursor - 1)
                        cursor -= 1
                    self._redraw_input(buf, cursor, rule, status_line)

                elif ord(ch) >= 32:
                    # Printable char
                    buf.insert(cursor, ch)
                    cursor += 1
                    if cursor == len(buf):
                        # Fast path: appending at end
                        sys.stdout.write(ch)
                        sys.stdout.flush()
                    else:
                        self._redraw_input(buf, cursor, rule, status_line)

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

        # After submission: redraw final state with both rules
        # Move to input line start, clear to end of screen
        sys.stdout.write(f"\033[u")      # restore to input start
        sys.stdout.write(f"\033[J")      # clear to end of screen
        # Redraw input + bottom rule
        display = text if len(text) <= 200 else text[:200] + "..."
        sys.stdout.write(f"{display}\n")
        sys.stdout.write(f"\033[2m{rule}\033[0m\n")
        sys.stdout.flush()

        return text

    def _redraw_input(self, buf: list[str], cursor: int, rule: str, status: str) -> None:
        """Redraw the input line and rules below."""
        text = "".join(buf)
        # Move to saved position (start of input text)
        sys.stdout.write(f"\033[u")
        # Clear from cursor to end of screen
        sys.stdout.write(f"\033[J")
        # Write input text
        sys.stdout.write(text)
        # Bottom rule + status below
        sys.stdout.write(f"\n\033[2m{rule}\033[0m")
        if status:
            sys.stdout.write(f"\n\033[2m  {status}\033[0m")
        # Move cursor back to correct position in input
        sys.stdout.write(f"\033[u")
        if cursor > 0:
            sys.stdout.write(f"\033[{cursor}C")
        sys.stdout.flush()
