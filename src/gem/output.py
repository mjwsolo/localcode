"""Centralized output manager — ALL terminal output goes through here.

Codex-inspired design:
- ▪ bullet prefix on each section (tool groups, text responses, status)
- Bold action headers: "Ran", "Read", "Wrote", "Searched"
- └ tree connectors for tool results
- Clear spacing between sections
- "Working (Xs · esc to interrupt)" status line
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    IDLE = auto()
    THINKING = auto()
    TOOL_CALL = auto()
    STREAMING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class ToolAction:
    name: str
    args: str
    status: str = "running"  # running | done | error
    result: str = ""


@dataclass
class OutputState:
    phase: Phase = Phase.IDLE
    start_time: float = 0.0
    thinking_peek: str = ""
    tokens: int = 0
    tool_actions: list[ToolAction] = field(default_factory=list)
    content_chunks: list[str] = field(default_factory=list)
    error: str = ""


# Tool name → human-readable action header
TOOL_HEADERS = {
    "bash":         "Ran",
    "read_file":    "Read",
    "write_file":   "Wrote",
    "edit_file":    "Edited",
    "grep":         "Searched",
    "glob":         "Found",
    "list_files":   "Listed",
    "web_search":   "Searched web",
    "web_fetch":    "Fetched",
}


def _cols() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


class OutputManager:
    """Single output controller for the entire app."""

    def __init__(self) -> None:
        self.state = OutputState()
        self._lock = threading.Lock()
        self._indicator_thread: threading.Thread | None = None
        self._indicator_running = False
        self._event_callback = None
        self._approval_callback = None  # TUI: blocks worker thread until approved/denied
        self._stream_started = False
        self._custom_stage = ""
        self._status_line = ""
        self._input_field = None  # set by app.py to enable typeahead

    def set_event_callback(self, callback) -> None:
        self._event_callback = callback

    def set_approval_callback(self, callback) -> None:
        """Set callback for TUI approval: callback(tool_name, command) -> bool."""
        self._approval_callback = callback

    def _emit_event(self, event_type: str, **payload) -> None:
        if self._event_callback is not None:
            try:
                self._event_callback(event_type, payload)
            except Exception:
                pass

    # ── Phase transitions ────────────────────────────────────────────

    def start_thinking(self, reset: bool = True) -> None:
        """Begin thinking phase — shows working indicator."""
        with self._lock:
            self.state.phase = Phase.THINKING
            self._custom_stage = ""
            if reset:
                self.state.start_time = time.time()
                self.state.tokens = 0
                self.state.thinking_peek = ""
                self.state.tool_actions.clear()
                self.state.content_chunks.clear()
                self._stream_started = False
            else:
                self.state.thinking_peek = ""
        self._emit_event("thinking_start", reset=str(reset).lower())
        self._start_indicator()

    def start_streaming(self) -> None:
        """Transition to content streaming — stops indicator."""
        self._stop_indicator()
        with self._lock:
            self.state.phase = Phase.STREAMING
            self._custom_stage = ""
        self._emit_event("stream_start")
        # Clear indicator and add breathing room before response
        sys.stdout.write("\r\033[K\n\n  ")
        sys.stdout.flush()
        self._stream_started = False

    def done(self) -> None:
        """Finished — cleanup."""
        self._stop_indicator()
        elapsed = time.time() - self.state.start_time
        tools_used = len(self.state.tool_actions)
        with self._lock:
            self.state.phase = Phase.DONE
            self._custom_stage = ""
        self._emit_event("done")
        # Summary line + breathing room (guard against stale start_time)
        parts = [f"{elapsed:.1f}s"]
        if tools_used:
            tool_names = {}
            for t in self.state.tool_actions:
                tool_names[t.name] = tool_names.get(t.name, 0) + 1
            tool_summary = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in tool_names.items())
            parts.append(f"tools: {tool_summary}")
        sys.stdout.write(f"\n\033[2m  Done in {' — '.join(parts)}\033[0m\n")
        sys.stdout.flush()

    def set_error(self, msg: str) -> None:
        self._stop_indicator()
        with self._lock:
            self.state.phase = Phase.ERROR
            self.state.error = msg
            self._custom_stage = ""
        self._emit_event("error", message=msg[:240])
        sys.stdout.write(f"\n\033[31m▪ Error: {msg}\033[0m\n\n")
        sys.stdout.flush()

    # ── Thinking ─────────────────────────────────────────────────────

    def feed_thinking(self, chunk: str) -> None:
        with self._lock:
            self.state.tokens += max(1, len(chunk) // 4)
        self._emit_event("thinking_chunk", chunk=chunk[:2000])

    def thinking_done(self, full_text: str) -> None:
        """Emit the complete thinking text for display in TUI."""
        self._last_thinking_text = full_text  # Store for direct access
        self._emit_event("thinking_done", text=full_text[:8000])

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._custom_stage = stage
        self._emit_event("stage", stage=stage[:120])

    def set_thinking_peek(self, text: str) -> None:
        with self._lock:
            self.state.thinking_peek = text[:120]
        if text.strip():
            self._emit_event("thinking_peek", text=text[:120])

    # ── Tool calls ───────────────────────────────────────────────────

    def log_tool(self, name: str, args: str = "") -> int:
        """Log a tool call with Codex-style formatting."""
        self._stop_indicator()
        idx = len(self.state.tool_actions)
        self.state.tool_actions.append(ToolAction(name=name, args=args))
        self._emit_event("tool_start", name=name, args=args[:200], index=str(idx))

        header = TOOL_HEADERS.get(name, name)
        # Bold header + cyan args
        if name == "bash":
            sys.stdout.write(f"\n\033[1m▪ {header}\033[0m \033[36m{args[:80]}\033[0m\n")
        elif name in ("read_file", "write_file", "edit_file"):
            sys.stdout.write(f"\n\033[1m▪ {header}\033[0m \033[36m{args[:80]}\033[0m\n")
        elif name in ("grep", "glob"):
            sys.stdout.write(f"\n\033[1m▪ {header}\033[0m \033[36m{args[:80]}\033[0m\n")
        else:
            sys.stdout.write(f"\n\033[1m▪ {header}\033[0m \033[36m{args[:80]}\033[0m\n")
        sys.stdout.flush()
        self._start_indicator()
        return idx

    @staticmethod
    def _filter_noise(text: str) -> str:
        lines = text.splitlines()
        filtered = [l for l in lines if "MallocStackLogging" not in l and "can't turn off" not in l]
        return "\n".join(filtered)

    def tool_result(self, result: str, error: bool = False, idx: int = -1) -> None:
        """Show tool result with tree-style indentation."""
        result = self._filter_noise(result)
        if idx >= 0 and idx < len(self.state.tool_actions):
            self.state.tool_actions[idx].status = "error" if error else "done"
            self.state.tool_actions[idx].result = result
        with self._lock:
            self._custom_stage = ""
        self._emit_event(
            "tool_result",
            error=str(error).lower(),
            index=str(idx),
            result=result[:4000],
        )
        self._stop_indicator()
        lines = result.strip().splitlines()
        if error:
            for line in lines[:5]:
                sys.stdout.write(f"\033[31m  └ {line[:90]}\033[0m\n")
        elif lines:
            max_lines = 8
            for i, line in enumerate(lines[:max_lines]):
                sys.stdout.write(f"\033[2m  └ {line[:90]}\033[0m\n")
            if len(lines) > max_lines:
                sys.stdout.write(f"\033[2m    … +{len(lines) - max_lines} lines\033[0m\n")
        sys.stdout.flush()
        self._start_indicator()

    # ── Content streaming ────────────────────────────────────────────

    _col_pos = 0  # track column position for soft wrapping

    def stream(self, chunk: str) -> None:
        """Stream content to terminal with soft wrapping."""
        if self.state.phase != Phase.STREAMING:
            self.start_streaming()
            self._col_pos = 2  # we start with 2-char indent
        with self._lock:
            self.state.content_chunks.append(chunk)
            self.state.tokens += max(1, len(chunk) // 4)
        self._emit_event("content", chunk=chunk[:2000], chars=str(len(chunk)))
        cols = _cols()
        wrap_at = cols - 1  # leave 1 char margin
        out = []
        for ch in chunk:
            if ch == "\n":
                out.append("\n  ")
                self._col_pos = 2
            else:
                if self._col_pos >= wrap_at:
                    out.append("\n  ")
                    self._col_pos = 2
                out.append(ch)
                self._col_pos += 1
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def print_info(self, text: str) -> None:
        """Print dim info text."""
        was_running = self._indicator_running
        if was_running:
            self._stop_indicator()
        sys.stdout.write(f"\033[2m  {text}\033[0m\n")
        sys.stdout.flush()
        if was_running:
            self._start_indicator()

    # ── Indicator (background thread) ────────────────────────────────

    def _start_indicator(self) -> None:
        if self._indicator_running:
            return
        self._indicator_running = True
        self._indicator_thread = threading.Thread(target=self._run_indicator, daemon=True)
        try:
            self._indicator_thread.start()
        except RuntimeError:
            self._indicator_thread = threading.Thread(target=self._run_indicator, daemon=True)
            self._indicator_thread.start()

    def _stop_indicator(self) -> None:
        if not self._indicator_running:
            return
        self._indicator_running = False
        if self._indicator_thread:
            try:
                self._indicator_thread.join(timeout=1)
            except (KeyboardInterrupt, RuntimeError):
                pass
        try:
            sys.stdout.write("\033[?25h")  # show cursor
            sys.stdout.write("\r\033[J")   # clear from cursor to end of screen
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass

    # Gem-themed labels and icons
    _LABELS = [
        "mining", "cutting facets", "polishing",
        "examining", "shaping", "refining",
        "digging deeper", "crystallizing", "forging",
    ]
    _ICONS = ["·", "▲", "◆"]

    def _run_indicator(self) -> None:
        tick = 0
        while self._indicator_running:
            tick += 1
            icon = self._ICONS[tick % len(self._ICONS)]
            elapsed = time.time() - self.state.start_time
            if elapsed < 60:
                timer = f"{elapsed:.0f}s"
            else:
                timer = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

            with self._lock:
                custom = getattr(self, '_custom_stage', '')

            label = custom or self._LABELS[int(elapsed) // 6 % len(self._LABELS)]
            hint = " · ctrl+c to stop" if elapsed > 10 else ""
            line = f"\033[32m {icon} {label}... ({timer})\033[0m{hint}"

            try:
                # Collect typeahead keystrokes (non-blocking)
                input_field = self._input_field
                if input_field:
                    input_field.collect_typeahead()
                    typeahead = input_field.get_typeahead_text()
                    queued_count = len(input_field._queue)
                else:
                    typeahead = ""
                    queued_count = 0

                # Draw spinner + input field below it
                cols = _cols()
                rule = "  " + ("─" * max(24, min(96, cols - 4)))
                status = getattr(self, '_status_line', '')
                sys.stdout.write(f"\r{line}\033[K")
                # Input field below spinner
                sys.stdout.write(f"\n\033[2m{rule}\033[0m\033[K")
                # Show typeahead text or queued indicator
                if queued_count:
                    sys.stdout.write(f"\n  › \033[2m({queued_count} queued)\033[0m\033[K")
                elif typeahead:
                    sys.stdout.write(f"\n  › {typeahead}\033[K")
                else:
                    sys.stdout.write(f"\n\033[2m  › \033[0m\033[K")
                sys.stdout.write(f"\n\033[2m{rule}\033[0m\033[K")
                if status:
                    sys.stdout.write(f"\n\033[2m  {status}\033[0m\033[K")
                sys.stdout.write(f"\033[?25l")  # hide cursor
                lines_back = 4 if status else 3
                sys.stdout.write(f"\033[{lines_back}A\r")
                sys.stdout.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._indicator_running = False
                break
            time.sleep(0.5)
