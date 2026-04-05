"""Centralized output manager — ALL terminal output goes through here.

Inspired by OpenCode's DisplayState pattern:
- Single source of truth for what's on screen
- Explicit phase transitions (idle → thinking → tool_call → streaming → done)
- No race conditions — one thread owns the display
- Tool results appear inline under their tool call
- Thinking indicator updates in-place

This replaces the scattered stdout/stderr/Rich console calls throughout the app.
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


# Gem-themed labels
LABELS = [
    "mining", "cutting facets", "polishing",
    "examining", "shaping", "refining",
    "digging deeper", "crystallizing", "forging",
]
ICONS = ["·", "▲", "◆"]


def _cols() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


class OutputManager:
    """Single output controller for the entire app.

    Usage:
        out = OutputManager()
        out.start_thinking()          # shows pulsating indicator
        out.set_thinking_peek("analyzing code...")  # shows under indicator
        out.log_tool("read_file", "path=hello.py")  # shows ● read_file
        out.tool_result("hello.py (10 lines)")       # shows ⎿ result
        out.start_streaming()         # stops indicator
        out.stream("Hello world")     # prints content
        out.done()                    # cleanup
    """

    def __init__(self) -> None:
        self.state = OutputState()
        self._lock = threading.Lock()
        self._indicator_thread: threading.Thread | None = None
        self._indicator_running = False

    # ── Phase transitions ────────────────────────────────────────────

    def start_thinking(self) -> None:
        """Begin thinking phase — shows pulsating indicator."""
        with self._lock:
            self.state.phase = Phase.THINKING
            self.state.start_time = time.time()
            self.state.tokens = 0
            self.state.thinking_peek = ""
            self.state.tool_actions.clear()
            self.state.content_chunks.clear()
        self._start_indicator()

    def start_streaming(self) -> None:
        """Transition to content streaming — stops indicator."""
        self._stop_indicator()
        with self._lock:
            self.state.phase = Phase.STREAMING

    def done(self) -> None:
        """Finished — cleanup."""
        self._stop_indicator()
        with self._lock:
            self.state.phase = Phase.DONE
        sys.stdout.write("\n")
        sys.stdout.flush()

    def set_error(self, msg: str) -> None:
        self._stop_indicator()
        with self._lock:
            self.state.phase = Phase.ERROR
            self.state.error = msg
        sys.stdout.write(f"\033[31m  error: {msg}\033[0m\n")
        sys.stdout.flush()

    # ── Thinking ─────────────────────────────────────────────────────

    def feed_thinking(self, chunk: str) -> None:
        """Feed thinking tokens — updates peek text and token count."""
        with self._lock:
            self.state.tokens += max(1, len(chunk) // 4)
            text = chunk.replace('\n', ' ').strip()
            if text and len(text) > 3:
                self.state.thinking_peek = text

    def set_stage(self, stage: str) -> None:
        """Override the indicator label (e.g. 'editing hello.py')."""
        with self._lock:
            self._custom_stage = stage

    # ── Tool calls ───────────────────────────────────────────────────

    def log_tool(self, name: str, args: str = "") -> int:
        """Log a tool call. Returns index for updating result."""
        self._stop_indicator()  # pause indicator to print cleanly
        idx = len(self.state.tool_actions)
        self.state.tool_actions.append(ToolAction(name=name, args=args))
        sys.stdout.write(f"\033[32m  ● {name}\033[0m \033[2m{args[:60]}\033[0m\n")
        sys.stdout.flush()
        self._start_indicator()  # resume indicator
        return idx

    def tool_result(self, result: str, error: bool = False, idx: int = -1) -> None:
        """Show tool result inline."""
        if idx >= 0 and idx < len(self.state.tool_actions):
            self.state.tool_actions[idx].status = "error" if error else "done"
            self.state.tool_actions[idx].result = result
        self._stop_indicator()
        if error:
            sys.stdout.write(f"\033[31m    ⎿ {result[:80]}\033[0m\n")
        else:
            # Format nicely based on content
            lines = result.strip().splitlines()
            if "Added" in result and "removed" in result:
                sys.stdout.write(f"\033[2m    ⎿ {result[:80]}\033[0m\n")
            elif lines:
                sys.stdout.write(f"\033[2m    ⎿ {lines[0][:80]}\033[0m\n")
        sys.stdout.flush()
        self._start_indicator()

    # ── Content streaming ────────────────────────────────────────────

    def stream(self, chunk: str) -> None:
        """Stream content to terminal."""
        if self.state.phase != Phase.STREAMING:
            self.start_streaming()
        with self._lock:
            self.state.content_chunks.append(chunk)
            self.state.tokens += max(1, len(chunk) // 4)
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def print_info(self, text: str) -> None:
        """Print dim info text."""
        sys.stdout.write(f"\033[2m  {text}\033[0m\n")
        sys.stdout.flush()

    # ── Indicator (background thread) ────────────────────────────────

    def _start_indicator(self) -> None:
        if self._indicator_running:
            return
        self._indicator_running = True
        self._indicator_thread = threading.Thread(target=self._run_indicator, daemon=True)
        try:
            self._indicator_thread.start()
        except RuntimeError:
            # Thread already started/dead — create new one
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
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            pass

    @staticmethod
    def _flip_text(text: str, tick: int) -> str:
        """Subtle cursor — a dot moves after the text to show activity."""
        # No per-character animation (causes flicker). Just rotate the icon.
        return text

    def _run_indicator(self) -> None:
        tick = 0
        cols = _cols()
        while self._indicator_running:
            tick += 1
            icon = ICONS[tick % len(ICONS)]

            elapsed = time.time() - self.state.start_time
            if elapsed < 60:
                timer = f"{elapsed:.0f}s"
            else:
                timer = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

            with self._lock:
                tokens = self.state.tokens
                peek = self.state.thinking_peek
                custom = getattr(self, '_custom_stage', '')

            label = custom or LABELS[int(elapsed) // 6 % len(LABELS)]

            # Cost savings: what this would cost on cloud APIs
            cost = tokens * 0.000015  # ~$15/1M tokens (Claude/GPT-4 average)
            savings = f" · saved ${cost:.2f}" if cost > 0.01 else ""

            flip_label = self._flip_text(f"{label}...", tick)
            line1 = f"\033[32m {icon} {flip_label} ({timer}{savings})\033[0m"

            raw_len = len(f" {icon} {label}... ({timer})")
            if raw_len > cols - 2:
                line1 = line1[:cols - 5] + "..."

            # Line 2: thinking peek (what model is considering)
            if peek:
                peek_text = peek[:cols - 6]
                sys.stderr.write(f"\r{line1}\033[K\n\033[2m    {peek_text}\033[0m\033[K\033[A")
            else:
                sys.stderr.write(f"\r{line1}\033[K")
            sys.stderr.flush()
            time.sleep(0.12)
