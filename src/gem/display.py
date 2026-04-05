"""Jem display — clean terminal output.

Thinking: collapsed by default (like Codex), just shows pulsating indicator.
Tool calls: clean one-line summaries.
Responses: streamed with syntax highlighting on code blocks.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Any


def _cols() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


class ThinkingIndicator:
    """Pulsating indicator — shows while model is working.

    Like Codex: just icon + label + timer on one line.
    Thinking text is NOT shown by default (collapsed).
    """

    ICONS = ["·", "▲", "◆"]  # dot → triangle → diamond cycle
    LABELS = [
        "mining", "cutting facets", "polishing",
        "examining", "shaping", "refining",
        "digging deeper", "crystallizing", "forging",
    ]

    def set_stage(self, stage: str) -> None:
        """Override the label with a specific stage description."""
        self._stage = stage

    @property
    def _current_label(self) -> str:
        if hasattr(self, '_stage') and self._stage:
            return self._stage
        return self.LABELS[int(time.time() - self._start_time) // 6 % len(self.LABELS)]

    def __init__(self) -> None:
        self._running = False
        self._start_time = 0.0
        self._tokens = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return  # already running, don't create another thread
        self._running = True
        if self._start_time == 0:
            self._start_time = time.time()  # keep original start time on restart
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, chunk: str) -> None:
        """Count tokens and show thinking peek under indicator."""
        with self._lock:
            self._tokens += max(1, len(chunk) // 4)
            # Show a one-line peek of what the model is thinking
            text = chunk.replace('\n', ' ').strip()
            if text and len(text) > 3:
                self._peek = text

    def log_action(self, action: str) -> None:
        """Log an action below the indicator (tool call, file read, etc)."""
        # Clear indicator line, print action, re-print indicator
        sys.stderr.write(f"\r\033[K")
        sys.stdout.write(f"\033[32m  ● {action}\033[0m\n")
        sys.stdout.flush()

    def add_tokens(self, n: int) -> None:
        """Add token count directly."""
        with self._lock:
            self._tokens += n

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Clear indicator line and peek line
        sys.stderr.write("\r\033[K\n\033[K\033[A")
        sys.stderr.flush()
        self._peek = ""

    def _run(self) -> None:
        tick = 0
        while self._running:
            tick += 1
            icon = self.ICONS[tick % len(self.ICONS)]
            label = self._current_label

            elapsed = time.time() - self._start_time
            if elapsed < 60:
                timer = f"{elapsed:.0f}s"
            else:
                timer = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

            with self._lock:
                tokens = self._tokens
            cost = tokens * 0.000015
            savings = f" · ${cost:.2f}" if cost > 0.001 else ""

            # Keep line SHORT — must fit in terminal width
            cols = 60
            try:
                cols = os.get_terminal_size().columns
            except Exception:
                pass

            # Line 1: indicator
            text = f" {icon} {label}... ({timer}{savings})"
            if len(text) > cols - 2:
                text = text[:cols - 5] + "..."

            # Line 2: thinking peek (what model is considering)
            peek = getattr(self, '_peek', '')
            if peek:
                peek_display = peek[:cols - 6]
                sys.stderr.write(f"\r\033[32m{text}\033[0m\033[K\n\033[2m    {peek_display}\033[0m\033[K\033[A")
            else:
                sys.stderr.write(f"\r\033[32m{text}\033[0m\033[K")
            sys.stderr.flush()
            time.sleep(0.15)


class ToolCallDisplay:
    """Display tool calls and results — one line each."""

    ICONS = {
        "read_file": "📄", "write_file": "📝", "edit_file": "✏️",
        "grep": "🔍", "glob": "📁", "bash": "💻",
        "web_search": "🌐", "web_fetch": "🌐", "current_datetime": "🕐",
        "git_status": "📊", "git_diff": "📊", "git_log": "📊",
        "git_commit": "📊", "multi_edit": "✏️",
        "search_code": "🔍", "list_files": "📁",
    }

    @staticmethod
    def show_call(name: str, args_preview: str) -> None:
        icon = ToolCallDisplay.ICONS.get(name, "⚙️")
        args_short = args_preview[:50]
        sys.stdout.write(f"\033[32m  ● {name}\033[0m \033[2m{args_short}\033[0m\n")
        sys.stdout.flush()

    @staticmethod
    def show_result(content: str, is_error: bool = False) -> None:
        lines = content.strip().splitlines()
        if is_error:
            preview = lines[0][:80] if lines else "error"
            sys.stdout.write(f"\033[31m    ⎿ {preview}\033[0m\n")
        else:
            if "---" in content and "+++" in content:
                # Count additions/removals for Codex-style summary
                added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
                removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
                sys.stdout.write(f"\033[2m    ⎿ Added {added} lines, removed {removed} lines\033[0m\n")
            elif lines:
                preview = lines[0][:80]
                sys.stdout.write(f"\033[2m    ⎿ {preview}\033[0m\n")
        sys.stdout.flush()


class ToolProgressCallback:
    @staticmethod
    def update(tool_name: str, progress: str) -> None:
        sys.stderr.write(f"\r\033[2m    {tool_name}: {progress[:60]}\033[0m\033[K")
        sys.stderr.flush()

    @staticmethod
    def clear() -> None:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


class DiffPreview:
    @staticmethod
    def show(diff_text: str) -> None:
        for line in diff_text.splitlines()[:20]:
            if line.startswith("+++") or line.startswith("---"):
                sys.stdout.write(f"\033[1m{line}\033[0m\n")
            elif line.startswith("+"):
                sys.stdout.write(f"\033[32m{line}\033[0m\n")
            elif line.startswith("-"):
                sys.stdout.write(f"\033[31m{line}\033[0m\n")
            elif line.startswith("@@"):
                sys.stdout.write(f"\033[36m{line}\033[0m\n")
            else:
                sys.stdout.write(f"{line}\n")
        sys.stdout.flush()


class ContextBudgetDisplay:
    @staticmethod
    def show(used_chars: int, max_chars: int) -> None:
        bar_width = 30
        pct = min(1.0, used_chars / max(1, max_chars))
        filled = int(pct * bar_width)
        empty = bar_width - filled
        if pct < 0.6:
            color = "\033[32m"
        elif pct < 0.8:
            color = "\033[33m"
        else:
            color = "\033[31m"
        bar = f"{color}{'█' * filled}\033[2m{'░' * empty}\033[0m"
        sys.stderr.write(f"\r  context: [{bar}] {pct:.0%}\033[K\n")
        sys.stderr.flush()


class ResponseDisplay:
    """Stream model response. Code blocks get syntax highlighting."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._in_code_block = False
        self._code_buffer: list[str] = []

    def print_chunk(self, text: str) -> None:
        self._chunks.append(text)
        if "```" in text:
            parts = text.split("```")
            for i, part in enumerate(parts):
                if i > 0:
                    self._in_code_block = not self._in_code_block
                    if self._in_code_block:
                        sys.stdout.write("\n")
                        self._code_buffer = [part]
                        continue
                    else:
                        self._code_buffer.append(part)
                        self._render_code_block()
                        continue
                if self._in_code_block:
                    self._code_buffer.append(part)
                else:
                    sys.stdout.write(part)
                    sys.stdout.flush()
        elif self._in_code_block:
            self._code_buffer.append(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _render_code_block(self) -> None:
        code = "".join(self._code_buffer).strip()
        self._code_buffer.clear()
        try:
            from rich.console import Console
            from rich.syntax import Syntax
            lines = code.split("\n")
            lang = lines[0].strip() if lines else ""
            known_langs = {"python", "py", "javascript", "js", "typescript", "ts",
                          "bash", "sh", "json", "yaml", "html", "css", "rust", "go"}
            if lang in known_langs:
                code_body = "\n".join(lines[1:])
            else:
                lang = "text"
                code_body = code
            Console(highlight=False).print(Syntax(code_body, lang, theme="monokai", padding=1))
            return
        except Exception:
            pass
        sys.stdout.write(f"\n{code}\n")
        sys.stdout.flush()

    def end(self) -> None:
        if self._code_buffer:
            sys.stdout.write("".join(self._code_buffer))
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        self._chunks.clear()
        self._code_buffer.clear()
        self._in_code_block = False

    @staticmethod
    def print_info(text: str) -> None:
        sys.stdout.write(f"\033[2m  {text}\033[0m\n")
        sys.stdout.flush()


class SessionStats:
    CLOUD_COST_PER_1M_INPUT = 3.00
    CLOUD_COST_PER_1M_OUTPUT = 15.00

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    def record(self, response_data: dict) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += response_data.get("prompt_eval_count", 0)
        self.total_completion_tokens += response_data.get("eval_count", 0)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def cloud_cost_saved(self) -> float:
        input_cost = (self.total_prompt_tokens / 1_000_000) * self.CLOUD_COST_PER_1M_INPUT
        output_cost = (self.total_completion_tokens / 1_000_000) * self.CLOUD_COST_PER_1M_OUTPUT
        return input_cost + output_cost

    def summary(self) -> str:
        saved = self.cloud_cost_saved
        return (
            f"{self.total_tokens:,} tokens · "
            f"{self.total_requests} requests · "
            f"${saved:.4f} saved vs cloud"
        )
