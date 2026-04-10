"""Streaming output parser — parse LLM output as tokens arrive.

Key capabilities:
1. Route tokens to UI (code vs text vs diff) in real-time
2. Detect when output is complete (all blocks closed) and stop early
3. Detect repetition loops and terminate
4. Parse tool calls from partial output
5. Track token count and timing

This replaces waiting for full response before parsing.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class TokenType(Enum):
    """What kind of content is this token part of?"""
    TEXT = auto()          # Natural language
    CODE = auto()          # Inside a code block
    DIFF = auto()          # Inside a search/replace block
    TOOL_JSON = auto()     # Inside a JSON tool call
    THINKING = auto()      # Model reasoning (dim in UI)


@dataclass
class StreamState:
    """Tracks the state of the streaming parser."""
    buffer: list[str] = field(default_factory=list)
    token_count: int = 0
    start_time: float = 0.0
    current_type: TokenType = TokenType.TEXT

    # Code block tracking
    in_code_block: bool = False
    code_fence_count: int = 0

    # Search/replace tracking
    search_opens: int = 0
    search_closes: int = 0

    # Tool call tracking
    brace_depth: int = 0
    in_tool_json: bool = False

    # Early stop
    stopped: bool = False
    stop_reason: str = ""

    # Repetition detection
    last_100_chars: str = ""

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    @property
    def tokens_per_sec(self) -> float:
        elapsed = self.elapsed_ms / 1000
        return self.token_count / elapsed if elapsed > 0 else 0

    @property
    def full_text(self) -> str:
        return "".join(self.buffer)


@dataclass
class StreamChunk:
    """A parsed chunk ready for display."""
    text: str
    token_type: TokenType
    is_complete: bool = False
    stop_reason: str = ""


class StreamingParser:
    """Parse streaming LLM output token-by-token.

    Usage:
        parser = StreamingParser()
        for token in llm.stream(...):
            chunk = parser.feed(token)
            if chunk.is_complete:
                break
            ui.display(chunk)
        result = parser.get_result()
    """

    def __init__(self, mode: str = "auto") -> None:
        """
        Args:
            mode: "code" — expect code output, detect code block completion
                  "edit" — expect search/replace blocks, detect block closure
                  "tool" — expect JSON tool calls
                  "auto" — detect from content
        """
        self.mode = mode
        self.state = StreamState(start_time=time.time())
        self._stop_checks: list[Callable[[str], str | None]] = []

        # Register stop conditions based on mode
        self._stop_checks.append(self._check_repetition)

        if mode == "edit":
            self._stop_checks.append(self._check_edit_complete)
        elif mode == "code":
            self._stop_checks.append(self._check_code_complete)
        elif mode == "tool":
            self._stop_checks.append(self._check_tool_complete)
        else:
            # Auto mode — check everything
            self._stop_checks.append(self._check_edit_complete)
            self._stop_checks.append(self._check_code_complete)
            self._stop_checks.append(self._check_tool_complete)

    def feed(self, token: str) -> StreamChunk:
        """Feed a token and get back a typed chunk for display."""
        self.state.buffer.append(token)
        self.state.token_count += 1

        # Track accumulated text for stop conditions
        full = self.state.full_text

        # Detect token type
        token_type = self._classify_token(token, full)
        self.state.current_type = token_type

        # Check stop conditions
        for check in self._stop_checks:
            reason = check(full)
            if reason:
                self.state.stopped = True
                self.state.stop_reason = reason
                return StreamChunk(
                    text=token,
                    token_type=token_type,
                    is_complete=True,
                    stop_reason=reason,
                )

        return StreamChunk(text=token, token_type=token_type)

    def get_result(self) -> dict:
        """Get the final parsed result after streaming completes."""
        full = self.state.full_text
        return {
            "raw": full,
            "token_count": self.state.token_count,
            "elapsed_ms": self.state.elapsed_ms,
            "tokens_per_sec": self.state.tokens_per_sec,
            "early_stopped": self.state.stopped,
            "stop_reason": self.state.stop_reason,
        }

    # ── Token classification ────────────────────────────────────────

    def _classify_token(self, token: str, full: str) -> TokenType:
        """Determine what type of content this token belongs to."""

        # Track code fences
        if "```" in token:
            self.state.code_fence_count += token.count("```")
            self.state.in_code_block = (self.state.code_fence_count % 2 == 1)

        # Track search/replace blocks
        if "<<<SEARCH" in token:
            self.state.search_opens += 1
        if "SEARCH>>>" in token:
            self.state.search_closes += 1

        # Track JSON braces
        for ch in token:
            if ch == "{":
                self.state.brace_depth += 1
                if self.state.brace_depth == 1 and '"tool"' in full[-200:]:
                    self.state.in_tool_json = True
            elif ch == "}":
                self.state.brace_depth = max(0, self.state.brace_depth - 1)
                if self.state.brace_depth == 0:
                    self.state.in_tool_json = False

        # Return type
        if self.state.in_tool_json:
            return TokenType.TOOL_JSON
        if self.state.search_opens > self.state.search_closes:
            return TokenType.DIFF
        if self.state.in_code_block:
            return TokenType.CODE

        # Detect thinking/reasoning
        if any(token.strip().startswith(w) for w in
               ("I ", "Let me", "First", "Now", "The ", "This ")):
            if not self.state.in_code_block:
                return TokenType.THINKING

        return TokenType.TEXT

    # ── Stop conditions ─────────────────────────────────────────────

    def _check_repetition(self, text: str) -> str | None:
        """Detect model stuck in a loop."""
        if len(text) < 200:
            return None
        last_100 = text[-100:]
        prev_100 = text[-200:-100]
        if last_100 == prev_100:
            return "repetition_detected"

        # Also check for shorter repeat patterns (e.g., same line repeated)
        lines = text.splitlines()
        if len(lines) >= 6:
            last_3 = lines[-3:]
            prev_3 = lines[-6:-3]
            if last_3 == prev_3 and all(l.strip() for l in last_3):
                return "line_repetition"

        return None

    def _check_edit_complete(self, text: str) -> str | None:
        """Check if all search/replace blocks are closed."""
        opens = text.count("<<<SEARCH")
        closes = text.count("SEARCH>>>")
        if opens > 0 and closes >= opens:
            # All blocks closed — check for trailing garbage
            after_last = text.split("SEARCH>>>")[-1].strip()
            if len(after_last) > 80:
                return "all_edit_blocks_closed"
        return None

    def _check_code_complete(self, text: str) -> str | None:
        """Check if code output is followed by unwanted explanation."""
        # Count fences
        fence_count = text.count("```")
        if fence_count >= 2 and fence_count % 2 == 0:
            # All code blocks closed
            after_last_fence = text.split("```")[-1].strip()
            if len(after_last_fence) > 100:
                # Model is adding explanation after code — stop
                return "code_block_closed_with_trailing"
        return None

    def _check_tool_complete(self, text: str) -> str | None:
        """Check if a JSON tool call is complete."""
        if '"tool"' not in text:
            return None

        # Find the last complete JSON object with "tool" in it
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    if '"tool"' in candidate:
                        # JSON object complete
                        after = text[i + 1:].strip()
                        if len(after) > 50:
                            return "tool_json_complete"
                    start = -1

        return None

    # ── TOOL: write_file detection ──────────────────────────────────

    @staticmethod
    def _check_write_file_complete(text: str) -> str | None:
        """Check if a TOOL: write_file block is complete."""
        if "TOOL: write_file" not in text:
            return None
        # Check if the code block after it is closed
        match = re.search(r'TOOL:\s*write_file.*?```\w*\n.*?```', text, re.DOTALL)
        if match:
            after = text[match.end():].strip()
            if len(after) > 50:
                return "write_file_complete"
        return None


class StreamRouter:
    """Routes streaming tokens to the right UI handler based on type.

    Integrates StreamingParser with OutputManager.
    """

    def __init__(self, parser: StreamingParser, callbacks: dict[TokenType, Callable]) -> None:
        self.parser = parser
        self.callbacks = callbacks

    def feed(self, token: str) -> bool:
        """Feed a token, route to UI. Returns False if streaming should stop."""
        chunk = self.parser.feed(token)

        # Route to the right callback
        cb = self.callbacks.get(chunk.token_type)
        if cb:
            cb(chunk.text)

        return not chunk.is_complete


def create_default_router(parser: StreamingParser, out: Any) -> StreamRouter:
    """Create a StreamRouter with standard UI callbacks.

    Args:
        parser: The streaming parser
        out: OutputManager instance (from output.py)
    """
    import sys

    def show_code(text: str) -> None:
        sys.stdout.write(f"\033[37m{text}\033[0m")
        sys.stdout.flush()

    def show_text(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def show_diff(text: str) -> None:
        # Color: green for additions, red for removals
        if text.startswith("+"):
            sys.stdout.write(f"\033[32m{text}\033[0m")
        elif text.startswith("-"):
            sys.stdout.write(f"\033[31m{text}\033[0m")
        else:
            sys.stdout.write(f"\033[2m{text}\033[0m")
        sys.stdout.flush()

    def show_thinking(text: str) -> None:
        sys.stdout.write(f"\033[2m{text}\033[0m")
        sys.stdout.flush()

    def show_tool(text: str) -> None:
        sys.stdout.write(f"\033[33m{text}\033[0m")
        sys.stdout.flush()

    return StreamRouter(parser, {
        TokenType.CODE: show_code,
        TokenType.TEXT: show_text,
        TokenType.DIFF: show_diff,
        TokenType.THINKING: show_thinking,
        TokenType.TOOL_JSON: show_tool,
    })
