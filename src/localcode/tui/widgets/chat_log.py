"""Scrollable chat message log — terminal coding tools style.

Messages displayed with clear visual hierarchy:
- User: ❯ bold prompt
- Assistant: indented markdown with syntax highlighting
- Tools: ● header with ⎿ tree connectors for results
- Info/errors: dim/red with consistent indentation
"""
from __future__ import annotations

import time

from ...theme import C



from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich.theme import Theme
from textual.strip import Strip
from textual.widgets import RichLog

# Custom theme: subtle, engineer-friendly
_CUSTOM_THEME = Theme({
    # Code blocks use bold (terminal-default fg) on a subtle inset
    # background — no pink, keeping the "only brand carries colour"
    # discipline. Inset bg is intentionally a fixed hex (not
    # ansi_default) because a slight contrast tint is what makes
    # inline `code` actually look different from prose.
    "markdown.code": "bold on #2a2a2a",
    "markdown.code_inline": "bold on #2a2a2a",
    "markdown.h1": "bold white",
    "markdown.h2": "bold white",
    "markdown.h3": "bold white",
    "markdown.h4": "bold white",
})


# Tool display names (action verb for headers)
_TOOL_HEADERS = {
    "bash": ("Bash", "ran"),
    "read_file": ("Read", "read"),
    "write_file": ("Write", "wrote"),
    "append_file": ("Append", "appended"),
    "edit_file": ("Update", "updated"),
    "grep": ("Search", "searched"),
    "glob": ("Glob", "found"),
    "web_search": ("WebSearch", "searched web"),
    "web_fetch": ("WebFetch", "fetched"),
    "multi_edit": ("MultiEdit", "edited"),
}


def _looks_like_markdown(text: str) -> bool:
    """Trigger the end-of-stream markdown rewrite for responses that
    would look bad as raw text.

    History: this used to trigger on a lone backtick, which caused
    flicker on prose with inline code references. It was then narrowed
    to ONLY fenced code blocks, which swung too far — `**bold**`,
    `- bullets`, and `### headers` all started rendering as literal
    source characters in regular responses. Current rule: fire on structural markdown signals
    (fences, bold, bullets, numbered lists, headers) but NOT on lone
    backticks, so inline `foo` in prose doesn't trigger the rewrite.
    """
    if not text:
        return False
    # Closed fenced block — high-value rewrite (syntax highlighting).
    if "```" in text:
        c = text.count("```")
        if c >= 2 and c % 2 == 0:
            return True
    # Bold **like this** — must be paired, not just an accidental **.
    if text.count("**") >= 2:
        return True
    # Line-leading markers for lists and headers. Checking at line
    # start avoids false positives like "5 - 3 = 2" inside prose.
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith(("- ", "* ", "### ", "## ", "# ", "> ")):
            return True
        if len(s) >= 3 and s[0].isdigit() and s[1:3] == ". ":
            return True
    return False


class ChatLog(RichLog):
    """Scrollable message area styled like terminal coding tools."""

    ALLOW_SELECT = False
    HIGHLIGHT_STYLE = ""

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        background: $surface;
        overflow-y: scroll;
        overflow-x: hidden;
        padding: 0 1;
    }
    ChatLog:focus {
        background-tint: transparent 0%;
    }
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("auto_scroll", True)
        super().__init__(wrap=True, highlight=False, markup=False, **kwargs)
        # History of all content for re-rendering on thinking toggle
        self._history: list[tuple[str, ...]] = []
        # Thinking block states: index in _history -> expanded bool
        self._thinking_states: dict[int, bool] = {}
        # Map line numbers to thinking history indices for click detection
        self._thinking_line_map: dict[int, int] = {}
        # Custom selection state
        self._sel_start: tuple[int, int] | None = None  # (x, y) in content coords
        self._sel_end: tuple[int, int] | None = None
        self._selecting: bool = False
        self._selected_text: str = ""
        self._last_thinking_click: tuple[int, int | None, float] | None = None
        # Flag: set by append_user, consumed by whichever assistant-side
        # render method writes first (stream flush, tool header, thinking,
        # info, error). One structural blank gets emitted exactly once per
        # user turn. Previous approach — writing a trailing NBSP row in
        # append_user — was unreliable: RichLog's wrap pass silently
        # collapsed pure-whitespace trailing rows on turns 2+, leaving
        # the prompt and response visually glued together.
        self._prompt_gap_pending: bool = False

    def write(self, content, *args, **kwargs) -> "ChatLog":
        """Override to always constrain width to the visible area."""
        try:
            w = self._content_width()
            if w > 10 and "width" not in kwargs:
                kwargs["width"] = w
        except Exception:
            pass
        return super().write(content, *args, **kwargs)

    def _content_width(self, *, fallback: int = 76) -> int:
        """Actual visible width for chat content.

        Use the widget width, not the app width. The app can be wider
        than the RichLog content area because of screen padding,
        scrollbars, docked widgets, or future split panes; using it
        caused wrapped text to stay visually stale after terminal
        resizes.
        """
        try:
            w = int(self.size.width) - 2  # RichLog horizontal padding/scrollbar
        except Exception:
            w = fallback
        if w <= 0:
            try:
                w = int(self.app.size.width) - 6
            except Exception:
                w = fallback
        return max(w, 40)

    def on_mount(self) -> None:
        """Apply custom Rich theme for Slack-style inline code."""
        if hasattr(self, "_console"):
            self._console.push_theme(_CUSTOM_THEME)

    def on_resize(self, event) -> None:
        """Re-render all content on terminal resize so text reflows to new width.

        CRITICAL: do NOT rerender while a stream is in progress. `_rerender`
        clears `self.lines` and replays from `self._history`, but the
        in-flight streamed prose isn't in `_history` yet (it's recorded
        only at `finish_stream(full_text)`). A resize mid-turn — which
        Textual fires every time the floating #active-step widget toggles
        visibility between `display:none` and `display:block` at each tool
        start — would wipe the prose the model just streamed. That was
        the actual root cause of 'white text disappears when tool starts'
        — not pop-and-rewrite. The width won't change meaningfully during
        streaming (the floating widget is 1 row tall); skipping rerender
        keeps prose intact, and the next non-streaming resize will reflow
        normally.
        """
        if self._history and not getattr(self, "_stream_started", False):
            self._rerender()

    def on_mouse_down(self, event) -> None:
        """Start selection on mouse down."""
        self._sel_start = (event.x, event.y + int(self.scroll_offset.y))
        self._sel_end = self._sel_start
        self._selecting = True
        self._selected_text = ""
        self.capture_mouse()
        # Last mouse-event y in widget-local coords; the edge-scroll
        # timer reads this to decide direction.
        self._drag_local_y: int = event.y
        # Edge-scroll timer — only started when the drag hits an edge.
        # Textual Timer; stopped on mouse-up or when the drag moves
        # back into the viewport interior.
        self._autoscroll_timer = None

    # Number of rows from top/bottom within which we treat the drag as
    # "at the edge" and start auto-scrolling. 1–2 is the sweet spot:
    # too small and users have to drag pixel-perfectly; too large and
    # the chat scrolls out from under them mid-selection.
    _AUTOSCROLL_EDGE = 1
    # Interval (seconds) at which the edge-scroll timer fires. Lower =
    # smoother / faster auto-scroll; too low spams the render loop.
    _AUTOSCROLL_INTERVAL = 0.05

    def _edge_scroll_tick(self) -> None:
        """Fire while the user is dragging at the viewport edge.

        Mirrors the native-terminal behavior minimal TUI and agent both
        implement: when the mouse is pinned to the top/bottom of the
        selectable area during drag, the view scrolls and the selection
        focus follows. Without this, users can only select what's
        already visible.
        """
        if not self._selecting:
            self._stop_autoscroll()
            return
        y = getattr(self, "_drag_local_y", 0)
        h = self.size.height
        direction = 0
        if y <= self._AUTOSCROLL_EDGE:
            direction = -1          # drag at top → scroll upward (show earlier content)
        elif y >= h - self._AUTOSCROLL_EDGE - 1:
            direction = 1           # drag at bottom → scroll downward

        if direction == 0:
            # Drag has moved away from the edge; stop the timer.
            self._stop_autoscroll()
            return

        before = int(self.scroll_offset.y)
        if direction < 0:
            self.scroll_up(animate=False)
        else:
            self.scroll_down(animate=False)
        after = int(self.scroll_offset.y)
        if after == before:
            # Hit the scroll boundary; no more rows to reveal that way.
            self._stop_autoscroll()
            return

        # Extend the selection focus to the new edge row so the
        # highlight keeps up with the scroll.
        if self._sel_end is not None:
            x = self._sel_end[0]
            new_y = after + (0 if direction < 0 else h - 1)
            self._sel_end = (x, new_y)
        self.refresh()

    def _start_autoscroll(self) -> None:
        if self._autoscroll_timer is not None:
            return
        try:
            self._autoscroll_timer = self.set_interval(
                self._AUTOSCROLL_INTERVAL, self._edge_scroll_tick
            )
        except Exception:
            self._autoscroll_timer = None

    def _stop_autoscroll(self) -> None:
        t = self._autoscroll_timer
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
        self._autoscroll_timer = None

    def on_mouse_move(self, event) -> None:
        """Extend selection while dragging; auto-scroll if at edge."""
        if not self._selecting:
            return
        self._drag_local_y = event.y
        self._sel_end = (event.x, event.y + int(self.scroll_offset.y))
        self.refresh()

        # Engage / disengage the edge-scroll timer based on whether
        # the current mouse row is within _AUTOSCROLL_EDGE of either
        # viewport boundary.
        h = self.size.height
        at_edge = (event.y <= self._AUTOSCROLL_EDGE
                   or event.y >= h - self._AUTOSCROLL_EDGE - 1)
        if at_edge:
            self._start_autoscroll()
        else:
            self._stop_autoscroll()

    def on_mouse_up(self, event) -> None:
        """Finalize selection and extract text."""
        if not self._selecting:
            return
        self._selecting = False
        self._stop_autoscroll()
        self.release_mouse()
        self._sel_end = (event.x, event.y + int(self.scroll_offset.y))

        # Check if it was a click (no drag) — handle thinking toggle.
        # dy previously required `<= 0`, which with abs() collapses to
        # "dy must equal 0 exactly". A single row of jitter between the
        # mouse-down and mouse-up frames killed the click, which is why
        # the thinking-cell ▶/▼ toggle appeared broken — almost no
        # clicks satisfied the strict equality. Allow 1 row of drift;
        # anything beyond that is a real drag (text selection).
        if self._sel_start and self._sel_end:
            dx = abs(self._sel_end[0] - self._sel_start[0])
            dy = abs(self._sel_end[1] - self._sel_start[1])
            if dx <= 2 and dy <= 1:
                # Clean click — check for thinking toggle
                if self._handle_thinking_click(
                    self._sel_start[1],
                    visible_y=event.y,
                    x=event.x,
                ):
                    try:
                        event.stop()
                    except Exception:
                        pass
                self._sel_start = None
                self._sel_end = None
                return

        # Extract selected text and set clipboard via OSC 52
        if self._sel_start and self._sel_end:
            self._selected_text = self._extract_selection()
            if self._selected_text.strip():
                self._set_clipboard_osc52(self._selected_text)

    def on_click(self, event) -> None:
        """Direct click fallback for thinking disclosure rows."""
        try:
            if self._handle_thinking_click(
                event.y + int(self.scroll_offset.y),
                visible_y=event.y,
                x=event.x,
            ):
                event.stop()
        except Exception:
            pass

    def _set_clipboard_osc52(self, text: str) -> None:
        """Set clipboard via OSC 52 escape sequence — works with iTerm2, Terminal.app, etc."""
        import base64
        import os
        encoded = base64.b64encode(text.encode()).decode()
        # Write directly to terminal fd, bypassing Textual's stdout
        try:
            fd = os.open("/dev/tty", os.O_WRONLY)
            os.write(fd, f"\033]52;c;{encoded}\a".encode())
            os.close(fd)
        except OSError:
            # Fallback to pbcopy
            import subprocess
            subprocess.run(["pbcopy"], input=text.encode(), check=True)

    def _thinking_hist_at_y(self, y: int, *, tolerance: int = 2) -> int | None:
        """Return the thinking history entry nearest a clicked content row."""
        if not self._thinking_line_map:
            return None
        closest_line, hist_idx = min(
            self._thinking_line_map.items(),
            key=lambda item: abs(item[0] - y),
        )
        if abs(closest_line - y) <= tolerance:
            return hist_idx
        return None

    def _toggle_thinking_entry(self, hist_idx: int) -> None:
        self._thinking_states[hist_idx] = not self._thinking_states.get(hist_idx, False)
        self._rerender()

    def _is_duplicate_thinking_click(self, y: int, x: int | None) -> bool:
        last = self._last_thinking_click
        now = time.monotonic()
        if last is None:
            self._last_thinking_click = (y, x, now)
            return False
        last_y, last_x, last_at = last
        self._last_thinking_click = (y, x, now)
        if now - last_at > 0.25:
            return False
        if abs(last_y - y) > 1:
            return False
        if x is None or last_x is None:
            return True
        return abs(last_x - x) <= 2

    def _line_text_at(self, y: int) -> str:
        if not (0 <= y < len(self.lines)):
            return ""
        line = self.lines[y]
        return getattr(line, "text", str(line))

    def _looks_like_thinking_header(self, text: str) -> bool:
        return (
            "▶" in text
            or "▼" in text
            or "Thinking Process" in text
            or "thinking (" in text
        )

    def _nearest_or_latest_thinking(self, y: int, *, tolerance: int = 12) -> int | None:
        hist_idx = self._thinking_hist_at_y(y, tolerance=tolerance)
        if hist_idx is not None:
            return hist_idx
        if self._thinking_states:
            return max(self._thinking_states)
        for idx in range(len(self._history) - 1, -1, -1):
            if self._history[idx][0] == "thinking":
                return idx
        return None

    def _handle_thinking_click(
        self,
        y: int,
        *,
        visible_y: int | None = None,
        x: int | None = None,
    ) -> bool:
        """Toggle thinking expand/collapse on click.

        Mouse events are converted to content coordinates before they
        reach this method. The thinking headers are also tracked in
        content coordinates, so do not bound lookup by `len(self.lines)`:
        that value can represent the visible/rendered buffer after
        scroll or wrap changes and made clicks miss the stored header.
        """
        try:
            hist_idx = self._thinking_hist_at_y(y)
            if hist_idx is not None:
                if self._is_duplicate_thinking_click(y, x):
                    return True
                self._toggle_thinking_entry(hist_idx)
                return True

            # The disclosure triangle is rendered in the left gutter
            # ("  ▶ ..."). Treat that gutter as clickable only near a
            # known/header-looking thinking row; otherwise clicking the
            # margin while selecting old transcript text would toggle
            # the latest hidden reasoning block.
            if x is not None and x <= 6:
                hist_idx = self._thinking_hist_at_y(y, tolerance=4)
                if hist_idx is not None:
                    if self._is_duplicate_thinking_click(y, x):
                        return True
                    self._toggle_thinking_entry(hist_idx)
                    return True

            # Defensive fallback for Textual/RichLog coordinate drift:
            # if the row looks like a thinking header in the visible
            # buffer, toggle the nearest stored thinking block rather
            # than blindly toggling the first one.
            viewport_y = y - int(self.scroll_offset.y) if visible_y is None else visible_y
            candidate_rows = {
                y - 1, y, y + 1,
                viewport_y - 1, viewport_y, viewport_y + 1,
            }
            for candidate_y in candidate_rows:
                line_text = self._line_text_at(candidate_y)
                if self._looks_like_thinking_header(line_text):
                    hist_idx = self._nearest_or_latest_thinking(y)
                    if hist_idx is not None:
                        if self._is_duplicate_thinking_click(y, x):
                            return True
                        self._toggle_thinking_entry(hist_idx)
                        return True
        except Exception:
            pass
        return False

    def _extract_selection(self) -> str:
        """Extract plain text from the selected region."""
        if not self._sel_start or not self._sel_end:
            return ""
        # Normalize: start should be before end
        y1, y2 = self._sel_start[1], self._sel_end[1]
        x1, x2 = self._sel_start[0], self._sel_end[0]
        if y1 > y2 or (y1 == y2 and x1 > x2):
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        lines_text = []
        for y in range(y1, y2 + 1):
            if 0 <= y < len(self.lines):
                lt = self.lines[y].text
                if y == y1 and y == y2:
                    lines_text.append(lt[x1:x2])
                elif y == y1:
                    lines_text.append(lt[x1:])
                elif y == y2:
                    lines_text.append(lt[:x2])
                else:
                    lines_text.append(lt)
        return "\n".join(lines_text)

    def get_selection(self) -> str:
        """Return currently selected text — extract fresh from current selection."""
        if self._sel_start and self._sel_end:
            return self._extract_selection()
        return self._selected_text

    def _sel_range(self) -> tuple[int, int, int, int] | None:
        """Get normalized selection range: (y1, x1, y2, x2)."""
        if not getattr(self, '_sel_start', None) or not getattr(self, '_sel_end', None):
            return None
        y1, x1 = self._sel_start[1], self._sel_start[0]
        y2, x2 = self._sel_end[1], self._sel_end[0]
        if y1 > y2 or (y1 == y2 and x1 > x2):
            y1, y2 = y2, y1
            x1, x2 = x2, x1
        return (y1, x1, y2, x2)

    def render_line(self, y: int) -> Strip:
        """Override to apply selection highlight."""
        strip = super().render_line(y)
        sel = self._sel_range()
        if not sel:
            return strip
        y1, x1, y2, x2 = sel
        # Convert to content coords (account for scroll)
        content_y = y + int(self.scroll_offset.y)
        if content_y < y1 or content_y > y2:
            return strip  # not in selection
        # Apply highlight to selected region
        highlight = Style(bgcolor="#264f78")
        new_segments = []
        col = 0
        for segment in strip._segments:
            seg_len = len(segment.text)
            if content_y == y1 and content_y == y2:
                # Single line selection
                for i, ch in enumerate(segment.text):
                    c = col + i
                    if x1 <= c < x2:
                        new_segments.append(Segment(ch, highlight))
                    else:
                        new_segments.append(Segment(ch, segment.style))
            elif content_y == y1:
                for i, ch in enumerate(segment.text):
                    if col + i >= x1:
                        new_segments.append(Segment(ch, highlight))
                    else:
                        new_segments.append(Segment(ch, segment.style))
            elif content_y == y2:
                for i, ch in enumerate(segment.text):
                    if col + i < x2:
                        new_segments.append(Segment(ch, highlight))
                    else:
                        new_segments.append(Segment(ch, segment.style))
            else:
                # Fully selected line
                new_segments.append(Segment(segment.text, highlight))
            col += seg_len
        return Strip(new_segments)

    def _track_lines(self, count: int = 1) -> int:
        """Track current line position, return starting line."""
        start = getattr(self, '_line_counter', 0)
        self._line_counter = start + count
        return start

    # ── Public append methods (record to history + render) ──

    def _spacer(self) -> None:
        self._history.append(("spacer",))
        self.write(Text(""))
        self._track_lines()

    def append_user(self, text: str) -> None:
        self._dispatch_gap("user")
        self._history.append(("user", text))
        # Reset streaming state so the next token stream knows it's fresh.
        self._stream_started = False
        self._render_user(text)
        # The blank between prompt and assistant output is emitted by
        # whichever assistant-side method fires first. See
        # _ensure_prompt_gap(). Keeping the gap off the trailing edge
        # of this write avoids the whitespace-collapse bug that hid it
        # from turn 2 onward.
        self._prompt_gap_pending = True

    # ── Spacing contract ─────────────────────────────────────────────
    #
    # agent pattern (src/components/VirtualMessageList.tsx): one source
    # of truth decides how many blank rows go BEFORE each new entry,
    # based on what the previous entry was. Render methods own only
    # CONTENT — no leading/trailing whitespace tricks. This kills the
    # whole bug class of "spacing drifts between turns" by removing
    # the per-method ad-hoc spacing logic that kept getting it wrong.
    #
    # Default: 1 blank row before any new entry. Exception list below
    # specifies pairs that should be tight (zero blanks): consecutive
    # tools, tool→done, tool→result, etc.

    _TIGHT_PAIRS: set[tuple[str, str]] = {
        ("tool", "tool"),
        ("tool", "tool_done"),
        ("tool", "tool_result"),
        ("tool_done", "tool_result"),
        ("tool_done", "tool"),
        ("tool_result", "tool"),
        ("user", "queued"),
    }

    def _gap_between(self, prev_kind: str | None, this_kind: str) -> int:
        """Return how many blank rows should precede `this_kind` given
        `prev_kind`. None means no prior entry → no leading blank."""
        if prev_kind is None:
            return 0
        if (prev_kind, this_kind) in self._TIGHT_PAIRS:
            return 0
        return 1

    def _dispatch_gap(self, kind: str) -> None:
        """Write the spacing rows that precede `kind`, then update
        `_last_kind`. Every append_* MUST call this first.

        Idempotent: the streaming flush's orphan-recovery branch
        (`_flush_stream` line ~847) and Markdown's per-block trailing
        whitespace can leave stray blank Strips at the tail that the
        unified trim doesn't reach. If we just always wrote the target
        gap, those stray blanks would *compound* with the dispatch
        blank → 2-3 visible rows where there should be 1. Instead we
        peek at trailing blanks already on the log and only add the
        deficit. Never pops — popping risks deleting non-gap content
        the renderer thought was permanent."""
        prev = getattr(self, "_last_kind", None)
        target_gap = self._gap_between(prev, kind)
        existing = 0
        n_lines = len(self.lines)
        while existing < n_lines:
            line = self.lines[-(existing + 1)]
            text = getattr(line, "text", line if isinstance(line, str) else "")
            if text.strip():
                break
            existing += 1
        deficit = max(0, target_gap - existing)
        for _ in range(deficit):
            # NBSP keeps the blank from being trimmed by Rich's wrap
            # under width-constrained renders. Plain Text("") fired
            # in some contexts but not others; NBSP fires in all.
            self.write(Text(" "))
            self._track_lines()
        self._last_kind = kind

    # The flag-based methods are kept as thin no-ops because many call
    # sites still invoke them. New code should rely on _dispatch_gap.
    def _ensure_prompt_gap(self) -> None:
        self._prompt_gap_pending = False

    def _consume_prompt_gap(self) -> None:
        self._prompt_gap_pending = False

    def append_queued(self, text: str) -> None:
        self._dispatch_gap("queued")
        self._history.append(("queued", text))
        self._render_queued(text)

    def append_assistant(self, text: str) -> None:
        self._dispatch_gap("assistant")
        self._history.append(("assistant", text))
        self._render_assistant(text)

    _step_count: int = 0

    def reset_steps(self) -> None:
        self._step_count = 0

    def append_tool(self, name: str, args: str) -> None:
        self._dispatch_gap("tool")
        self._history.append(("tool", name, args))
        self._render_tool(name, args)

    def append_tool_done(self, name: str, args: str, summary: str) -> None:
        self._dispatch_gap("tool_done")
        self._history.append(("tool_done", name, args, summary))
        self._render_tool_done(name, args, summary)

    def append_tool_result(self, result: str, error: bool = False) -> None:
        self._dispatch_gap("tool_result")
        self._history.append(("tool_result", result, "true" if error else "false"))
        self._render_tool_result(result, error)

    def append_thinking(self, text: str, expanded: bool = False) -> None:
        self._dispatch_gap("thinking")
        idx = len(self._history)
        self._history.append(("thinking", text))
        self._thinking_states[idx] = expanded
        self._render_thinking(text, expanded, idx)

    def append_approval(self, tool_name: str, command: str) -> None:
        self._dispatch_gap("approval")
        self._history.append(("approval", tool_name, command))
        self._render_approval(tool_name, command)

    def append_info(self, text: str) -> None:
        self._dispatch_gap("info")
        self._history.append(("info", text))
        self._render_info(text)

    def append_error(self, text: str) -> None:
        self._dispatch_gap("error")
        self._history.append(("error", text))
        self._render_error(text)

    def append_turn_summary(self, elapsed: float, tools: list[str],
                            tokens_in: int = 0, tokens_out: int = 0,
                            tokens_total: int = 0) -> None:
        """Render the per-turn summary line.

        The UI intentionally shows generated tokens only. Prompt/input
        token counts remain in telemetry/events for engineering analysis,
        but showing `in: 3.0k` beside a two-character "hi" makes users
        think LocalCode counted their typed input wrong. agent/terminal coding tools-style
        chat summaries optimize for user comprehension: time, tools, and
        visible output size.
        """
        def _fmt(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)
        parts = []
        if elapsed < 60:
            parts.append(f"{elapsed:.1f}s")
        else:
            m, s = divmod(int(elapsed), 60)
            parts.append(f"{m}m{s:02d}s")
        if tools:
            from collections import Counter
            counts = Counter(tools)
            tool_parts = []
            for name, count in counts.items():
                header = _TOOL_HEADERS.get(name, (name, name))
                label = header[0]
                if count > 1:
                    tool_parts.append(f"{label} x{count}")
                else:
                    tool_parts.append(label)
            parts.append(", ".join(tool_parts))
        if tokens_out:
            parts.append(f"{_fmt(tokens_out)} tokens")
        summary_text = " · ".join(parts)
        self._dispatch_gap("turn_summary")
        self._history.append(("turn_summary", summary_text))
        self._render_turn_summary(summary_text)

    def _stream_avail_width(self) -> int:
        """Available width for streamed text (matches _render_assistant)."""
        return self._content_width() - 2  # 2 for leading indent

    def _write_wrapped(self, line: str) -> None:
        """Write a line with word-wrapping to fit the widget width."""
        import textwrap
        avail = self._stream_avail_width()
        if not line.strip():
            self.write(Text(f"  {line}"))
            self._track_lines()
            return
        wrapped = textwrap.fill(line, width=avail)
        for wline in wrapped.split("\n"):
            self.write(Text(f"  {wline}"))
            self._track_lines()

    # Streaming perf — agent pattern: stable prefix never re-renders,
    # only the trailing partial does. Plus three perceived-speed wins:
    #
    # 1. Coalesce window 100 ms → 33 ms = ~30 fps screen updates instead
    #    of ~10. At 27 tok/s decode this is the difference between
    #    "tokens land instantly" and "noticeable lag". Pop+rewrite cost
    #    is dominated by Strip object construction, not flush rate, so
    #    3× the flush rate ≈ 1.2× the work — well worth it.
    # 2. Eager flush on word boundary (' ' or punctuation) so text
    #    appears word-by-word to the eye, not chunk-by-chunk. At ~5 chars
    #    per word + 33 ms cadence, the practical effective rate is one
    #    visible word every 30-50 ms — feels real-time.
    # 3. Skip scroll_end when already at bottom (the common case during
    #    streaming) — saves ~1-3 ms per flush of layout work.
    _STREAM_COALESCE_SEC = 0.033
    _STREAM_WORD_BOUNDARIES = " .,;:!?\n"

    def _render_partial_line(self, line_text: str) -> None:
        """Write the current partial line (no fade, no re-style work)."""
        avail = self._stream_avail_width()
        display = f"  {line_text[:avail]}" if len(line_text) > avail else f"  {line_text}"
        self.write(Text(display))
        self._track_lines()

    def _render_assistant_to_lines(self, text: str) -> list:
        """Single source of truth for assistant-text rendering.

        Returns the list of Rich Text objects (one per visual row) that
        the SAME text would produce whether rendered mid-stream or at
        end-of-turn replay. Both `_flush_stream` (incremental) and
        `_render_assistant` (history replay) call this — guaranteeing
        zero visual drift between "what I see while it streams" and
        "what I see when streaming finishes". Without this the user
        sees a 'snap to even spacing' as raw text gets re-laid-out
        through Markdown at the moment streaming completes.
        """
        avail_w = self._content_width()
        # Cheap fast-path mirroring agent's hasMarkdownSyntax (~3ms
        # saved per render on plain text). Code-fence / heading / bold
        # / list / inline-code are the markers that actually matter for
        # IQ2_M model output; checking presence first avoids feeding
        # plain prose through commonmark for every chunk.
        has_md = any(m in text for m in ("```", "###", "**", "- ", "1. ", "`"))
        out: list = []
        if has_md:
            from rich.console import Console
            from io import StringIO
            buf = StringIO()
            console = Console(
                file=buf, width=avail_w - 2,
                force_terminal=True, color_system="truecolor",
            )
            console.print(Markdown(text, code_theme="monokai"))
            rendered = buf.getvalue()
            # `console.print` always ends with a newline → split would
            # yield a trailing empty string that becomes a permanent
            # blank row tacked onto the response. Trim it once here.
            if rendered.endswith("\n"):
                rendered = rendered[:-1]
            for rline in rendered.split("\n"):
                out.append(Text.from_ansi(f"  {rline}"))
        else:
            import textwrap
            for line in text.split("\n"):
                if line.strip():
                    wrapped = textwrap.fill(line, width=avail_w - 2)
                    for wline in wrapped.split("\n"):
                        out.append(Text(f"  {wline}"))
                else:
                    out.append(Text("  "))
        # Strip trailing blank rows. Models routinely emit a trailing
        # `\n` (or two), and `console.print(Markdown(…))` always tacks
        # on its own newline. Without this trim, every assistant turn
        # ends with 1-2 blank rows that PILE ON TOP OF the next entry's
        # `_dispatch_gap` blank — visually 3+ rows of whitespace between
        # one round's response and the next round's tool call. End-of-
        # turn `_rerender` happens to re-pop these (because it replays
        # cleanly from history), so the spacing "snaps right" only after
        # streaming finishes — exactly what the user keeps reporting.
        while out and not getattr(out[-1], "plain", "").strip():
            out.pop()
        return out

    def _init_stream_state(self) -> None:
        self._stream_text = ""            # unflushed chars (after last newline)
        self._stream_full = ""            # full accumulated text for markdown render
        self._stream_buf = ""             # coalesce buffer since last flush
        self._stream_started = False
        self._stream_line_idx = -1        # legacy single-line tracker (unused by progressive renderer)
        self._stream_start_idx = -1       # index where this stream's lines begin
        self._stream_lines_written = 0    # how many lines our block currently occupies
        self._stream_timer = None         # pending Textual Timer

    def stream_token(self, token: str) -> None:
        """Accept a token. Accumulate; flush ≤10/s or on newline."""
        if not hasattr(self, '_stream_text'):
            self._init_stream_state()
        self._stream_full += token
        self._stream_buf += token

        # First token of a new stream: consume the pending prompt gap
        # (one blank row between ❯ prompt and response), then record
        # where content begins so finish_stream's markdown rewrite
        # knows what to pop. Gap is written BEFORE _stream_start_idx
        # so the pop-and-rewrite on the markdown path doesn't delete
        # the blank we just emitted.
        if not self._stream_started:
            # `_ensure_prompt_gap` was the legacy hook — it's a no-op now
            # since the spacing system was centralized into `_dispatch_gap`
            # (see `_render_*` callers). Without the dispatch_gap call here,
            # the streaming render had NO leading blank between the user
            # prompt and the assistant content; that gap only appeared
            # after end-of-turn `_rerender` re-played history through the
            # gap-dispatching path → visible "snap" the user saw at the
            # moment streaming finished. Calling _dispatch_gap("assistant")
            # here makes streaming and history-replay produce the same
            # vertical layout.
            self._dispatch_gap("assistant")
            self._stream_started = True
            self._stream_line_idx = -1
            # Record start index AFTER the gap so the pop-and-rewrite on
            # subsequent flushes doesn't delete the blank we just added.
            self._stream_start_idx = len(self.lines)

        # Flush immediately on newline OR word boundary. Newlines bound
        # the partial-line size; word boundaries make text appear
        # word-by-word visually rather than chunked at the coalesce
        # interval. At 27 tok/s and ~5 chars/word, this gives one
        # visible word every ~30-50 ms — feels real-time without the
        # constant per-character flush overhead.
        last_char = self._stream_buf[-1] if self._stream_buf else ""
        if "\n" in self._stream_buf or last_char in self._STREAM_WORD_BOUNDARIES:
            self._cancel_stream_timer()
            self._flush_stream()
            return

        # Otherwise schedule a coalesced flush. One timer only — subsequent
        # tokens in the window just extend _stream_buf.
        if self._stream_timer is None:
            try:
                self._stream_timer = self.set_timer(
                    self._STREAM_COALESCE_SEC, self._flush_stream
                )
            except Exception:
                # If we're somehow not mounted yet, fall back to eager flush.
                self._flush_stream()

    def _cancel_stream_timer(self) -> None:
        t = getattr(self, "_stream_timer", None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
            self._stream_timer = None

    def _flush_stream(self) -> None:
        """Re-render the full accumulated stream through the SAME
        markdown path used at end-of-turn. Result: streaming output
        looks identical to the final render, so the user no longer
        sees a "snap to even spacing" when streaming completes (raw
        text → markdown reflow was visible at finish; now the markdown
        IS what's on screen the whole time).

        Strategy:
          1. If our block is still at the tail of self.lines, pop it
             entirely, re-render `_stream_full` via _render_assistant_to_lines,
             write the result, and update _stream_lines_written.
          2. If our block lost the tail (a tool widget or rerender wrote
             after our last flush — see comment in `finish_stream`), give
             up on the rewrite-in-place: reset start_idx + lines_written
             to the new tail and write the FULL stream there. Old block's
             stale rendering will be repainted by `_rerender` from history
             on the next on_resize / end-of-turn cycle.
        """
        self._stream_timer = None
        if not getattr(self, "_stream_started", False):
            return
        buf = self._stream_buf
        self._stream_buf = ""
        if not buf:
            return
        # Keep `_stream_text` updated for any legacy reader; the new
        # path renders from `_stream_full` directly.
        self._stream_text += buf

        # Capture "is the user following the tail?" BEFORE we add any
        # content. If the user scrolled up to read history mid-stream,
        # we leave their scroll position alone; only snap to the new
        # bottom when they were already there. Earlier code's condition
        # was inverted ("scroll if NOT at bottom") which actively fought
        # the user's scroll-up — every flush yanked them back to the
        # tail. Within 2 rows of max counts as "following" so a stray
        # pixel doesn't stick the user mid-history forever.
        _was_following_tail = (
            self.scroll_offset.y >= self.max_scroll_y - 2
        )

        expected_tail = self._stream_start_idx + self._stream_lines_written
        is_at_tail = expected_tail == len(self.lines)
        if is_at_tail and self._stream_lines_written > 0:
            # Pop our entire block. Safe because the at-tail check
            # guarantees nothing else lives in this index range.
            for _ in range(self._stream_lines_written):
                self.lines.pop()
            self._line_counter = max(0, self._line_counter - self._stream_lines_written)
            self._line_cache.clear()
            self._stream_lines_written = 0
        elif not is_at_tail:
            # Lost the tail — reset to the new bottom of the log. The
            # earlier block stays where it was (with its now-stale
            # render); _rerender on the next layout event will replace
            # it from `_history` cleanly.
            self._stream_start_idx = len(self.lines)
            self._stream_lines_written = 0

        # Render the whole accumulated stream via the unified helper.
        # Identical output to end-of-turn `_render_assistant`, so when
        # streaming ends and history-replay runs, nothing visually moves.
        rendered = self._render_assistant_to_lines(self._stream_full)
        for line in rendered:
            self.write(line)
            self._track_lines()
            self._stream_lines_written += 1

        # Auto-follow tail ONLY when the user was already at the bottom
        # before we added content. If they scrolled up to read history,
        # _was_following_tail is False and we don't move them.
        if _was_following_tail:
            self.scroll_end(animate=False)

    def finish_stream(self, full_text: str = "") -> None:
        """Finalize streaming. Drain any pending buffer, record what was
        streamed into history, reset stream state. No pop, no Markdown
        rewrite.

        CRITICAL: the history append happens on EVERY finish_stream
        call (not just end-of-turn). Reason: mid-turn, Textual fires
        `on_resize` whenever the floating #active-step widget toggles
        between display:none and display:block at each tool_start,
        which triggers `_rerender()` → `super().clear()` and replay
        from `_history`. If this round's streamed prose isn't in
        `_history` yet, rerender wipes it. Recording per-round ensures
        a mid-turn rerender recovers the text instead of destroying it.
        """
        # Drain any pending coalesce buffer so the last partial line is
        # committed to display before the next event (tool_start, etc.).
        self._cancel_stream_timer()
        if getattr(self, "_stream_buf", ""):
            self._flush_stream()

        # Record THIS round's streamed prose. `_stream_full` holds only
        # the current round (reset by `_init_stream_state` below), so
        # recording it here produces exactly one history entry per
        # round — no double-append across rounds. We deliberately do
        # NOT fall back to `full_text` here because the chat screen's
        # response_done handler passes the CUMULATIVE assistant text
        # across all rounds; if earlier rounds were already recorded
        # per-round (they were), adding the cumulative text would
        # duplicate everything in history.
        round_text = getattr(self, "_stream_full", "").strip()
        if round_text and getattr(self, "_stream_started", False):
            self._history.append(("assistant", round_text))

        # Reset stream bookkeeping so the next stream round starts fresh.
        if getattr(self, "_stream_started", False):
            self._init_stream_state()

        # Same auto-follow gate as `_flush_stream` — don't snap the
        # user back to the bottom at end-of-turn if they scrolled up
        # to read history during the stream.
        if self.scroll_offset.y >= self.max_scroll_y - 2:
            self.scroll_end(animate=False)

    # ── Search ──

    def search_text(self, query: str) -> list[tuple[int, str]]:
        """Search all history entries for query (case-insensitive).

        Returns list of (history_index, matching_snippet) tuples.
        """
        if not query:
            return []
        q = query.lower()
        results = []
        for i, entry in enumerate(self._history):
            for part in entry[1:]:
                text = str(part)
                if q in text.lower():
                    # Extract a snippet around the match
                    idx = text.lower().index(q)
                    start = max(0, idx - 40)
                    end = min(len(text), idx + len(query) + 40)
                    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                    results.append((i, snippet))
                    break  # one match per history entry
        return results

    def scroll_to_history_index(self, hist_idx: int) -> None:
        """Scroll to approximate position of a history entry."""
        # Estimate line position by counting entries before this one
        line = 0
        for i, entry in enumerate(self._history):
            if i >= hist_idx:
                break
            kind = entry[0]
            if kind in ("spacer",):
                line += 1
            elif kind in ("user", "info", "error", "turn_summary"):
                line += 2
            elif kind == "assistant":
                line += max(2, len(str(entry[1]).splitlines()))
            elif kind in ("tool", "tool_done"):
                line += 1
            elif kind == "tool_result":
                line += min(6, len(str(entry[1]).splitlines())) + 1
            elif kind == "thinking":
                line += 2
            else:
                line += 1
        self.scroll_to(y=max(0, line - 2), animate=False)

    def clear(self) -> None:
        """Clear display and history."""
        super().clear()
        if not hasattr(self, '_rerendering'):
            self._history.clear()
            self._thinking_states.clear()
            self._thinking_line_map.clear()
        self._line_counter = 0

    def _rerender(self) -> None:
        """Clear display and re-render from history (preserves history).

        Uses the SAME spacing contract as the live path: each entry's
        kind is dispatched through `_dispatch_gap`, then the body is
        rendered with no leading/trailing whitespace. Identical visual
        output to the original turn — no drift after a resize.
        """
        self._rerendering = True
        saved_scroll = self.scroll_offset.y
        super().clear()
        self._thinking_line_map.clear()
        self._line_counter = 0
        # Reset spacing state so the first entry has no leading blank
        self._last_kind = None
        for i, entry in enumerate(self._history):
            kind = entry[0]
            if kind == "spacer":
                self.write(Text(""))
                self._track_lines()
            elif kind == "raw":
                self.write(entry[1])
                self._track_lines()
            else:
                # Centralized spacing: one rule for every kind transition.
                self._dispatch_gap(kind)
                if kind == "user":
                    self._render_user(entry[1])
                elif kind == "assistant":
                    self._render_assistant(entry[1])
                elif kind == "thinking":
                    expanded = self._thinking_states.get(i, False)
                    self._render_thinking(entry[1], expanded, i)
                elif kind == "tool":
                    self._render_tool(entry[1], entry[2])
                elif kind == "tool_done":
                    self._render_tool_done(entry[1], entry[2], entry[3])
                elif kind == "tool_result":
                    self._render_tool_result(entry[1], entry[2] == "true")
                elif kind == "info":
                    self._render_info(entry[1])
                elif kind == "error":
                    self._render_error(entry[1])
                elif kind == "turn_summary":
                    self._render_turn_summary(entry[1])
                elif kind == "approval":
                    self._render_approval(entry[1], entry[2])
                elif kind == "queued":
                    self._render_queued(entry[1])
        del self._rerendering
        # Force layout recalculation then restore scroll
        self.refresh(layout=True)
        self.scroll_to(y=saved_scroll, animate=False)

    # ── Render methods (write to display, no history recording) ──

    def _render_user(self, text: str) -> None:
        # Spacing before the prompt is owned by `_dispatch_gap("user")`;
        # an explicit leading blank here doubled up with it. The
        # trailing blank below the prompt is handled by the next entry's
        # `_dispatch_gap` (since prev=user is not in any TIGHT_PAIR with
        # the assistant/tool that follows).
        line = Text()
        line.append("❯ ", style=f"bold {C.primary}")
        line.append(text, style="bold white")
        self.write(line)
        self._track_lines()

    def _render_queued(self, text: str) -> None:
        line = Text()
        line.append("  ↻ ", style="dim yellow")
        line.append(text, style="dim italic")
        line.append("  (queued)", style="dim yellow")
        self.write(line)
        self._track_lines()

    def _render_assistant(self, text: str) -> None:
        # Delegate to the unified renderer so end-of-turn history
        # replay produces byte-identical output to mid-stream rendering.
        # That equality is the load-bearing invariant for "no snap when
        # streaming finishes" — if these diverge by even one line, the
        # user sees a reflow at finish_stream / _rerender.
        for line in self._render_assistant_to_lines(text):
            self.write(line)
            self._track_lines()

    def _render_thinking(self, text: str, expanded: bool, hist_idx: int) -> None:
        lines = text.strip().splitlines()
        if not lines:
            return
        # Spacing before the thinking header is owned by
        # `_dispatch_gap("thinking")`; an explicit leading blank here
        # was redundant and stacked with the dispatch blank.
        # Record the line number for click detection.
        # Use `len(self.lines)` (RichLog's content-row index) rather than
        # the per-write `_line_counter`. RichLog stores one Strip per
        # rendered row; the click handler converts mouse `event.y +
        # scroll_offset.y` into the SAME index space, so the map key
        # must come from `self.lines` to avoid drift on turns where any
        # earlier write expanded into more than one rendered row.
        header_line = len(self.lines)
        self._thinking_line_map[header_line] = hist_idx
        header = Text()
        if expanded:
            header.append("  ▼ ", style="cyan")
            header.append(f"thinking ({len(lines)} lines)", style="dim italic cyan")
        else:
            header = Text(no_wrap=True, overflow="ellipsis")
            header.append("  ▶ ", style="cyan")
            try:
                max_preview = self._content_width() - 22
            except Exception:
                max_preview = 50
            max_preview = max(max_preview, 30)
            preview = lines[0][:max_preview]
            if len(lines) > 1 or len(lines[0]) > max_preview:
                preview = preview.rstrip() + "…"
            header.append(preview, style="dim italic")
            if len(lines) > 1:
                header.append(f"  (+{len(lines) - 1} lines)", style="dim cyan")
        self.write(header)
        self._track_lines()

        if expanded:
            import textwrap
            avail_w = self._content_width()
            max_show = min(len(lines), 30)
            for line_text in lines[:max_show]:
                wrapped = textwrap.fill(line_text, width=avail_w - 4,
                                        initial_indent="    ",
                                        subsequent_indent="    ")
                for wline in wrapped.split("\n"):
                    tl = Text(wline, style="dim italic")
                    self.write(tl)
                    self._track_lines()
            if len(lines) > max_show:
                more = Text()
                more.append(f"    … +{len(lines) - max_show} more lines", style="dim italic")
                self.write(more)
                self._track_lines()
            # Spacer after expanded thinking
            self.write(Text(""))
            self._track_lines()

    def _visible_width(self) -> int:
        """How many columns of actual content fit on a line, accounting
        for widget padding + scrollbar + a 2-col safety buffer. Rich's
        no_wrap+ellipsis on a Text can still overflow when the passed
        width exceeds the actually-visible area; pre-truncating the
        composed string to this number makes the overflow impossible."""
        try:
            w = self.size.width - 4
        except Exception:
            w = 76
        return max(20, w)

    def _truncate_oneline(self, s: str, width: int) -> str:
        """Truncate a pre-composed one-line string at visible width with
        a trailing ellipsis. Used to pre-shorten tool-header strings
        before handing them to Rich, so wrap doesn't extend off-screen."""
        if len(s) <= width:
            return s
        return s[: max(1, width - 1)] + "…"

    def _render_tool(self, name: str, args: str) -> None:
        self._step_count += 1
        if name == "skill":
            self._render_skill_header(args, done=False)
            return
        header_info = _TOOL_HEADERS.get(name, (name, name))
        display_name = header_info[0]
        args_one = (args.strip().replace("\n", " ")) if args else ""
        # Pre-compose the line as a plain string and truncate before
        # styling. Rich's overflow="ellipsis" on Text is fragile when
        # the Text contains multiple styled spans; manual truncation
        # here makes the off-screen overflow impossible.
        prefix = f"  ● {display_name}"
        composed = f"{prefix}({args_one})" if args_one else prefix
        composed = self._truncate_oneline(composed, self._visible_width())
        # Split back into styled spans: everything up to and including
        # the display name is blue-bold, the args blob is dim.
        header = Text(no_wrap=True)
        split = len(prefix)
        header.append(composed[:split], style=f"bold {C.primary}")
        if len(composed) > split:
            header.append(composed[split:], style="dim")
        self.write(header)
        self._track_lines()

    def _render_tool_done(self, name: str, args: str, summary: str) -> None:
        if name == "skill":
            self._render_skill_header(args, done=True, summary=summary)
            return
        header_info = _TOOL_HEADERS.get(name, (name, name))
        display_name = header_info[0]
        args_one = (args.strip().replace("\n", " ")) if args else ""
        prefix = f"  ✓ {display_name}"
        body = prefix
        if args_one:
            body += f"({args_one})"
        if summary:
            body += f"  {summary}"
        body = self._truncate_oneline(body, self._visible_width())
        line = Text(no_wrap=True)
        split_name = len(prefix)
        line.append(body[:split_name], style=f"bold {C.success}")
        # Split the args+summary region: args in bright green, summary
        # in dim green. We find the args closing ')' to split cleanly.
        tail = body[split_name:]
        if args_one and tail.startswith("("):
            end = tail.find(")")
            if end == -1:
                line.append(tail, style=f"{C.success}")
            else:
                line.append(tail[: end + 1], style=f"{C.success}")
                if len(tail) > end + 1:
                    line.append(tail[end + 1 :], style=f"dim {C.success}")
        else:
            line.append(tail, style=f"dim {C.success}")
        self.write(line)
        self._track_lines()

    # Skills are loaded via the `skill` tool but read as a recipe, not a
    # plain tool call. Render them on a single line as `Skill <name>` in
    # a distinct color so the user can see at a glance WHICH skill fired
    # and distinguish it from the bash/read/grep tool stream. We drop
    # both the raw args JSON (redundant — the name is in our label) and
    # the tool-result summary (always a `# Skill: <name>` header echo of
    # the label — also redundant).
    def _render_skill_header(self, args: str, done: bool, summary: str = "") -> None:
        import re
        # Args can arrive either as JSON (double-quoted) from the model
        # OR as a Python dict repr (single-quoted) when stringified in
        # transit. Match both.
        m = (re.search(r'"name"\s*:\s*"([^"]+)"', args or "")
             or re.search(r"'name'\s*:\s*'([^']+)'", args or ""))
        skill_name = m.group(1) if m else (args or "").strip().strip("'\"{} ") or "?"
        line = Text(no_wrap=True, overflow="ellipsis")
        # Skill rows: terminal-default fg (with weight + glyph carrying
        # the meaning). No more soft violet — keeps the palette principle
        # "only the brand carries colour."
        prefix = "  ✓ " if done else "  ● "
        line.append(prefix, style="bold")
        line.append("Skill ", style="bold")
        line.append(skill_name)
        self.write(line)
        self._track_lines()

    def _render_tool_result(self, result: str, error: bool) -> None:
        if error:
            err_one = result.replace("\n", " ").strip()
            line = Text(no_wrap=True, overflow="ellipsis")
            line.append("    ⎿ ", style="dim")
            line.append(err_one, style="bold red")
            self.write(line)
            self._track_lines()
            return

        if _is_diff(result):
            self._render_diff(result)
            return

        # agent pattern: split bash-like output into stdout vs stderr
        # so warnings/errors render in a dim-red panel below the main
        # output. Heuristic — only applies if stderr-style markers are
        # detected; otherwise the whole result renders as stdout.
        from ..formatting import truncate_with_tail, split_stdout_stderr
        stdout_text, stderr_text = split_stdout_stderr(result)

        # If we extracted distinct stderr, render stdout normally and
        # then render stderr as a labeled dim-red continuation block.
        if stderr_text.strip() and stdout_text.strip():
            self._render_tool_result_block(stdout_text, "")  # stdout (no label)
            self._render_tool_result_block(stderr_text, "stderr", style_color=C.error)
            return

        # Single-stream path — original behavior.
        lines = result.strip().splitlines()
        if not lines:
            return
        # Apply truncation only when the raw output exceeds what fits
        # comfortably in chat (head 8 + tail 2 + ellipsis = 11 rows).
        if len(lines) > 12:
            truncated = truncate_with_tail(
                "\n".join(lines), head=8, tail=2, persist_label="tool",
            )
            display_lines = truncated.splitlines()
        else:
            display_lines = lines

        total = len(display_lines)
        if total == 1:
            line = Text()
            line.append("    ⎿ ", style=f"dim {C.primary}")
            line.append(display_lines[0][:120], style="dim")
            self.write(line)
            self._track_lines()
            return
        for i, line_text in enumerate(display_lines):
            line = Text()
            connector = "    │ " if i < total - 1 else "    ⎿ "
            line.append(connector, style=f"dim {C.primary}")
            # Ellipsis row from truncate_with_tail gets italic styling
            # so users can spot the truncation marker visually.
            if line_text.startswith("… +") and "more lines" in line_text:
                line.append(line_text[:140], style="dim italic")
            else:
                line.append(line_text[:140], style="dim")
            self.write(line)
            self._track_lines()

    def _render_tool_result_block(self, text: str, label: str,
                                   style_color: str = "") -> None:
        """Render one stream (stdout OR stderr) as a tree-connected block,
        with optional label. Used by the stdout/stderr split in
        _render_tool_result. Stderr blocks pass label='stderr' + a
        red color to differentiate from normal output.
        """
        from ..formatting import truncate_with_tail
        lines = text.strip().splitlines()
        if not lines:
            return
        # Truncate as usual
        if len(lines) > 12:
            truncated = truncate_with_tail(
                "\n".join(lines), head=8, tail=2, persist_label="tool",
            )
            display_lines = truncated.splitlines()
        else:
            display_lines = lines

        # Optional one-row label header — used to mark the stderr block
        if label:
            hdr = Text()
            hdr.append("    ⎿ ", style=f"dim {C.primary}")
            hdr.append(f"{label}:", style=f"bold {style_color or 'dim'}")
            self.write(hdr)
            self._track_lines()

        body_color = style_color or "dim"
        total = len(display_lines)
        for i, line_text in enumerate(display_lines):
            line = Text()
            # If we already wrote a label header, every line uses │
            # connector. Otherwise reserve ⎿ for the last line.
            if label:
                connector = "    │ "
            else:
                connector = "    │ " if i < total - 1 else "    ⎿ "
            line.append(connector, style=f"dim {C.primary}")
            if line_text.startswith("… +") and "more lines" in line_text:
                line.append(line_text[:140], style=f"dim italic {style_color}".strip())
            else:
                line.append(line_text[:140], style=body_color)
            self.write(line)
            self._track_lines()

    def _render_diff(self, diff_text: str) -> None:
        """Delegate to the extracted diff renderer in
        widgets/messages/diff.py. The implementation lives there so
        it can be tested in isolation and reused (e.g. for inline
        approval-prompt previews). This method is the dispatcher
        ChatLog uses; the heavy lifting is in render_diff().
        """
        from .messages.diff import render_diff
        render_diff(self, diff_text)

    def _render_info(self, text: str) -> None:
        # Pre-wrap so EVERY line of the info block keeps the 2-space
        # indent. Previously we wrote a single Text("  {text}") and
        # RichLog's natural wrap only indented the first line — wrapped
        # continuations sat flush to the left edge (visible on plan-mode
        # info text and other multi-line messages).
        import textwrap
        try:
            avail = max(20, self.size.width - 4)
        except Exception:
            avail = 76
        for paragraph in (text or "").split("\n"):
            if not paragraph.strip():
                self.write(Text("", style="dim"))
                self._track_lines()
                continue
            wrapped = textwrap.fill(
                paragraph, width=avail,
                initial_indent="  ", subsequent_indent="  ",
                break_long_words=False, break_on_hyphens=False,
            )
            for line in wrapped.split("\n"):
                self.write(Text(line, style="dim"))
                self._track_lines()

    def _render_error(self, text: str) -> None:
        # Render each paragraph (newline-separated) as ONE Text object
        # without manual pre-wrapping. Letting RichLog do the wrap with
        # its own width math avoids the double-wrap-truncation bug
        # where my textwrap.fill output (e.g. "...restart your Mac.")
        # got re-wrapped by RichLog and ended up clipped at the first
        # wrap boundary, hiding the continuation row.
        import textwrap
        try:
            # Account for ChatLog padding (1 col each side) + ✗ prefix
            # (4 cols) + a small safety margin so the line never
            # genuinely runs into the right edge.
            avail = max(20, self.size.width - 8)
        except Exception:
            avail = 72
        paragraphs = (text or "").split("\n")
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                self.write(Text(""))
                self._track_lines()
                continue
            # Pre-wrap is fine; the previous bug was avail being too
            # close to the actual visible width, so RichLog's own
            # second-pass wrap stripped the tail. Subtracting 8 above
            # gives RichLog enough headroom to NOT need a second wrap.
            initial = "  ✗ " if i == 0 else "    "
            wrapped = textwrap.fill(
                paragraph, width=avail,
                initial_indent=initial, subsequent_indent="    ",
                break_long_words=True,    # was False — ensures even no-space text breaks
                break_on_hyphens=True,    # was False — allows hyphen breaks
            )
            for j, line in enumerate(wrapped.split("\n")):
                styled = Text(no_wrap=True)   # block RichLog's second-pass wrap
                if i == 0 and j == 0:
                    styled.append("  ✗ ", style="bold red")
                    styled.append(line[4:], style="red")
                else:
                    styled.append(line, style="red")
                self.write(styled)
                self._track_lines()

    def _render_approval(self, tool_name: str, command: str) -> None:
        # Collapse multi-line commands (heredocs etc) to single line.
        # Don't pre-truncate — the no_wrap+ellipsis widget handles overflow at
        # the actual terminal width.
        cmd_oneline = command.replace("\n", " ↵ ").strip()
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("  Allow ", style=f"bold {C.warning}")
        line.append(f"{tool_name}", style=f"bold {C.warning}")
        line.append("? ", style=f"bold {C.warning}")
        line.append(cmd_oneline, style="dim")
        self.write(line)
        self._track_lines()
        # First token is what the "always allow" option will whitelist
        # for the rest of the session — e.g. "git", "pip", "python".
        first_token = (command.strip().split() or [""])[0][:20] or tool_name
        hint = Text()
        hint.append("  ", style="dim")
        hint.append("1", style="bold white")
        hint.append(" allow once  ", style="dim")
        hint.append("2", style="bold white")
        hint.append(f" always allow `{first_token}` (this session)  ", style="dim")
        hint.append("3", style="bold white")
        hint.append(" deny", style="dim")
        self.write(hint)
        self._track_lines()

    def _render_turn_summary(self, summary_text: str) -> None:
        line = Text()
        line.append("  ", style="")
        line.append(summary_text, style="dim")
        self.write(line)
        self._track_lines()


def _is_diff(text: str) -> bool:
    lines = text.strip().splitlines()[:10]
    diff_markers = sum(1 for l in lines if l.startswith(("+", "-", "@@", "diff ")))
    return diff_markers >= 3
