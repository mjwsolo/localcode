"""Scrollable chat message log — Claude Code style.

Messages displayed with clear visual hierarchy:
- User: ❯ bold prompt
- Assistant: indented markdown with syntax highlighting
- Tools: ● header with ⎿ tree connectors for results
- Info/errors: dim/red with consistent indentation
"""
from __future__ import annotations


from rich.markdown import Markdown
from rich.padding import Padding
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich.theme import Theme
from textual.strip import Strip
from textual.widgets import RichLog

# Custom theme: subtle, engineer-friendly
_CUSTOM_THEME = Theme({
    "markdown.code": "#d72b6b on #2a2a2a",
    "markdown.code_inline": "#d72b6b on #2a2a2a",
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
    "edit_file": ("Update", "updated"),
    "grep": ("Search", "searched"),
    "glob": ("Glob", "found"),
    "web_search": ("WebSearch", "searched web"),
    "web_fetch": ("WebFetch", "fetched"),
    "multi_edit": ("MultiEdit", "edited"),
}


class ChatLog(RichLog):
    """Scrollable message area styled like Claude Code."""

    ALLOW_SELECT = False  # We implement our own selection

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        overflow-y: scroll;
        overflow-x: hidden;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(wrap=True, **kwargs)
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

    def write(self, content, *args, **kwargs) -> "ChatLog":
        """Override to always constrain width to the visible area."""
        try:
            w = self.size.width - 2
            if w > 10 and "width" not in kwargs:
                kwargs["width"] = w
        except Exception:
            pass
        return super().write(content, *args, **kwargs)

    def on_mount(self) -> None:
        """Apply custom Rich theme for Slack-style inline code."""
        if hasattr(self, "_console"):
            self._console.push_theme(_CUSTOM_THEME)

    def on_mouse_down(self, event) -> None:
        """Start selection on mouse down."""
        self._sel_start = (event.x, event.y + int(self.scroll_offset.y))
        self._sel_end = self._sel_start
        self._selecting = True
        self._selected_text = ""
        self.capture_mouse()

    def on_mouse_move(self, event) -> None:
        """Extend selection while dragging."""
        if self._selecting:
            self._sel_end = (event.x, event.y + int(self.scroll_offset.y))
            self.refresh()  # redraw with updated highlight

    def on_mouse_up(self, event) -> None:
        """Finalize selection and extract text."""
        if not self._selecting:
            return
        self._selecting = False
        self.release_mouse()
        self._sel_end = (event.x, event.y + int(self.scroll_offset.y))

        # Check if it was a click (no drag) — handle thinking toggle
        if self._sel_start and self._sel_end:
            dx = abs(self._sel_end[0] - self._sel_start[0])
            dy = abs(self._sel_end[1] - self._sel_start[1])
            if dx <= 2 and dy <= 0:
                # Clean click — check for thinking toggle
                self._handle_thinking_click(self._sel_start[1])
                self._sel_start = None
                self._sel_end = None
                return

        # Extract selected text and set clipboard via OSC 52
        if self._sel_start and self._sel_end:
            self._selected_text = self._extract_selection()
            if self._selected_text.strip():
                self._set_clipboard_osc52(self._selected_text)

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

    def _handle_thinking_click(self, y: int) -> None:
        """Toggle thinking expand/collapse on click."""
        try:
            # Widen search range to account for scroll/render drift
            for check_y in range(max(0, y - 2), min(len(self.lines), y + 3)):
                if check_y in self._thinking_line_map:
                    hist_idx = self._thinking_line_map[check_y]
                    self._thinking_states[hist_idx] = not self._thinking_states.get(hist_idx, False)
                    self._rerender()
                    return
            # Fallback: check if the clicked line contains a thinking marker
            for check_y in range(max(0, y - 2), min(len(self.lines), y + 3)):
                if check_y < len(self.lines):
                    line_text = str(self.lines[check_y])
                    if "▶" in line_text or "▼" in line_text:
                        # Find the closest thinking entry in history
                        for idx in sorted(self._thinking_states.keys()):
                            self._thinking_states[idx] = not self._thinking_states[idx]
                            self._rerender()
                            return
                        # No states yet — find thinking entries in history
                        for idx, entry in enumerate(self._history):
                            if entry[0] == "thinking":
                                self._thinking_states[idx] = True
                                self._rerender()
                                return
        except Exception:
            pass

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
        self._history.append(("user", text))
        self._render_user(text)

    def append_queued(self, text: str) -> None:
        self._history.append(("queued", text))
        self._render_queued(text)

    def append_assistant(self, text: str) -> None:
        self._history.append(("assistant", text))
        self._render_assistant(text)

    _step_count: int = 0

    def reset_steps(self) -> None:
        self._step_count = 0

    def append_tool(self, name: str, args: str) -> None:
        self._history.append(("tool", name, args))
        self._render_tool(name, args)

    def append_tool_done(self, name: str, args: str, summary: str) -> None:
        self._history.append(("tool_done", name, args, summary))
        self._render_tool_done(name, args, summary)

    def append_tool_result(self, result: str, error: bool = False) -> None:
        self._history.append(("tool_result", result, "true" if error else "false"))
        self._render_tool_result(result, error)

    def append_thinking(self, text: str, expanded: bool = False) -> None:
        idx = len(self._history)
        self._history.append(("thinking", text))
        self._thinking_states[idx] = expanded
        self._render_thinking(text, expanded, idx)

    def append_approval(self, tool_name: str, command: str) -> None:
        self._history.append(("approval", tool_name, command))
        self._render_approval(tool_name, command)

    def append_info(self, text: str) -> None:
        self._history.append(("info", text))
        self._render_info(text)

    def append_error(self, text: str) -> None:
        self._history.append(("error", text))
        self._render_error(text)

    def append_turn_summary(self, elapsed: float, tools: list[str],
                            tokens_in: int = 0, tokens_out: int = 0,
                            cost_saved: float = 0.0) -> None:
        # Pre-format the summary text so we can replay it
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
            if tokens_out >= 1000:
                parts.append(f"{tokens_out / 1000:.1f}k tokens")
            else:
                parts.append(f"{tokens_out} tokens")
        if cost_saved > 0:
            parts.append(f"${cost_saved:.3f} saved")
        summary_text = " · ".join(parts)
        self._history.append(("turn_summary", summary_text))
        self._render_turn_summary(summary_text)

    def _stream_avail_width(self) -> int:
        """Available width for streamed text (matches _render_assistant)."""
        try:
            w = self.app.size.width - 6  # padding + scrollbar
        except Exception:
            w = 76
        return max(w, 40) - 2  # 2 for leading indent

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

    def stream_token(self, token: str) -> None:
        """Append a streaming token — renders incrementally line by line."""
        if not hasattr(self, '_stream_text'):
            self._stream_text = ""
            self._stream_started = False
        self._stream_text += token

        # Add initial spacer before first content
        if not self._stream_started:
            self.write(Text(""))
            self._track_lines()
            self._stream_started = True

        # Render complete lines as they come in
        # Keep the last (incomplete) line in the buffer
        lines = self._stream_text.split("\n")
        if len(lines) > 1:
            # We have at least one complete line — render all complete lines
            for complete_line in lines[:-1]:
                self._write_wrapped(complete_line)
            # Keep only the incomplete remainder
            self._stream_text = lines[-1]
            self.scroll_end(animate=False)

    def finish_stream(self, full_text: str = "") -> None:
        """Finalize streaming — render any remaining text and record in history."""
        if hasattr(self, '_stream_text'):
            # Render any remaining partial line
            if self._stream_text.strip():
                self._write_wrapped(self._stream_text)
            self._stream_text = ""
            self._stream_started = False
        # Record in history for re-rendering (text was already shown line by line)
        if full_text.strip():
            self._history.append(("assistant", full_text))
        self.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear display and history."""
        super().clear()
        if not hasattr(self, '_rerendering'):
            self._history.clear()
            self._thinking_states.clear()
            self._thinking_line_map.clear()
        self._line_counter = 0

    def _rerender(self) -> None:
        """Clear display and re-render from history (preserves history)."""
        self._rerendering = True
        # Save scroll position to restore after re-render
        saved_scroll = self.scroll_offset.y
        super().clear()
        self._thinking_line_map.clear()
        self._line_counter = 0
        for i, entry in enumerate(self._history):
            kind = entry[0]
            if kind == "spacer":
                self.write(Text(""))
                self._track_lines()
            elif kind == "user":
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
            elif kind == "raw":
                self.write(entry[1])
                self._track_lines()
        del self._rerendering
        # Restore scroll position — expanding thinking shouldn't jump to bottom
        self.scroll_to(y=saved_scroll, animate=False)

    # ── Render methods (write to display, no history recording) ──

    def _render_user(self, text: str) -> None:
        self.write(Text(""))
        self._track_lines()
        line = Text()
        line.append("❯ ", style="bold #5f87ff")
        line.append(text, style="bold white")
        self.write(line)
        self._track_lines()

    def _render_queued(self, text: str) -> None:
        self.write(Text(""))
        self._track_lines()
        line = Text()
        line.append("  ↻ ", style="dim yellow")
        line.append(text, style="dim italic")
        line.append("  (queued)", style="dim yellow")
        self.write(line)
        self._track_lines()

    def _render_assistant(self, text: str) -> None:
        self.write(Text(""))
        self._track_lines()
        try:
            avail_w = self.app.size.width - 6  # padding + scrollbar
        except Exception:
            avail_w = 76
        avail_w = max(avail_w, 40)

        has_md = any(m in text for m in ("```", "###", "**", "- ", "1. ", "`"))
        if has_md:
            from rich.console import Console
            from io import StringIO
            buf = StringIO()
            console = Console(file=buf, width=avail_w - 2, force_terminal=True, color_system="truecolor")
            console.print(Markdown(text, code_theme="monokai"))
            rendered = buf.getvalue()
            for rline in rendered.split("\n"):
                self.write(Text.from_ansi(f"  {rline}"))
                self._track_lines()
        else:
            import textwrap
            for line in text.split("\n"):
                wrapped = textwrap.fill(line, width=avail_w - 2) if line.strip() else ""
                for wline in (wrapped.split("\n") if wrapped else [""]):
                    self.write(Text(f"  {wline}"))
                    self._track_lines()

    def _render_thinking(self, text: str, expanded: bool, hist_idx: int) -> None:
        lines = text.strip().splitlines()
        if not lines:
            return
        self.write(Text(""))
        self._track_lines()
        # Record the line number for click detection
        header_line = getattr(self, '_line_counter', 0)
        self._thinking_line_map[header_line] = hist_idx
        header = Text()
        if expanded:
            header.append("  ▼ ", style="cyan")
            header.append(f"thinking ({len(lines)} lines)", style="dim italic cyan")
        else:
            header = Text(no_wrap=True, overflow="ellipsis")
            header.append("  ▶ ", style="cyan")
            try:
                max_preview = self.app.size.width - 28
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
            try:
                avail_w = self.app.size.width - 6
            except Exception:
                avail_w = 76
            avail_w = max(avail_w, 40)
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

    def _render_tool(self, name: str, args: str) -> None:
        self._step_count += 1
        header_info = _TOOL_HEADERS.get(name, (name, name))
        display_name = header_info[0]
        header = Text()
        header.append("  ● ", style="bold #5f87ff")
        header.append(f"{display_name}", style="bold #5f87ff")
        if args:
            args_short = args.strip().replace("\n", " ")[:60]
            header.append(f"({args_short})", style="dim")
        self.write(header)
        self._track_lines()

    def _render_tool_done(self, name: str, args: str, summary: str) -> None:
        header_info = _TOOL_HEADERS.get(name, (name, name))
        display_name = header_info[0]
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("  ✓ ", style="bold #32cd32")
        line.append(f"{display_name}", style="bold #32cd32")
        if args:
            args_short = args.strip().replace("\n", " ")[:40]
            line.append(f"({args_short})", style="#32cd32")
        if summary:
            line.append(f"  {summary[:60]}", style="dim #32cd32")
        self.write(line)
        self._track_lines()

    def _render_tool_result(self, result: str, error: bool) -> None:
        if error:
            line = Text()
            line.append("    ⎿ ", style="dim")
            line.append(result[:200], style="bold red")
            self.write(line)
            self._track_lines()
            return

        lines = result.strip().splitlines()
        if not lines:
            return

        if _is_diff(result):
            self._render_diff(result)
            return

        total = len(lines)
        if total == 1:
            line = Text()
            line.append("    ⎿ ", style="dim #5f87ff")
            line.append(lines[0][:120], style="dim")
            self.write(line)
            self._track_lines()
        else:
            max_lines = min(total, 6)
            for i, line_text in enumerate(lines[:max_lines]):
                line = Text()
                if i < max_lines - 1:
                    line.append("    │ ", style="dim #5f87ff")
                else:
                    line.append("    ⎿ ", style="dim #5f87ff")
                line.append(line_text[:120], style="dim")
                self.write(line)
                self._track_lines()
            if total > max_lines:
                more = Text()
                more.append(f"      … +{total - max_lines} more lines", style="dim italic")
                self.write(more)
                self._track_lines()

    def _render_diff(self, diff_text: str) -> None:
        lines = diff_text.strip().splitlines()
        added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))

        summary = Text()
        summary.append("    ⎿ ", style="dim #5f87ff")
        parts = []
        if added:
            parts.append(f"Added {added} line{'s' if added != 1 else ''}")
        if removed:
            parts.append(f"removed {removed} line{'s' if removed != 1 else ''}")
        summary.append(", ".join(parts) if parts else "no changes", style="dim")
        self.write(summary)
        self._track_lines()

        old_line = 0
        new_line = 0
        shown = 0
        for line_text in lines:
            if shown >= 4:
                more = Text()
                more.append(f"      … +{len(lines) - shown} more lines", style="dim italic")
                self.write(more)
                self._track_lines()
                break
            if line_text.startswith("---") or line_text.startswith("+++"):
                continue
            if line_text.startswith("@@"):
                import re as _re
                m = _re.search(r'-(\d+)', line_text)
                if m:
                    old_line = int(m.group(1))
                m2 = _re.search(r'\+(\d+)', line_text)
                if m2:
                    new_line = int(m2.group(1))
                continue

            dl = Text()
            if line_text.startswith("-"):
                dl.append(f"    {old_line:>4} ", style="dim red")
                dl.append("- ", style="bold red")
                dl.append(line_text[1:], style="red")
                old_line += 1
            elif line_text.startswith("+"):
                dl.append(f"    {new_line:>4} ", style="dim #32cd32")
                dl.append("+ ", style="bold #32cd32")
                dl.append(line_text[1:], style="#32cd32")
                new_line += 1
            else:
                # Skip blank context lines
                content = line_text[1:] if line_text.startswith(" ") else line_text
                if not content.strip():
                    old_line += 1
                    new_line += 1
                    continue
                dl.append(f"    {new_line:>4} ", style="dim")
                dl.append("  ", style="")
                dl.append(content, style="dim")
                old_line += 1
                new_line += 1

            self.write(dl)
            self._track_lines()
            shown += 1

    def _render_info(self, text: str) -> None:
        self.write(Text(""))
        self._track_lines()
        self.write(Text(f"  {text}", style="dim"))
        self._track_lines()

    def _render_error(self, text: str) -> None:
        line = Text()
        line.append("  ✗ ", style="bold red")
        line.append(text, style="red")
        self.write(line)
        self._track_lines()

    def _render_approval(self, tool_name: str, command: str) -> None:
        self.write(Text(""))
        self._track_lines()
        # Collapse multi-line commands (heredocs etc) to single line
        cmd_oneline = command.replace("\n", " ↵ ").strip()
        try:
            max_w = self.app.size.width - 12
        except Exception:
            max_w = 70
        max_w = max(max_w, 40)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("  Allow ", style="bold yellow")
        line.append(f"{tool_name}", style="bold yellow")
        line.append("? ", style="bold yellow")
        line.append(f"{cmd_oneline[:max_w]}", style="dim")
        self.write(line)
        self._track_lines()
        hint = Text()
        hint.append("  Press ", style="dim")
        hint.append("1", style="bold white")
        hint.append(" to allow, ", style="dim")
        hint.append("2", style="bold white")
        hint.append(" to deny", style="dim")
        self.write(hint)
        self._track_lines()

    def _render_turn_summary(self, summary_text: str) -> None:
        self.write(Text(""))
        self._track_lines()
        line = Text()
        line.append("  ", style="")
        line.append(summary_text, style="dim")
        self.write(line)
        self._track_lines()


def _is_diff(text: str) -> bool:
    lines = text.strip().splitlines()[:10]
    diff_markers = sum(1 for l in lines if l.startswith(("+", "-", "@@", "diff ")))
    return diff_markers >= 3
