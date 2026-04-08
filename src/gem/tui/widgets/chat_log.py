"""Scrollable chat message log using RichLog."""
from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog


class ChatLog(RichLog):
    """Scrollable message area. Auto-scrolls to bottom."""

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        overflow-y: scroll;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    def append_user(self, text: str) -> None:
        """Display a user message."""
        self.write(Text(f"› {text}", style="bold"))
        self.write(Text(""))  # spacer

    def append_assistant(self, text: str) -> None:
        """Display an assistant response (may contain markdown)."""
        has_md = any(m in text for m in ("```", "###", "**", "- ", "1. ", "`"))
        if has_md:
            self.write(Markdown(text))
        else:
            self.write(Text(f"  {text}"))
        self.write(Text(""))  # spacer

    def append_tool(self, name: str, args: str) -> None:
        """Display a tool call."""
        headers = {
            "bash": "Ran", "read_file": "Read", "write_file": "Wrote",
            "edit_file": "Edited", "grep": "Searched", "glob": "Found",
            "web_search": "Searched web",
        }
        header = headers.get(name, name)
        self.write(Text(f"▪ {header} {args}", style="bold cyan"))

    def append_tool_result(self, result: str, error: bool = False) -> None:
        """Display tool result."""
        style = "red" if error else "dim"
        self.write(Text(f"  └ {result[:200]}", style=style))

    def append_thinking(self, text: str) -> None:
        """Display thinking summary."""
        self.write(Text(f"  thought: {text[:120]}…", style="dim italic"))

    def append_info(self, text: str) -> None:
        """Display informational text."""
        self.write(Text(f"  {text}", style="dim"))

    def append_error(self, text: str) -> None:
        """Display error text."""
        self.write(Text(f"  Error: {text}", style="bold red"))

    def stream_token(self, token: str) -> None:
        """Append a streaming token to the last line."""
        # RichLog doesn't support inline append, so we use write with end=""
        # For now, accumulate and re-render
        if not hasattr(self, '_stream_buf'):
            self._stream_buf = []
        self._stream_buf.append(token)
        # Re-render the streaming line every few tokens
        if token.endswith(("\n", ".", "!", "?")) or len(self._stream_buf) % 10 == 0:
            text = "".join(self._stream_buf)
            # Remove the last written line and replace
            if hasattr(self, '_stream_line_written') and self._stream_line_written:
                try:
                    self.clear()  # TODO: better approach - only clear last line
                except Exception:
                    pass
            self.write(Text(f"  {text}"))
            self._stream_line_written = True

    def finish_stream(self) -> None:
        """Finalize streaming — render the complete response."""
        if hasattr(self, '_stream_buf') and self._stream_buf:
            text = "".join(self._stream_buf)
            self.append_assistant(text)
            self._stream_buf.clear()
            self._stream_line_written = False
