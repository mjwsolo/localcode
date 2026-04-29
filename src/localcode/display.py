"""Minimal stubs for legacy CLI display classes.

The full Rich-based CLI display layer was retired when LocalCode moved
to a Textual TUI. The TUI doesn't render through these classes, but
the backend engine (`LocalCodeApp`) still constructs `SessionStats` /
`ToolCallDisplay` / `ResponseDisplay` in code paths shared with the
old CLI. Rather than tear those construction sites out (large
refactor across `app.py`), keep no-op stubs here so instantiation
succeeds and the TUI continues to drive the UI directly.
"""
from __future__ import annotations


class SessionStats:
    """Per-session counters. CLI renderer gone; only attribute access matters."""

    def __init__(self) -> None:
        self.tools_called = 0
        self.tool_errors = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.start_time = 0.0

    def record(self, *_args, **_kwargs) -> None:
        return


class ToolCallDisplay:
    """Legacy CLI tool-call renderer. TUI handles this through OutputManager."""

    def __init__(self) -> None:
        pass

    def show(self, *_args, **_kwargs) -> None:
        return

    def result(self, *_args, **_kwargs) -> None:
        return


class ResponseDisplay:
    """Legacy CLI response renderer. TUI handles this through OutputManager."""

    def __init__(self) -> None:
        pass

    def show(self, *_args, **_kwargs) -> None:
        return

    @staticmethod
    def print_info(*_args, **_kwargs) -> None:
        return


class ThinkingIndicator:
    """Type-hint target only — never instantiated. Stub keeps imports happy."""

    pass
