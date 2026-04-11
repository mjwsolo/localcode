"""Bridge between OutputManager events and Textual messages.

The existing OutputManager fires events via set_event_callback().
This bridge receives those events (on the worker thread) and posts
Textual messages to update the UI (thread-safe).
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from textual.message import Message

if TYPE_CHECKING:
    from .app import LocalCodeTUI


class AgentEvent(Message):
    """Event from the agent loop, posted to Textual app."""

    def __init__(self, event_type: str, payload: dict[str, Any]) -> None:
        super().__init__()
        self.event_type = event_type
        self.payload = payload


class ApprovalRequest(Message):
    """Agent needs user approval for a destructive command."""

    def __init__(self, tool_name: str, command: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.command = command


class TUIBridge:
    """Adapts OutputManager → Textual messages."""

    def __init__(self, tui_app: "LocalCodeTUI") -> None:
        self.tui_app = tui_app
        # For tool approval: worker blocks on this event
        self._approval_event = threading.Event()
        self._approval_result: bool = False

    def on_event(self, event_type: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Called by OutputManager._emit_event from worker thread.

        OutputManager calls: callback(event_type, payload_dict)
        Direct calls use: on_event(event_type, key=val, ...)

        Streaming events (content, thinking) bypass the Textual message queue
        and directly update the widget via call_from_thread for real-time display.
        Other events use post_message.
        """
        if payload is None:
            payload = kwargs
        elif isinstance(payload, dict):
            payload.update(kwargs)
        try:
            # For content streaming, bypass message queue — directly invoke
            # the widget update on the UI thread for real-time line-by-line display
            if event_type == "content":
                chunk = payload.get("chunk", "")
                if chunk:
                    self.tui_app.call_from_thread(self._direct_stream_content, chunk)
                return
            msg = AgentEvent(event_type, payload)
            if event_type in ("thinking_start", "thinking_chunk",
                              "thinking_done", "stream_start", "thinking_peek"):
                self.tui_app.call_from_thread(self.tui_app.post_message, msg)
            else:
                self.tui_app.post_message(msg)
        except Exception:
            pass  # app might be shutting down

    def _direct_stream_content(self, chunk: str) -> None:
        """Directly update chat log with streaming content on UI thread.

        Called via call_from_thread — runs on the Textual event loop,
        bypassing the message queue for immediate visual updates.
        """
        try:
            screen = self.tui_app.screen
            # Update stream buffer on the screen for history tracking
            if hasattr(screen, '_stream_buf'):
                screen._stream_buf.append(chunk)
            if hasattr(screen, '_thinking_phase'):
                screen._thinking_phase = "generating"
            # Track token counts
            toks = max(1, len(chunk) // 4)
            if hasattr(screen, '_turn_tokens'):
                screen._turn_tokens += toks
            if hasattr(screen, '_context_used'):
                screen._context_used += toks
            # Hide animation on first content
            if hasattr(screen, '_active_mode') and screen._active_mode:
                if hasattr(screen, '_hide_active_step'):
                    screen._hide_active_step()
            # Directly stream to chat log widget
            log = screen.query_one("#chat-log")
            log.stream_token(chunk)
        except Exception:
            pass

    def request_approval(self, tool_name: str, command: str) -> bool:
        """Block the worker thread until user approves/denies. Returns True if approved."""
        self._approval_event.clear()
        self._approval_result = False
        self.tui_app.post_message(ApprovalRequest(tool_name, command))
        self._approval_event.wait()  # blocks worker thread
        return self._approval_result

    def set_approval(self, approved: bool) -> None:
        """Called from the Textual UI thread when user clicks Allow/Deny."""
        self._approval_result = approved
        self._approval_event.set()  # unblocks worker thread
