"""Bridge between OutputManager events and Textual messages.

The existing OutputManager fires events via set_event_callback().
This bridge receives those events (on the worker thread) and posts
Textual messages to update the UI (thread-safe).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
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
        """
        if payload is None:
            payload = kwargs
        elif isinstance(payload, dict):
            payload.update(kwargs)
        try:
            self.tui_app.post_message(AgentEvent(event_type, payload))
        except Exception:
            pass  # app might be shutting down

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
