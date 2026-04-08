"""Main chat screen — message log + input + status."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState

from ..bridge import AgentEvent, ApprovalRequest
from ..widgets.approval import ApprovalModal
from ..widgets.chat_log import ChatLog

if TYPE_CHECKING:
    from ..app import LocalCodeTUI


class ChatScreen(Screen):
    """Main chat interface with scrollable log, input, and status bar."""

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
    }
    #spinner-line {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $success;
        display: none;
    }
    #spinner-line.active {
        display: block;
    }
    #queue-line {
        height: 1;
        padding: 0 1;
        color: $warning;
        display: none;
    }
    #queue-line.active {
        display: block;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._agent_busy = False
        self._pending_messages: list[str] = []
        self._stream_buf: list[str] = []
        self._turn_start: float = 0
        self._tools_used: list[str] = []

    @property
    def tui(self) -> "LocalCodeTUI":
        return self.app  # type: ignore

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log", wrap=True, highlight=True, markup=True)
        yield Static("", id="spinner-line")
        yield Static("", id="queue-line")
        yield Input(placeholder="Type a message...", id="chat-input")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self._update_status()
        self.query_one("#chat-input", Input).focus()
        log = self.query_one("#chat-log", ChatLog)
        log.append_info("localcode ready. Type a message or /help for commands.")

    def _update_status(self) -> None:
        config = self.tui.gem_config
        mode = config.runtime.laptop_26b_runtime_mode
        mode_label = "fast" if not mode.endswith("-think") else "reasoning"
        model = config.runtime.model
        bar = self.query_one("#status-bar", Static)
        bar.update(f" {model} {mode_label}")

    def _update_spinner(self, text: str = "") -> None:
        spinner = self.query_one("#spinner-line", Static)
        if text:
            spinner.update(f" ◆ {text}")
            spinner.add_class("active")
        else:
            spinner.remove_class("active")

    def _update_queue(self) -> None:
        q = self.query_one("#queue-line", Static)
        if self._pending_messages:
            q.update(f" ↻ {len(self._pending_messages)} message(s) queued")
            q.add_class("active")
        else:
            q.remove_class("active")

    # ── Input handling ──

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        # Slash commands
        if text.startswith("/"):
            self._handle_command(text)
            return

        # Display user message
        log = self.query_one("#chat-log", ChatLog)
        log.append_user(text)

        if self._agent_busy:
            self._pending_messages.append(text)
            self._update_queue()
        else:
            self._start_turn(text)

    def _handle_command(self, text: str) -> None:
        log = self.query_one("#chat-log", ChatLog)
        if text == "/quit":
            self.app.exit()
        elif text == "/clear":
            log.clear()
            if self.tui.gem_app:
                self.tui.gem_app.session.messages.clear()
        elif text == "/switch":
            config = self.tui.gem_config
            mode = config.runtime.laptop_26b_runtime_mode
            if mode.endswith("-think"):
                config.runtime.laptop_26b_runtime_mode = "turbo"
                log.append_info("Switched to fast mode")
            else:
                config.runtime.laptop_26b_runtime_mode = "turbo-think"
                log.append_info("Switched to reasoning mode")
            self._update_status()
        elif text == "/undo":
            if self.tui.gem_app:
                result = self.tui.gem_app._handle_command("/undo")
                log.append_info("Reverted last change" if result else "Nothing to undo")
        elif text == "/help":
            log.append_info("/switch  toggle fast/reasoning mode")
            log.append_info("/undo    revert last file change")
            log.append_info("/clear   clear conversation")
            log.append_info("/quit    exit")
        else:
            log.append_info(f"Unknown command: {text}")

    # ── Agent turn ──

    def _start_turn(self, text: str) -> None:
        if not self.tui.ensure_backend():
            log = self.query_one("#chat-log", ChatLog)
            log.append_error("Backend not ready. Is the server running?")
            return
        self._agent_busy = True
        self._stream_buf.clear()
        self._tools_used.clear()
        self._turn_start = time.time()
        self._update_spinner("thinking...")
        self.run_agent_turn(text)

    @work(exclusive=True, thread=True)
    def run_agent_turn(self, user_text: str) -> None:
        """Run model inference on background thread. Streams directly to TUI."""
        if not self.tui.gem_app:
            return
        app = self.tui.gem_app
        bridge = self.tui.bridge
        try:
            # Stream directly from the runtime — bypass OutputManager's stdout
            from ..bridge import AgentEvent

            # Add user message to session
            app.session.messages.append({"role": "user", "content": user_text})

            # Build messages for the model
            from ...composer import compose_messages
            from ...context_manager import build_context
            context = build_context(app.repo_root, app.session, app.config, app.toolkit)
            from ...prompts import build_system_prompt
            system = build_system_prompt(app.profile, context)
            composed = compose_messages(
                app.profile, system, context,
                app.session.messages, user_text,
            )

            # Stream from engine
            full_response = []
            for event in app.engine.stream_chat_events(composed):
                if event["type"] == "content":
                    chunk = str(event["content"])
                    full_response.append(chunk)
                    bridge.on_event("content", chunk=chunk)
                elif event["type"] == "thinking":
                    chunk = str(event["content"])
                    bridge.on_event("thinking_peek", text=chunk[:120])

            # Save response to session
            text = "".join(full_response).strip()
            if text:
                app.session.messages.append({"role": "assistant", "content": text})
                bridge.on_event("response_done", text=text)

        except Exception as e:
            bridge.on_event("error", message=str(e))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "run_agent_turn" and event.state == WorkerState.SUCCESS:
            self._on_turn_done()
        elif event.state == WorkerState.ERROR:
            log = self.query_one("#chat-log", ChatLog)
            log.append_error(f"Agent error: {event.worker.error}")
            self._on_turn_done()

    def _on_turn_done(self) -> None:
        self._agent_busy = False
        self._update_spinner("")

        # Flush stream buffer
        log = self.query_one("#chat-log", ChatLog)
        if self._stream_buf:
            text = "".join(self._stream_buf)
            if text.strip():
                log.append_assistant(text)
            self._stream_buf.clear()

        # Show done summary
        elapsed = time.time() - self._turn_start
        parts = [f"{elapsed:.1f}s"]
        if self._tools_used:
            from collections import Counter
            counts = Counter(self._tools_used)
            tool_str = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in counts.items())
            parts.append(f"tools: {tool_str}")
        log.append_info(f"Done in {' — '.join(parts)}")

        # Auto-submit queued messages
        if self._pending_messages:
            next_msg = self._pending_messages.pop(0)
            self._update_queue()
            log.append_user(next_msg)
            self._start_turn(next_msg)

    # ── Agent events (from bridge) ──

    def on_agent_event(self, event: AgentEvent) -> None:
        log = self.query_one("#chat-log", ChatLog)
        t = event.event_type
        p = event.payload

        if t == "content":
            chunk = p.get("chunk", "")
            self._stream_buf.append(chunk)
            # Show streaming text immediately
            self._update_spinner(f"generating... ({len(self._stream_buf)} chunks)")
        elif t == "response_done":
            text = p.get("text", "")
            if text:
                log.append_assistant(text)
                self._stream_buf.clear()
        elif t == "tool_start":
            name = p.get("name", "")
            args = p.get("args", "")
            self._tools_used.append(name)
            log.append_tool(name, args)
            self._update_spinner(f"{name}...")
        elif t == "tool_result":
            result = p.get("result", "")
            error = p.get("error", False)
            log.append_tool_result(result, error=bool(error))
        elif t == "thinking_start":
            self._update_spinner("thinking...")
        elif t == "thinking_peek":
            text = p.get("text", "")
            if text:
                self._update_spinner(f"thinking: {text[:60]}...")
        elif t == "stream_start":
            self._update_spinner("")
        elif t == "error":
            msg = p.get("message", "Unknown error")
            log.append_error(msg)
        elif t == "stage":
            stage = p.get("stage", "")
            if stage:
                self._update_spinner(stage)

    # ── Tool approval ──

    def on_approval_request(self, event: ApprovalRequest) -> None:
        """Show approval modal, set result on bridge."""

        def _on_result(approved: bool) -> None:
            self.tui.bridge.set_approval(approved)
            log = self.query_one("#chat-log", ChatLog)
            if approved:
                log.append_info("└ approved")
            else:
                log.append_info("└ denied")

        self.app.push_screen(
            ApprovalModal(event.tool_name, event.command),
            callback=_on_result,
        )
