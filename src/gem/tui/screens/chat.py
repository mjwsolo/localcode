"""Main chat screen — Claude Code style layout.

Layout (top to bottom):
  ──────────── LocalCode ──────────────
  [scrollable chat log]
  ◆ mining... (3s · ↓ 200 tokens · $0.02 saved)
  [input field]
  model · mode · 5% context
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import TYPE_CHECKING

from rich.text import Text as RichText

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState

from ..bridge import AgentEvent, ApprovalRequest
from ..widgets.chat_log import ChatLog

if TYPE_CHECKING:
    from ..app import LocalCodeTUI

# Our original thinking icons (green, cycling) from display.py
_THINK_ICONS = ["·", "▲", "◆"]
_THINK_LABELS = [
    "mining", "cutting facets", "polishing",
    "examining", "shaping", "refining",
    "digging deeper", "crystallizing", "forging",
]

# Thinking phase labels from display.py
_THINK_LABELS = [
    "mining", "cutting facets", "polishing",
    "examining", "shaping", "refining",
    "digging deeper", "crystallizing", "forging",
]

_CLOUD_COST_PER_TOKEN = 0.000015

_TOOL_CALL_RE = re.compile(
    r'<\|?tool_call\|?>.*?<\|?/?tool_call\|?>', re.DOTALL
)


def _clean_display_text(text: str) -> str:
    """Strip tool call tokens and artifacts from text before display."""
    text = _TOOL_CALL_RE.sub("", text)
    text = text.replace("<unused25>", "")
    text = re.sub(r"<\|channel>thought\n?", "", text)
    text = re.sub(r"<channel\|>\n?", "", text)
    return text.strip()


def _is_diff_result(text: str) -> bool:
    """Check if tool result contains a diff."""
    lines = text.strip().splitlines()[:10]
    return sum(1 for l in lines if l.startswith(("+", "-", "@@", "diff "))) >= 3


class ChatScreen(Screen):
    """Main chat interface — Claude Code inspired, with full agent loop."""

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
    }
    #header-bar {
        dock: top;
        height: 1;
        padding: 0 1;
        color: #5f87ff;
        background: $surface;
    }
    #active-step {
        height: 1;
        padding: 0 1;
        margin: 1 0 0 0;
        color: #5f87ff;
        display: none;
    }
    #active-step.active {
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
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._agent_busy = False
        self._pending_messages: list[str] = []
        self._stream_buf: list[str] = []
        self._turn_start: float = 0
        self._tools_used: list[str] = []
        self._tick_count: int = 0
        self._spin_timer = None
        self._context_used: int = 0
        self._context_max: int = 32768
        self._turn_tokens: int = 0
        self._total_tokens: int = 0
        self._thinking_phase: str = ""
        self._response_shown: bool = False
        self._active_step_text: str = ""  # raw text for scanning animation
        self._active_tool_name: str = ""
        self._active_tool_args: str = ""
        self._scan_pos: int = 0
        self._step_timer = None
        self._thinking_text: str = ""  # full thinking from last turn
        self._thinking_expanded: bool = False  # toggle state

    @property
    def tui(self) -> "LocalCodeTUI":
        return self.app  # type: ignore

    def compose(self) -> ComposeResult:
        yield Static("", id="header-bar")
        yield ChatLog(id="chat-log", wrap=True, highlight=True, markup=True)
        yield Static("", id="active-step")
        yield Static("", id="queue-line")
        yield Input(placeholder="Type a message...", id="chat-input")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self._update_header()
        self._update_status()
        self.query_one("#chat-input", Input).focus()
        self.query_one("#chat-log", ChatLog)

    # ── Header bar ──

    def _update_header(self) -> None:
        try:
            width = self.app.size.width or 80
        except Exception:
            width = 80
        usable = width - 2

        left = "🏠 LocalCode"
        left_cols = 13
        line = f"{left} {'─' * (usable - left_cols)}"

        header = self.query_one("#header-bar", Static)
        header.update(line)

    def _show_thinking(self) -> None:
        self._turn_tokens = 0
        self._thinking_phase = ""
        self._response_shown = False
        if self._spin_timer is None:
            self._spin_timer = self.set_interval(0.15, self._tick_thinking)
        self._update_header()

    def _hide_thinking(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
            self._tick_count = 0
        self._update_header()

    def _tick_thinking(self) -> None:
        self._tick_count += 1
        self._update_header()

    # ── Active step (scanning highlight animation) ──

    _active_mode: str = ""  # "tool" or "thinking"

    # Map tool names to present-tense verbs for the live status
    _TOOL_VERBS = {
        "bash": "running",
        "read_file": "reading",
        "write_file": "writing",
        "edit_file": "editing",
        "grep": "searching",
        "glob": "searching",
        "list_files": "browsing files",
        "web_search": "searching the web",
        "code_search": "searching code",
    }

    def _show_active_step(self, name: str, args: str) -> None:
        """Show in-progress tool as a live verb status."""
        verb = self._TOOL_VERBS.get(name, f"running {name}")
        self._active_step_text = verb
        self._active_tool_name = name
        self._active_tool_args = args
        self._active_mode = "tool"
        self._scan_pos = 0
        w = self.query_one("#active-step", Static)
        w.add_class("active")
        if self._step_timer is not None:
            self._step_timer.stop()
        self._step_timer = self.set_interval(0.05, self._tick_active)
        self._tick_active()

    def _show_active_thinking(self, text: str = "thinking") -> None:
        """Show in-progress thinking status."""
        self._active_step_text = text
        self._active_tool_name = "thinking"
        self._active_tool_args = ""
        self._active_mode = "thinking"
        self._scan_pos = 0
        self._think_tick = 0
        w = self.query_one("#active-step", Static)
        w.add_class("active")
        if self._step_timer is not None:
            self._step_timer.stop()
        self._step_timer = self.set_interval(0.05, self._tick_active)

    def _hide_active_step(self) -> None:
        """Hide the active step animation."""
        w = self.query_one("#active-step", Static)
        w.remove_class("active")
        if self._step_timer is not None:
            self._step_timer.stop()
            self._step_timer = None
        self._active_step_text = ""
        self._active_mode = ""
        self._scan_pos = 0

    def _elapsed_str(self) -> str:
        """Format elapsed time since turn start."""
        elapsed = time.time() - self._turn_start if self._turn_start else 0
        if elapsed < 60:
            return f"({elapsed:.0f}s)"
        m, s = divmod(int(elapsed), 60)
        return f"({m}m{s:02d}s)"

    def _tick_active(self) -> None:
        """Single animation tick for both tools and thinking."""
        text = self._active_step_text
        if not text:
            return

        timer = self._elapsed_str()

        # Unified live status: "◆ thinking... (3s)" or "◆ searching... (12s)"
        label = f"◆ {text}..."
        self._scan_pos = (self._scan_pos + 1) % max(len(label), 1)
        rt = RichText()
        rt.append("  ", style="")
        for i, ch in enumerate(label):
            if i <= self._scan_pos:
                rt.append(ch, style="bold dim")
            else:
                rt.append(ch, style="dim italic")
        rt.append(f"  {timer}", style="dim")

        w = self.query_one("#active-step", Static)
        w.update(rt)

    # ── Status bar (bottom — model, mode, context remaining) ──

    def _update_status(self) -> None:
        self._update_header()
        config = self.tui.gem_config
        mode = config.runtime.laptop_26b_runtime_mode
        mode_label = "fast" if not mode.endswith("-think") else "reasoning"
        model = config.runtime.model
        # Context REMAINING (starts at 100%, decreases)
        pct_remaining = max(0, 100 - int(self._context_used / max(1, self._context_max) * 100))
        bar = self.query_one("#status-bar", Static)
        bar.update(f" model: {model}  ·  mode: {mode_label}  ·  {pct_remaining}% context remaining")

    def _update_queue(self) -> None:
        q = self.query_one("#queue-line", Static)
        if self._pending_messages:
            n = len(self._pending_messages)
            preview = self._pending_messages[0][:40]
            q.update(f" ↻ {n} queued: \"{preview}\"{'…' if len(self._pending_messages[0]) > 40 else ''}")
            q.add_class("active")
        else:
            q.remove_class("active")

    # ── Input handling ──

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            self._handle_command(text)
            return

        log = self.query_one("#chat-log", ChatLog)

        if self._agent_busy:
            self._pending_messages.append(text)
            self._update_queue()
        else:
            log.append_user(text)
            self._start_turn(text)

    def _handle_command(self, text: str) -> None:
        log = self.query_one("#chat-log", ChatLog)
        if text not in ("/quit", "/clear"):
            log.write(RichText(""))  # spacing before command output
        if text == "/quit":
            self.app.exit()
        elif text == "/clear":
            log.clear()
            if self.tui.gem_app:
                self.tui.gem_app.session.messages.clear()
            self._context_used = 0
            self._total_tokens = 0
            self._update_status()
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
        elif text == "/copy":
            # Copy last assistant response to clipboard
            last_text = ""
            for entry in reversed(self.query_one("#chat-log", ChatLog)._history):
                if entry[0] == "assistant":
                    last_text = entry[1]
                    break
            if last_text:
                try:
                    import subprocess
                    subprocess.run(["pbcopy"], input=last_text.encode(), check=True)
                    log.append_info("Copied to clipboard")
                except Exception:
                    log.append_error("Failed to copy")
            else:
                log.append_info("Nothing to copy")
        elif text == "/help":
            log.append_info("/switch  toggle fast/reasoning mode")
            log.append_info("/copy    copy last response to clipboard")
            log.append_info("/undo    revert last file change")
            log.append_info("/clear   clear conversation")
            log.append_info("/quit    exit")
        else:
            log.append_info(f"Unknown command: {text}")

    # ── Agent turn (uses full agent loop with tool execution) ──

    def _start_turn(self, text: str) -> None:
        if not self.tui.ensure_backend():
            log = self.query_one("#chat-log", ChatLog)
            log.append_error("Backend not ready. Is the server running?")
            return
        self._agent_busy = True
        self._stream_buf.clear()
        self._tools_used.clear()
        self._turn_start = time.time()
        self._context_used += max(1, len(text) // 4)
        log = self.query_one("#chat-log", ChatLog)
        log.reset_steps()
        self._show_thinking()
        self._update_status()
        self.run_agent_turn(text)

    @work(exclusive=True, thread=True)
    def run_agent_turn(self, user_text: str) -> None:
        """Run FULL agent loop on background thread.

        Uses GemApp.ask() which handles:
        - Context gathering (repo structure, retrieval, cartridge)
        - System prompt composition
        - Full agent loop with tool execution (read, write, bash, grep, etc.)
        - Multi-round tool calls until model is done
        - All events emitted via OutputManager → bridge → TUI
        """
        if not self.tui.gem_app:
            return
        app = self.tui.gem_app
        bridge = self.tui.bridge

        # Suppress stdout/stderr from OutputManager (Textual owns the terminal)
        # Events flow through the bridge callback set on OutputManager
        devnull = open(os.devnull, "w")
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Ensure the event callback is set (re-set every turn for safety)
        app.out.set_event_callback(bridge.on_event)

        try:
            sys.stdout = devnull
            sys.stderr = devnull

            # Stop the indicator thread before ask() starts a new one
            app.out._stop_indicator()

            # Use the full ask() method — handles everything
            assistant_text = app.ask(user_text, stream=True)

            if assistant_text:
                bridge.on_event("response_done", text=assistant_text)
            else:
                bridge.on_event("response_done", text="")
        except Exception as e:
            bridge.on_event("error", message=str(e))
        finally:
            # Stop any indicator threads before restoring stdout
            app.out._indicator_running = False
            app.out._stop_indicator()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "run_agent_turn" and event.state == WorkerState.SUCCESS:
            self._on_turn_done()
        elif event.state == WorkerState.ERROR:
            log = self.query_one("#chat-log", ChatLog)
            log.append_error(f"Agent error: {event.worker.error}")
            self._on_turn_done()

    def _on_turn_done(self) -> None:
        self._agent_busy = False
        self._hide_thinking()
        self._hide_active_step()

        log = self.query_one("#chat-log", ChatLog)

        # Only show response if response_done event didn't already handle it
        if not self._response_shown and self._stream_buf:
            text = "".join(self._stream_buf)
            text = _clean_display_text(text)
            if text:
                log.append_assistant(text)

        self._stream_buf.clear()

        # Turn summary
        elapsed = time.time() - self._turn_start
        cost = self._turn_tokens * _CLOUD_COST_PER_TOKEN
        log.append_turn_summary(
            elapsed, self._tools_used,
            tokens_out=self._turn_tokens,
            cost_saved=cost,
        )
        self._total_tokens += self._turn_tokens
        # Estimate total context from session messages
        if self.tui.gem_app:
            total_chars = sum(len(str(m.get("content", ""))) for m in self.tui.gem_app.session.messages)
            self._context_used = total_chars // 4  # ~4 chars per token
        self._update_status()

        # Auto-submit queued messages
        if self._pending_messages:
            next_msg = self._pending_messages.pop(0)
            self._update_queue()
            log.append_user(next_msg)
            self._start_turn(next_msg)

    # ── Agent events (from bridge via OutputManager) ──

    def on_agent_event(self, event: AgentEvent) -> None:
        log = self.query_one("#chat-log", ChatLog)
        t = event.event_type
        p = event.payload

        if t == "content":
            # Hide animation once content starts flowing
            if self._active_mode:
                self._hide_active_step()
            chunk = p.get("chunk", "")
            self._stream_buf.append(chunk)
            toks = max(1, len(chunk) // 4)
            self._turn_tokens += toks
            self._context_used += toks
            self._thinking_phase = "generating"
        elif t == "response_done":
            text = p.get("text", "")
            text = _clean_display_text(text)
            if text:
                log.append_assistant(text)
                self._response_shown = True
            self._stream_buf.clear()
        elif t == "tool_start":
            name = p.get("name", "")
            args = p.get("args", "")
            # Add spacing before first tool in a turn
            if not self._tools_used:
                log.write(RichText(""))
            self._tools_used.append(name)
            self._thinking_phase = name
            # Only show in floating #active-step widget (NOT in chat log)
            self._show_active_step(name, args)
        elif t == "tool_result":
            result = p.get("result", "")
            error = p.get("error", "")
            is_error = error == "true" or error is True
            # Hide floating animation
            self._hide_active_step()
            name = self._active_tool_name
            args = self._active_tool_args
            # Write completed ✓ line to chat log
            if is_error:
                log.append_tool(name, args)
                log.append_tool_result(result, error=True)
            else:
                lines = result.strip().splitlines()
                is_diff = len(lines) > 1 and _is_diff_result(result)
                if is_diff:
                    # For diffs, extract file path from --- line as summary
                    file_path = ""
                    for l in lines[:5]:
                        if l.startswith("--- ") or l.startswith("+++ "):
                            file_path = l.split("\t")[0][4:]  # strip --- /+++ prefix
                            break
                    log.append_tool_done(name, args, f"--- {file_path}" if file_path else "")
                    log.append_tool_result(result)
                else:
                    summary = lines[0][:80] if lines else ""
                    log.append_tool_done(name, args, summary)
            self._thinking_phase = ""
            # Show thinking indicator immediately after tool completion
            # to cover the gap while the model processes the result
            self._show_active_thinking("thinking")
        elif t == "thinking_start":
            self._thinking_phase = "thinking"
            self._thinking_text = ""
            self._show_active_thinking("thinking")
        elif t == "thinking_chunk":
            chunk = p.get("chunk", "")
            self._thinking_text += chunk
            self._thinking_phase = "thinking"
        elif t == "thinking_peek":
            self._thinking_phase = "thinking"
        elif t == "thinking_done":
            text = p.get("text", "")
            self._thinking_text = text
            self._hide_active_step()
            if text.strip():
                log.append_thinking(text, expanded=self._thinking_expanded)
        elif t == "stream_start":
            self._thinking_phase = "generating"
            # Show gem-themed animation while generating response
            self._show_active_thinking("generating")
        elif t == "error":
            msg = p.get("message", "Unknown error")
            log.append_error(msg)
        elif t == "stage":
            stage = p.get("stage", "")
            if stage:
                self._thinking_phase = stage
                # Show animation for stage changes (between tool rounds)
                if not self._active_mode:
                    self._show_active_thinking(stage)
        elif t == "done":
            pass  # handled by on_worker_state_changed

    # ── Tool approval (inline, like Claude Code) ──

    _awaiting_approval: bool = False

    def on_approval_request(self, event: ApprovalRequest) -> None:
        """Show approval inline in chat log — press 1 to allow, 2 to deny."""
        log = self.query_one("#chat-log", ChatLog)
        log.append_approval(event.tool_name, event.command)
        self._awaiting_approval = True
        # Disable input and remove focus so keys go to screen
        inp = self.query_one("#chat-input", Input)
        inp.disabled = True
        self.focus()  # focus the screen itself to capture keys

    def on_key(self, event) -> None:
        """Capture 1/2/y/n keys for inline approval. Block all other input."""
        if not self._awaiting_approval:
            return
        key = event.key
        log = self.query_one("#chat-log", ChatLog)
        if key in ("1", "y"):
            self._awaiting_approval = False
            log.append_info("  └ approved")
            self.tui.bridge.set_approval(True)
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()
        elif key in ("2", "n", "escape"):
            self._awaiting_approval = False
            log.append_info("  └ denied")
            self.tui.bridge.set_approval(False)
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()
        # Block ALL other keys during approval
        event.prevent_default()
        event.stop()
