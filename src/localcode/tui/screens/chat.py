"""Main chat screen — terminal coding tools style layout.

Layout (top to bottom):
  ──────────── LocalCode ──────────────
  [scrollable chat log]
  ◆ mining... (3s · ↓ 200 tokens)
  [input field]
  model · mode · 5% context
"""
from __future__ import annotations

from ...theme import C


import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text as RichText

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker, WorkerState


class _NoTintInput(Input):
    """Input that doesn't highlight on focus.

    Also overrides paste handling: Textual's stock `Input._on_paste`
    takes only `event.text.splitlines()[0]` and calls `event.stop()`,
    so a multi-line paste (e.g. a JSON blob) loses everything after
    the first newline AND blocks our screen-level `on_paste` from
    seeing it. We override here to join all lines with spaces so the
    full paste content makes it into the message the user submits.
    """
    DEFAULT_CSS = """
    _NoTintInput {
        background: $surface;
        &:focus {
            background-tint: transparent 0%;
            background: $surface;
        }
        /* Override Textual's global `*:disabled:can-focus { opacity: 0.7; }`
           rule (textual.app.App.DEFAULT_CSS). Without this, the chat input
           visibly dims to grey-on-black during the approval prompt — which
           reads as "the input broke" instead of "input is paused while you
           pick a verdict". opacity:1 keeps full brightness, background
           pinned to ansi_default so the row stays terminal-coloured. */
        &:disabled, &:disabled:can-focus {
            opacity: 1;
            background: ansi_default;
            background-tint: transparent 0%;
        }
    }
    """

    def _on_paste(self, event) -> None:  # type: ignore[override]
        # Textual paste has both propagation and default-action phases.
        # Stopping propagation alone is not enough on some terminal/Textual
        # combinations: our custom insertion runs, then Input's default paste
        # action inserts the same text again. Cancel default first so this
        # handler is the single source of truth.
        prevent_default = getattr(event, "prevent_default", None)
        if callable(prevent_default):
            prevent_default()
        text = getattr(event, "text", "") or ""
        if not text:
            event.stop()
            return
        # Collapse to a single line so Input's single-line nature
        # doesn't truncate at the first newline. Whitespace runs are
        # squeezed via " ".join(splitlines()) so pasted JSON / code
        # arrives readable rather than as a wall of double-spaces.
        if "\n" in text or "\r" in text:
            joined = " ".join(line for line in text.splitlines() if line is not None)
        else:
            joined = text
        selection = self.selection
        if selection.is_empty:
            self.insert_text_at_cursor(joined)
        else:
            self.replace(joined, *selection)
        event.stop()

    # Input-history navigation. bash / zsh / terminal coding tools pattern: pressing
    # ↑ when input is empty (or anywhere) cycles backwards through prior
    # submissions; ↓ cycles forward; ↓ past the newest clears the input
    # and returns to a fresh draft. ↓ when NOT browsing history clears
    # whatever draft is currently in the field — explicit "discard this
    # prompt" gesture. History is per-session (in-memory), not persisted
    # across restarts. Duplicates of the immediately-prior entry are
    # skipped so a long scrollback isn't padded by repeats.
    def _hist_init(self) -> None:
        if not hasattr(self, "_input_history"):
            self._input_history: list[str] = []
            self._input_history_pos: int = -1  # -1 = not browsing
            self._input_history_draft: str = ""  # what user had typed before nav

    def history_push(self, text: str) -> None:
        """Called by the screen's submit handler. Append non-empty,
        non-immediate-duplicate submissions to history; reset cursor."""
        self._hist_init()
        text = (text or "").rstrip()
        if not text:
            return
        if self._input_history and self._input_history[-1] == text:
            self._input_history_pos = -1
            return
        self._input_history.append(text)
        self._input_history_pos = -1
        self._input_history_draft = ""

    def _hist_navigate(self, direction: int) -> bool:
        """Walk history. direction: -1 = older (↑), +1 = newer (↓).
        Returns True if the input value was changed (so the caller can
        prevent default key handling)."""
        self._hist_init()
        if not self._input_history and direction < 0:
            # No history to walk back into — let ↑ bubble.
            return False
        pos = self._input_history_pos
        if direction < 0:  # up = older
            if pos == -1:
                # First ↑ press: stash whatever the user was drafting
                # so ↓-past-newest can restore it.
                self._input_history_draft = self.value
                pos = len(self._input_history) - 1
            elif pos > 0:
                pos -= 1
            else:
                return False  # already at oldest
        else:  # down = newer
            if pos == -1:
                # Not browsing history. ↓ clears whatever is waiting in
                # the input field — the user's "discard this draft"
                # gesture, similar to Esc but reachable from the home
                # row. If the field is already empty there's nothing to
                # clear, so let the key bubble (cursor-end / etc).
                if not self.value:
                    return False
                self.value = ""
                self.cursor_position = 0
                self._input_history_draft = ""
                return True
            if pos < len(self._input_history) - 1:
                pos += 1
            else:
                # Past newest → restore the in-progress draft (or empty).
                self._input_history_pos = -1
                self.value = self._input_history_draft
                self.cursor_position = len(self.value)
                return True
        self._input_history_pos = pos
        self.value = self._input_history[pos]
        self.cursor_position = len(self.value)
        return True

    def on_key(self, event) -> None:
        # ── Space → push-to-talk when voice mode is on ──
        # Three branches to handle the "user wants to keep typing
        # spaces in their sentence" vs "user wants to record more"
        # tension:
        #   1. Already recording → ALWAYS PTT (treat as key-repeat).
        #   2. Input ends with a space → user is mid-sentence → type
        #      a space (don't hijack).
        #   3. Otherwise (input empty OR ends in non-space char) →
        #      PTT. After dictation, the transcript lands and the
        #      user just hits Enter or starts editing — pressing
        #      Space again at the end of "hello" would START a new
        #      recording (appending dictation to the existing text).
        if event.key == "space":
            vs = getattr(self.app, "voice_state", None)
            if vs is not None and getattr(vs, "enabled", False):
                already_recording = getattr(
                    self.screen, "_ptt_recorder", None
                ) is not None
                cur = self.value or ""
                ends_with_space = cur.endswith(" ")
                if already_recording or not ends_with_space:
                    screen = self.screen
                    ptt = getattr(screen, "action_ptt_space", None)
                    if callable(ptt):
                        ptt()
                        event.prevent_default()
                        event.stop()
                        return

        # ── Arrow-key history navigation ──
        # Only intercept when there's something to scroll; otherwise let
        # Textual's default (cursor left/right within input) handle it.
        #
        # IMPORTANT: when the slash-command menu is open, up/down must
        # navigate the menu, not history. The screen-level on_key has
        # the menu navigation logic — let the event bubble up to it
        # instead of consuming it here. (Bug 2026-04-27: pressing up
        # while typing `/` returned the previous chat input instead of
        # moving the menu selection.)
        if event.key in ("up", "down"):
            screen = self.screen
            if getattr(screen, "_slash_matches", None):
                return  # let it bubble to ChatScreen.on_key
        if event.key == "up":
            if self._hist_navigate(-1):
                event.prevent_default()
                event.stop()
        elif event.key == "down":
            if self._hist_navigate(+1):
                event.prevent_default()
                event.stop()

    # Thin vertical-bar cursor (▏ = U+258F LEFT ONE EIGHTH BLOCK) instead
    # of the full-cell block Textual draws by default. Terminals are
    # cell-based so the cursor still occupies one cell, but ▏ renders as
    # a thin vertical line on the left edge — visually reads as `|`. The
    # character originally at cursor position is hidden while the cursor
    # is over it (same trade-off as block / underline cursors). When the
    # caret is past end-of-input there's nothing to hide.
    _CURSOR_GLYPH = "▏"
    # While recording, the cursor cycles through these 8 vertical fill
    # levels based on mic amplitude — visually "moves up and down" with
    # voice volume. Plus the rainbow palette in render_line tints it.
    _VOICE_FILL_LEVELS = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")

    @property
    def _active_cursor_glyph(self) -> str:
        try:
            screen = self.screen
            rec = getattr(screen, "_ptt_recorder", None)
            if rec is not None:
                # Map mic peak [0,1] → one of the 8 fill chars. Boost
                # by 8× so typical speech (peak ~0.05-0.3) actually
                # reaches the top of the bar. Floor at index 1 (▁) so
                # even silence shows a visible "I'm listening" baseline
                # instead of a blank cell.
                peak = float(getattr(rec, "peak", 0.0) or 0.0)
                idx = max(1, min(len(self._VOICE_FILL_LEVELS) - 1,
                                  int(peak * 8.0 * len(self._VOICE_FILL_LEVELS))))
                return self._VOICE_FILL_LEVELS[idx]
        except Exception:
            pass
        return self._CURSOR_GLYPH

    def render_line(self, y):  # type: ignore[override]
        from rich.text import Text as _RichText
        from textual.strip import Strip
        if y != 0 or not self.has_focus or not self._cursor_visible:
            # Stock path for non-cursor rendering — let Textual's
            # default handle no-focus / blink-off / line>0.
            return super().render_line(y)
        console = self.app.console
        console_options = self.app.console_options
        max_content_width = self.scrollable_content_region.width
        cursor_pos = self.cursor_position
        if not self.value:
            # Empty input — render the placeholder dimmed, then overlay
            # the bar glyph at column 0 so the user sees a `|` even
            # before they type anything.
            placeholder = _RichText(
                self.placeholder, justify="left", end="",
                style=self.get_component_rich_style("input--placeholder"),
            )
            if len(placeholder) == 0:
                placeholder = _RichText(self._active_cursor_glyph, end="")
            else:
                placeholder = _RichText(
                    self._active_cursor_glyph + str(placeholder)[1:],
                    end="",
                    style=placeholder.style,
                )
            strip = Strip(console.render(
                placeholder, console_options.update_width(max_content_width + 1),
            ))
            return strip.apply_style(self.rich_style)
        # Non-empty: take the value, OVERWRITE one cell at cursor with
        # the bar glyph (white, terminal-default background). Anything
        # past end-of-input gets a single padded space we can overwrite.
        cursor_style = self.get_component_rich_style("input--cursor")
        # While voice is recording, cycle the cursor through a rainbow
        # AND tint brightness by mic amplitude. The chat screen runs a
        # 50ms refresh timer while recording so this rebuilds every
        # frame, giving the "alive" pulse Claude Code shows.
        try:
            screen = self.screen
            rec = getattr(screen, "_ptt_recorder", None)
            if rec is not None:
                from rich.style import Style as _RichStyle
                import time as _t
                _RAINBOW = (
                    "#ff5470", "#ff8a5b", "#ffd166", "#9bff8a", "#5bffc1",
                    "#5bd1ff", "#5b96ff", "#9b5bff", "#e75bff", "#ff5bd1",
                )
                # Cycle every ~0.7s, plus jump-ahead on loud peaks.
                peak = float(getattr(rec, "peak", 0.0) or 0.0)
                phase = int(_t.time() * 5) + int(min(1.0, peak * 8.0) * 4)
                color = _RAINBOW[phase % len(_RAINBOW)]
                cursor_style = _RichStyle(color="white", bgcolor=color, bold=True)
        except Exception:
            pass
        result = self._value.copy()
        if not self.selection.is_empty:
            start, end = self.selection
            start, end = sorted((start, end))
            result.stylize_before(
                self.get_component_rich_style("input--selection"), start, end,
            )
        if cursor_pos >= len(result.plain):
            result.append(self._active_cursor_glyph, style=cursor_style)
        else:
            # Replace the character at cursor position with the bar
            # glyph. Rich's Text doesn't have a direct "replace at
            # index" — rebuild the plain string with substitution and
            # reapply the cursor style on that cell.
            plain = result.plain
            new_plain = plain[:cursor_pos] + self._active_cursor_glyph + plain[cursor_pos + 1:]
            new_text = _RichText(new_plain, end="", style=result.style)
            # Preserve the value's original styling on non-cursor spans
            for span in result.spans:
                # Spans don't change because we only swapped one char.
                new_text.stylize(span.style, span.start, span.end)
            new_text.stylize(cursor_style, cursor_pos, cursor_pos + 1)
            result = new_text
        segments = list(console.render(
            result, console_options.update_width(self.content_width),
        ))
        strip = Strip(segments)
        scroll_x, _ = self.scroll_offset
        strip = strip.crop(scroll_x, scroll_x + max_content_width + 1)
        strip = strip.extend_cell_length(max_content_width + 1)
        return strip.apply_style(self.rich_style)

from ..bridge import AgentEvent, ApprovalRequest
from ..widgets.chat_log import ChatLog
from ...autonomy import AutonomyLevel, apply_autonomy_to_permissions, get_policy

_SLASH_COMMANDS = [
    ("/permissions", "Toggle ask/auto-approve for commands"),
    ("/status", "Show runtime: server health, current model, perf config"),
    ("/restart", "Restart the model server (use when /status shows 'unreachable')"),
    ("/model", "List available models / switch (e.g. /model qwen)"),
    ("/thinking", "Show / set hidden reasoning policy (off|auto)"),
    ("/sounds", "Toggle completion + approval notification sounds"),
    ("/voice", "Toggle voice mode (push-to-talk dictation into the input box)"),
    ("/audio", "Toggle audio output (assistant reads replies aloud via macOS say)"),
    ("/vision", "Toggle vision mode (let the model see images)"),
    ("/clear", "Clear conversation history"),
    ("/exit", "Exit LocalCode"),
]
# /search dropped from the palette — Ctrl+F is the canonical entry. The
# command handler still treats it as a no-op alias to avoid surprising
# anyone who typed it before.

if TYPE_CHECKING:
    from ..app import LocalCodeTUI

# Cycling thinking indicator — icons + labels rotate per tick.
_THINK_ICONS = ["·", "•", "●"]
_THINK_LABELS = [
    "thinking", "reasoning", "working",
    "planning", "checking", "analyzing",
    "processing", "computing", "considering",
]

_TOOL_CALL_RE = re.compile(
    r'<\|?tool_call\|?>.*?<\|?/?tool_call\|?>', re.DOTALL
)


def _clean_display_text(text: str) -> str:
    """Strip tool call tokens and thinking-channel artefacts before
    display. Thinking-strip patterns come from the active model family's
    adapter — for Gemma 4 that's `<unused25>` + `<|channel>thought` +
    `<channel|>`; for Qwen it's `<think>` / `</think>`. Defaults to
    Gemma so existing behaviour stays byte-identical when family is
    unset (the pre-adapter hardcoded path)."""
    from ...model_families import strip_thinking_tokens
    text = _TOOL_CALL_RE.sub("", text)
    text = strip_thinking_tokens(text)
    return text.strip()


def _is_diff_result(text: str) -> bool:
    """Check if tool result contains a diff."""
    lines = text.strip().splitlines()[:10]
    return sum(1 for l in lines if l.startswith(("+", "-", "@@", "diff "))) >= 3


_STOP_KEYWORDS = {"stop", "cancel", "abort", "halt", "wait", "pause", "quit"}


def _is_stop_intent(text: str) -> bool:
    """Does this input look like 'make it stop' — not an instruction
    that happens to mention the word 'stop'?

    We deliberately err on the side of NOT cancelling when it's
    ambiguous. A long message containing 'stop' somewhere in the
    middle (e.g. 'stop treating this as a web app and make it a CLI')
    is a real instruction the user wants queued, not a panic button.
    We only cancel when the message is SHORT and the first word is a
    stop keyword — that matches how users actually type when they
    want to bail out.
    """
    s = text.strip().lower().rstrip(" .!?,")
    if not s or len(text) > 30:
        return False
    first = s.split(None, 1)[0] if s else ""
    return first in _STOP_KEYWORDS


class ChatScreen(Screen):
    """Main chat interface — terminal coding tools inspired, with full agent loop."""

    DEFAULT_CSS = """
    ChatScreen {
        layout: vertical;
        background: $surface;
        /* Pad the screen itself so docked widgets (header at top,
           status at bottom) sit 1 row in from the terminal edges
           instead of kissing them. Textual ignores padding-top on
           dock-top widgets, but DOES honor padding on the screen,
           which moves the dock anchor row inward. */
        padding: 1 0;
    }
    #header-bar {
        dock: top;
        height: 1;
        padding: 0 1;
        width: 1fr;
        color: #5f87ff;
        background: $surface;
    }
    #active-step {
        background: $surface;
        width: 100%;
        height: 1;
        padding: 0 1;
        margin: 1 0 0 0;
        color: #5f87ff;
        overflow: hidden;
        display: none;
    }
    #active-step.active {
        display: block;
    }
    #queue-line {
        background: $surface;
        height: 1;
        padding: 0 1 0 3;
        margin: 0;
        color: $warning;
        display: none;
    }
    #queue-line.active {
        display: block;
    }
    #slash-menu {
        background: $surface;
        height: auto;
        max-height: 10;
        padding: 0 1;
        display: none;
    }
    #slash-menu.active {
        display: block;
    }
    #search-bar {
        background: $surface;
        height: 1;
        padding: 0 1;
        display: none;
    }
    #search-bar.active {
        display: block;
    }
    #search-input {
        height: 1;
        width: 1fr;
        border: none;
        background: $surface;
        display: none;
    }
    #search-input.active {
        display: block;
    }
    /* Input row and its children — explicit black so Textual's
       default surface color doesn't bleed through. */
    #input-row {
        background: $surface;
        height: 1;
    }
    #input-prompt {
        background: $surface;
        color: #5f87ff;
        width: 2;
    }
    #chat-input {
        background: $surface;
    }
    #status-bar {
        dock: bottom;
        /* 4 rows tall: 2 blank rows above the text, the text on row 3,
           1 row of bottom breathing space. */
        height: 4;
        padding: 2 1 1 1;
        color: $text-muted;
        background: $surface;
    }
    /* When the slash palette is open, the status bar gets out of the
       way so the menu has its own row and doesn't visually collide
       with the model/mode/server text (image 78). */
    #status-bar.hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+f", "toggle_search", "Search"),
        # Push-to-talk: press F2 to start recording, F2 again to stop +
        # transcribe + drop the text into the input box. Only active when
        # /voice on has been run. F2 picked because it's near-universally
        # unused inside terminals and doesn't fight with text input.
        # Space = push-to-talk when voice mode is on; types a regular
        # space character into the input box when voice mode is off.
        # `priority=True` is critical — without it the Input widget's
        # default Space handler ("type a space char") consumes the key
        # before our screen binding ever sees it, and PTT silently
        # does nothing. We compensate in the handler by manually
        # inserting a space via `insert_text_at_cursor` when voice
        # is off, so typing still works normally.
        Binding("space", "ptt_space", "Push-to-talk / type space", priority=True),
        # Esc cancels an active recording (throws away the audio, no
        # transcription). Doesn't fire when no recording is in flight.
        ("escape", "ptt_cancel", "Cancel recording"),
    ]

    def __init__(self) -> None:
        super().__init__()
        import uuid as _uuid
        self._agent_busy = False
        # True from the moment the user invokes /model or any other
        # action that restarts llama-server, until the replacement
        # server is fully loaded and reports HTTP 200 on /health.
        # While this is set, user input is queued exactly like it is
        # during an agent turn — same `_pending_messages` drain path —
        # so the user can keep typing ("hi", "what's up", "u ok?")
        # without those messages racing the still-loading model and
        # getting 503 "Loading model" back as E3102.
        self._server_restarting: bool = False
        self._pending_messages: list[str] = []
        self._stream_buf: list[str] = []
        self._last_assistant_text: str = ""
        self._turn_start: float = 0
        self._tools_used: list[str] = []
        # Stable identifier for this chat session — shared across all
        # turn-trace records so offline analysis can group by session.
        self._telemetry_session_id: str = _uuid.uuid4().hex
        self._telemetry_turn: "TurnTrace | None" = None
        self._tick_count: int = 0
        self._spin_timer = None
        self._context_used: int = 0
        # Mirror runtime.py's _target_num_ctx logic so the denominator
        # matches the value llama-server was actually launched with.
        # Previous hardcoded 32768 was half the real 65536 on turbo
        # mode — the "remaining %" bar was effectively halved.
        self._context_max: int = self._detect_ctx_max()
        self._turn_tokens: int = 0
        # Live output-token estimate during a streaming turn. Sums:
        #   • content chunks (chars / 4)
        #   • thinking chunks (chars / 4)
        #   • tool-call args streamed via tool_preview (delta of cumulative
        #     chars per tool index, divided by 4)
        # Reset on turn start. Rendered next to the elapsed timer in
        # `_tick_active` as `↓ N tokens` so the user can see decode
        # progress, like terminal coding tools's "↓ 2.7k tokens" indicator.
        self._tool_args_seen: dict[int, int] = {}
        self._total_tokens: int = 0
        # Per-turn input/output token counts captured from llama-server's
        # `usage` field on the final SSE chunk (see runtime.py
        # stream_done event). Reset per turn so the summary line can
        # show "in: X · out: Y" for THIS turn rather than a session
        # cumulative total. Falls back to 0 when the backend doesn't
        # populate `usage` (e.g. some local fallback paths).
        self._turn_prompt_tokens: int = 0
        self._turn_completion_tokens: int = 0
        self._turn_total_tokens: int = 0
        self._thinking_phase: str = ""
        self._response_shown: bool = False
        self._active_step_text: str = ""  # raw text for scanning animation
        self._active_tool_name: str = ""
        self._active_tool_args: str = ""
        self._scan_pos: int = 0
        self._step_timer = None
        self._thinking_text: str = ""  # full thinking from last turn
        # Default to collapsed (user request 2026-04-27): thinking blocks
        # land as "▶ thinking" and only expand when the user clicks the
        # triangle. Earlier expanded-by-default flooded the chat with
        # multi-paragraph reasoning the user didn't want by default.
        self._thinking_expanded: bool = False
        self._slash_matches: list[tuple[str, str]] = []  # current filtered commands
        self._slash_selected: int = 0  # highlighted index in slash menu
        # Search state
        self._search_active: bool = False
        self._search_results: list[tuple[int, str]] = []
        self._search_idx: int = 0

    @property
    def tui(self) -> "LocalCodeTUI":
        return self.app  # type: ignore

    def compose(self) -> ComposeResult:
        # Brand sits in the bottom status row alongside server/context/
        # mode/model — agent/Gemini layout pattern. Top is left clean
        # for chat content.
        yield ChatLog(id="chat-log")
        yield Static("", id="active-step")
        yield Static("", id="queue-line")
        yield Static("", id="search-bar")
        yield Input(placeholder="Search conversation...", id="search-input")
        # Voice visualizer is a dedicated widget pinned to the right of
        # the input. It's the most reliable way to show a colored,
        # amplitude-cycling bar — coloring the Input's own cursor via
        # render_line was getting overwritten by textual's base-style
        # application pipeline and showing up white. The sibling widget
        # has its own render path so color + glyph both stick.
        from ..widgets.voice_visualizer import VoiceVisualizer
        with Horizontal(id="input-row"):
            yield Static("›", id="input-prompt")
            yield _NoTintInput(placeholder="", id="chat-input")
            yield VoiceVisualizer(id="voice-visualizer")
        # Slash command palette appears BELOW the input (terminal coding tools style),
        # not above. Visually it reads as a dropdown extending downward from
        # the prompt the user is typing in.
        yield Static("", id="slash-menu")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self._update_status()
        self.query_one("#chat-input", Input).focus()
        log = self.query_one("#chat-log", ChatLog)
        # If the user launched with --resume, the TUI app stored prior
        # messages in `_pending_resume_messages`. Replay them into the
        # chat log now so the conversation picks up visually where it
        # left off. We render in a compact "history" style (dim labels)
        # to distinguish from live turns.
        msgs = getattr(self.tui, "_pending_resume_messages", None)
        if msgs:
            log.append_info(f"[dim]── resumed session ({len(msgs)} messages) ──[/]")
            for m in msgs:
                role = (m.get("role") or "").lower()
                text = (m.get("content") or "").strip()
                if not text:
                    continue
                if role == "user":
                    log.append_info(f"[dim]you:[/] {text}")
                elif role == "assistant":
                    log.append_info(f"[dim]assistant:[/] {text}")
                # tool / system messages are noise on replay — skip
            log.append_info("[dim]── continuing… ──[/]")
            # Clear so a reload doesn't double-replay
            try:
                self.tui._pending_resume_messages = []
            except Exception:
                pass
        # Periodic status refresh — without this, the status bar stays
        # stale when the server dies silently (e.g. pressure_kill
        # SIGTERMs llama-server but no event reaches this screen). 2 s
        # is cheap (only calls is_running() + a string format) and
        # responsive enough that "server: ready" disappears within a
        # couple seconds of the actual process going down.
        self.set_interval(2.0, self._update_status)

    def on_resize(self, event) -> None:
        # The status bar's left/right padding is computed from the
        # terminal width — recompute when the user resizes. Cheap (no
        # network, no model calls) so always-on is fine.
        try:
            self._update_status()
        except Exception:
            pass
        try:
            log = self.query_one("#chat-log", ChatLog)
            if getattr(log, "_history", None) and not getattr(log, "_stream_started", False):
                log._rerender()
        except Exception:
            pass

    # Header bar method removed (commit 2026-04-25). Brand moved to the
    # left of #status-bar — see _update_status. If we ever bring back
    # an in-app top bar (e.g. for breadcrumbs / current-mode pill), it
    # should be a *new* widget, not a revival of this one.

    def _show_thinking(self) -> None:
        self._turn_tokens = 0
        self._tool_args_seen.clear()
        self._turn_total_tokens = 0
        self._thinking_phase = ""
        self._response_shown = False
        if self._spin_timer is None:
            self._spin_timer = self.set_interval(0.15, self._tick_thinking)
        self._show_active_thinking("thinking")

    def _hide_thinking(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
            self._tick_count = 0

    def _tick_thinking(self) -> None:
        self._tick_count += 1

    # ── Active step (scanning highlight animation) ──

    _active_mode: str = ""  # "tool" or "thinking"

    # Map tool names to present-tense verbs for the live status
    _TOOL_VERBS = {
        "bash": "running",
        "read_file": "reading",
        "write_file": "writing",
        "append_file": "appending",
        "edit_file": "editing",
        "multi_edit": "editing",
        "edit_diff": "editing",
        "grep": "searching",
        "glob": "searching",
        "list_files": "browsing files",
        "launch_app": "launching",
        "web_fetch": "fetching",
        "web_search": "searching the web",
        "code_search": "searching code",
    }

    def _show_active_step(self, name: str, args: str) -> None:
        """Show live tool call in the #active-step widget above the input."""
        self._active_tool_name = name
        self._active_tool_args = args
        # Show the tool name in the active step area with blue ball
        verb = self._TOOL_VERBS.get(name, name)
        args_short = args.strip().replace("\n", " ")[:40] if args else ""
        display = f"{verb}({args_short})" if args_short else verb
        self._active_step_text = display
        self._active_mode = "tool"
        self._scan_pos = 0
        w = self.query_one("#active-step", Static)
        w.add_class("active")
        self._tick_active()
        if self._step_timer is not None:
            self._step_timer.stop()
        self._step_timer = self.set_interval(0.05, self._tick_active)

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
        self._tick_active()
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
        """Format elapsed time since turn start with live token count.

        Returns a parenthesised badge like `(11s · ↓ 234 tokens)` or
        `(1m11s · ↓ 2.7k tokens)` so the user can see decode progress
        the way terminal coding tools does. Falls back to time-only when the
        turn hasn't emitted any output yet."""
        elapsed = time.time() - self._turn_start if self._turn_start else 0
        if elapsed < 60:
            timer = f"{elapsed:.0f}s"
        else:
            m, s = divmod(int(elapsed), 60)
            timer = f"{m}m{s:02d}s"
        toks = max(0, int(self._turn_tokens))
        if toks <= 0:
            return f"({timer})"
        if toks >= 1000:
            tok_str = f"{toks / 1000:.1f}k"
        else:
            tok_str = str(toks)
        return f"({timer} · ↓ {tok_str} tokens)"

    def _tick_active(self) -> None:
        """Single animation tick for both tools and thinking."""
        text = self._active_step_text
        if not text:
            return

        timer = self._elapsed_str()

        # ● for tools (blue ball), ◆ for thinking
        icon = "●" if self._active_mode == "tool" else "◆"
        label = f"{icon} {text}..."
        try:
            width = max(12, self.query_one("#active-step", Static).size.width - 2)
        except Exception:
            width = 80

        # Advance sweep position before width-specific rendering. The
        # narrow-terminal branch used to return before this increment, so
        # minimized terminals showed a static "writing..." / "thinking..."
        # line instead of the animated scan.
        self._scan_pos = (self._scan_pos + 1) % max(len(label), 1)

        if width < 64:
            compact_text = "thinking" if self._active_mode == "thinking" else text
            if self._active_mode == "tool" and "(" in compact_text:
                compact_text = compact_text.split("(", 1)[0]
            label = f"{icon} {compact_text}..."
            max_label = max(8, width - (len(timer) + 4 if width >= 28 else 2))
            if len(label) > max_label:
                label = label[: max(3, max_label - 1)] + "…"
            pos = self._scan_pos % max(len(label), 1)
            line = RichText()
            line.append("  ")
            line.append(label[:pos + 1], style=f"bold {C.primary}")
            line.append(label[pos + 1:], style="dim italic")
            if width >= 28:
                line.append(f" {timer}", style="dim")
            self.query_one("#active-step", Static).update(line)
            return

        pos = self._scan_pos
        bright = label[:pos + 1]
        dim = label[pos + 1:]
        # Escape markup characters in the text
        bright = bright.replace("[", "\\[")
        dim = dim.replace("[", "\\[")
        line = f"  [bold]{bright}[/][dim italic]{dim}[/]  [dim]{timer}[/]"

        w = self.query_one("#active-step", Static)
        w.update(line)

    # ── Status bar (bottom — model, mode, context remaining) ──

    def _detect_ctx_max(self) -> int:
        """Best-effort read of the context window the server is using.

        Prefers the runtime config's computed value (runtime.py's
        _target_num_ctx mirrors the logic for the active mode). Falls
        back to 65536 for Apple-Silicon turbo, 32768 otherwise — these
        match runtime.py's defaults.
        """
        try:
            cfg = self.tui.config
            rt = cfg.runtime
            # Let the runtime compute it the same way the server launch does.
            from ..runtime import LocalCodeRuntimeGateway
            gw = LocalCodeRuntimeGateway(rt)
            return int(gw._target_num_ctx())
        except Exception:
            return 65536

    def _recompute_context_used(self) -> int:
        """Recompute total prompt tokens currently held in the session.

        Single-path accounting — replaces the old mix of (a) per-input
        delta, (b) session-snapshot overwrite, (c) per-response delta
        that double-counted on every turn and forgot the system prompt.

        Sums: assistant/user/tool content + system prompt + tool schemas
        JSON + skills listing. Uses a 4:1 char→token heuristic; when
        llama-server's /tokenize endpoint is available, `_tokenize_len`
        can be swapped in later for exactness.
        """
        total = 0
        app = getattr(self.tui, "engine", None)
        if app is None:
            return 0
        # Session messages (user / assistant / tool).
        for m in getattr(app.session, "messages", []):
            c = m.get("content", "")
            total += len(str(c)) if c else 0
            for tc in m.get("tool_calls", []) or []:
                total += len(str(tc.get("function", {}).get("arguments", "")))
        # Baseline for the always-present system prompt + tools listing.
        # Actual SYSTEM_PROMPT + TOOL_SCHEMAS + skills block is ~4 KB in
        # our config; use that as a fixed offset rather than reparsing.
        total += 4000
        return total // 4  # chars → approx tokens

    def _update_status(self) -> None:
        config = self.tui.config
        mode = config.runtime.laptop_26b_runtime_mode
        mode_label = "fast" if not mode.endswith("-think") else "reasoning"
        from ...models_catalog import current as current_choice
        cur = current_choice(config)
        if cur is not None:
            model = cur.name
        else:
            raw_model = (config.runtime.model or "").strip()
            if raw_model.endswith(".gguf"):
                model = raw_model.split("/")[-1].replace(".gguf", "")
            elif "sha256-" in raw_model:
                model = "local gguf"
            else:
                model = raw_model or "no model selected"
        task_stage = ""
        try:
            task = getattr(getattr(self.tui.engine, "session", None), "current_task", None)
            if task is not None and getattr(task, "current_stage", ""):
                task_stage = str(getattr(task, "current_stage", "")).strip()
            elif getattr(self.tui.engine, "_last_turn_task_stage", ""):
                task_stage = str(getattr(self.tui.engine, "_last_turn_task_stage", "")).strip()
        except Exception:
            task_stage = ""
        # Server status — short, plain-English action label. The
        # degraded path (Ollama at ~1.7 tok/s vs ~27 tok/s normally)
        # tells the user EXACTLY what to do, not technical jargon
        # like "fallback" or "turbo".
        provider = (config.runtime.provider or "").lower()
        # Plain `key: value` — the value word ("ready", "loading",
        # "degraded", "not connected") IS the state; a leading glyph
        # would be decoration. Same shape as the other status fields.
        if self._server_restarting:
            server_label = "server: loading model"
        elif provider == "llama_cpp":
            # Actually probe ServerManager rather than hardcoding "ready"
            # whenever the provider config is llama_cpp. Prior bug
             # (2026-04-27): after pressure_kill SIGTERM'd llama-server,
            # the status bar still showed "ready" because nothing in
            # this branch checked liveness — user typed a message, got
            # "Backend not ready" from the gateway, and saw a status
            # bar that contradicted the error.
            try:
                from ...server_manager import ServerManager as _SM
                _mgr = _SM.get()
                # ALSO probe the live port — `is_running()` only knows
                # about processes we spawned, but a llama-server launched
                # by the user (or a prior session) on the same port is
                # serving fine and our HTTP requests would succeed.
                # Don't claim "stopped" when the agent can clearly reach it.
                _alive = _mgr.is_running() or _mgr.is_healthy()
            except Exception:
                _alive = True  # if we can't probe, fall through to old behaviour
            server_label = "server: ready" if _alive else "server: stopped"
        elif provider == "ollama":
            server_label = "server: degraded (restart to fix)"
        elif provider:
            server_label = f"server: {provider}"
        else:
            server_label = "server: not connected"
        # Context REMAINING (starts at 100%, decreases)
        pct_remaining = max(0, 100 - int(self._context_used / max(1, self._context_max) * 100))
        bar = self.query_one("#status-bar", Static)
        # Version tag lets the user see at a glance whether they're on
        # the latest build without `git log` in another terminal.
        # Shows `app_version + short git commit` when both are available;
        # falls back to "dev" if running from an uninstalled source
        # checkout with no git info. Computed once per session —
        # cached on self to avoid spawning `git` every status update.
        if not hasattr(self, "_build_tag"):
            try:
                from importlib.metadata import version as _pkgver
                app_ver = _pkgver("localcode")
            except Exception:
                app_ver = ""
            commit = ""
            try:
                import subprocess as _sp
                from pathlib import Path as _Path
                # Best-effort — dev installs have git metadata, installed
                # builds usually don't; either branch is fine.
                r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=1,
                            cwd=str(_Path(__file__).resolve().parent))
                if r.returncode == 0:
                    commit = r.stdout.strip()
            except Exception:
                pass
            # Cache the components separately so we can render the
            # version adaptively per terminal width below — full
            # `v0.2.12·c0def7a` on wide terminals, ellipsis-truncated
            # `v0.2.12…` on medium, dropped entirely on narrow.
            self._build_version = app_ver
            self._build_commit = commit
            # Default fallback used if width-aware rendering can't run
            # (e.g. very early paint before bar.size is populated).
            if app_ver and commit:
                self._build_tag = f"v{app_ver}…"
            elif app_ver:
                self._build_tag = f"v{app_ver}"
            elif commit:
                self._build_tag = f"dev…"
            else:
                self._build_tag = "dev"
        # Adaptive version display. Recomputed each tick because the
        # user can resize the terminal mid-session.
        try:
            term_cols = self.app.size.width
        except Exception:
            term_cols = 0
        ver = getattr(self, "_build_version", "")
        sha = getattr(self, "_build_commit", "")
        if ver and sha and term_cols >= 110:
            build_tag = f"v{ver}·{sha}"
        elif ver and sha:
            build_tag = f"v{ver}…"
        elif ver:
            build_tag = f"v{ver}"
        elif sha and term_cols >= 110:
            build_tag = f"dev·{sha}"
        elif sha:
            build_tag = "dev…"
        else:
            build_tag = "dev"
        # Don't show the version at all on narrow terminals — the left
        # side (server / context / mode / model) is more useful.
        show_version = term_cols == 0 or term_cols >= 80
        # Order: critical-info LEFT → secondary RIGHT. When the
        # terminal is narrow the rightmost stuff (build/model) gets
        # clipped first, but server status + context % stay visible.
        # Earlier the order was reversed and "server: ready · X%
        # context" was the first thing to disappear off-screen.
        # Also shorten the model label — full Unsloth quant names
        # like "Qwen 3.6 35B-A3B (Unsloth UD-IQ2_M)" eat 35 chars.
        short_model = (
            model.replace(" (Unsloth UD-", " ")
                 .replace(")", "")
                 .replace("-A3B", "")
        )
        # Brand prefix + status (LEFT) | version (RIGHT). Like a tmux /
        # vim statusline — left content pinned to the left edge, right
        # content pinned to the right edge, padded with spaces to fill
        # the terminal width. Without this the version sat awkwardly
        # mid-row on wide terminals with empty space trailing it.
        left = RichText.from_markup(
            f"🏠[{C.primary}]LocalCode[/]  │  "
            f"{server_label}  ·  context: {pct_remaining}% free  ·  "
            f"mode: {mode_label}"
            + (f"  ·  task: {task_stage}" if task_stage else "")
            + f"  ·  model: {short_model}"
        )
        # Use the adaptive `build_tag` chosen above; honour the
        # narrow-terminal "drop entirely" policy by emitting an empty
        # right-side, so the padding loop below collapses to no gap.
        right = RichText.from_markup(
            f"[dim]{build_tag}[/]" if show_version else ""
        )
        # `bar.size.width` is 0 before first paint; fall back to app
        # width minus the two-column padding the CSS adds.
        try:
            bar_width = bar.size.width or (self.app.size.width - 2)
        except Exception:
            bar_width = 0
        if bar_width and left.cell_len + right.cell_len + 2 <= bar_width:
            gap = bar_width - left.cell_len - right.cell_len
            combined = RichText()
            combined.append_text(left)
            combined.append(" " * gap)
            combined.append_text(right)
            bar.update(combined)
        else:
            # Narrow terminal — drop the right side rather than wrap.
            # Server / context / mode / model matter more than version.
            bar.update(left)

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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show slash command menu when user types /, or live search."""
        # Search input — live search as you type
        if event.input.id == "search-input":
            self._do_search(event.value)
            return
        text = event.value
        menu = self.query_one("#slash-menu", Static)
        # Also toggle the status bar in lockstep with the slash menu —
        # when the menu is open it pushes content downward and visually
        # collides with the status bar (image 78 showed the model/mode/
        # context line getting overlapped). Hiding the status bar while
        # the menu is open gives the menu its own clean row of space.
        status_bar = self.query_one("#status-bar", Static)
        if text.startswith("/") and not text.startswith("/ "):
            prefix = text.lower()
            self._slash_matches = [(cmd, desc) for cmd, desc in _SLASH_COMMANDS if cmd.startswith(prefix)]
            if self._slash_matches:
                self._slash_selected = min(self._slash_selected, len(self._slash_matches) - 1)
                self._render_slash_menu()
                menu.add_class("active")
                status_bar.add_class("hidden")
            else:
                self._slash_matches = []
                menu.remove_class("active")
                status_bar.remove_class("hidden")
        else:
            self._slash_matches = []
            self._slash_selected = 0
            menu.remove_class("active")
            status_bar.remove_class("hidden")

    def _render_slash_menu(self) -> None:
        """Render the slash palette below the input.

        Each row stays on exactly ONE line. If the description doesn't
        fit, truncate with `…` instead of wrapping (wrapping made it
        look like there were extra options below). Width is computed
        from the actual screen width minus the fixed command column.
        """
        menu = self.query_one("#slash-menu", Static)
        try:
            avail = max(20, (self.size.width or 80) - 18)
        except Exception:
            avail = 60
        lines = []
        for i, (cmd, desc) in enumerate(self._slash_matches):
            d = desc if len(desc) <= avail else desc[: max(0, avail - 1)].rstrip() + "…"
            if i == self._slash_selected:
                lines.append(f"[bold]{cmd:<14}[/]  [bold]{d}[/]")
            else:
                lines.append(f"[dim]{cmd:<14}[/]  [dim]{d}[/]")
        menu.update("\n".join(lines))

    # Screen-level on_paste was removed 2026-04-26: `_NoTintInput._on_paste`
    # (chat.py:32) now intercepts paste BEFORE Textual's stock handler and
    # inserts the joined text itself. Keeping the screen-level handler too
    # caused every paste to land twice — Input override inserted, event
    # bubbled (or didn't get stopped quickly enough), screen handler
    # inserted AGAIN. Single source of truth: the Input subclass.

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Search input — Enter navigates to next result
        if event.input.id == "search-input":
            if self._search_results:
                self._search_idx = (self._search_idx + 1) % len(self._search_results)
                self._show_search_result()
            return

        text = event.value.strip()
        if not text:
            return
        # Record into per-input history so ↑/↓ can recall it next time.
        # Skipped for slash commands (next line clears event.input)
        # because users normally don't want `/clear` and `/quit` in
        # their navigable history; safe to add later if desired.
        if hasattr(event.input, "history_push") and not text.startswith("/"):
            event.input.history_push(text)
        event.input.clear()
        # Invalidate any in-flight voice transcription so a worker that
        # finishes AFTER this submit doesn't write the just-submitted
        # text back into the now-empty input. Bumping the session
        # counter makes the worker's captured `session_at_start` mismatch
        # `_ptt_session`, so `_apply_partial_transcript` silently drops
        # the update. Also clear the prefix so the next voice session
        # starts fresh.
        self._ptt_session = getattr(self, "_ptt_session", 0) + 1
        self._ptt_input_prefix = ""
        self._ptt_last_transcript = ""

        if text.startswith("/"):
            self._handle_command(text)
            return

        log = self.query_one("#chat-log", ChatLog)

        if self._agent_busy or self._server_restarting:
            # Fast-path: if the message is the user trying to bail out
            # of a running turn (because the model is stuck in a loop
            # or going in the wrong direction), cancel immediately
            # instead of queueing. Without this, "stop" gets queued as
            # the NEXT instruction — which fires AFTER the current bad
            # turn finishes, i.e. too late to help. We do NOT honour
            # stop-intent during `_server_restarting`: there is no
            # agent turn to cancel; the server is loading and we just
            # queue everything until it's ready.
            if self._agent_busy and _is_stop_intent(text):
                self._request_cancel(text)
                return
            self._pending_messages.append(text)
            self._update_queue()
        else:
            log.append_user(text)
            log.scroll_end(animate=False)
            self._start_turn(text)

    def _request_cancel(self, text: str) -> None:
        """Mark the current turn for cancellation and drop any queue."""
        log = self.query_one("#chat-log", ChatLog)
        log.append_user(text)
        # Flip the flag on the app — agent.py polls this at round
        # boundaries and before each tool invocation.
        if self.tui.engine is not None:
            self.tui.engine.cancel_requested = True
        # Drop everything queued — the user is clearly signaling they
        # don't want more of this going on.
        dropped = len(self._pending_messages)
        self._pending_messages.clear()
        self._update_queue()
        note = "Stop requested — cancelling at next safe point"
        if dropped:
            note += f" (also dropped {dropped} queued message{'s' if dropped != 1 else ''})"
        log.append_info(note + ".")

    def _handle_command(self, text: str) -> None:
        log = self.query_one("#chat-log", ChatLog)
        if text not in ("/quit", "/exit", "/clear"):
            log.write(RichText(""))  # spacing before command output
        if text in ("/quit", "/exit"):
            quit_action = getattr(self.app, "action_quit", None)
            if callable(quit_action):
                quit_action()
            else:
                self.app.exit()
        elif text == "/clear":
            log.clear()
            if self.tui.engine:
                self.tui.engine.session.messages.clear()
            self._context_used = 0
            self._total_tokens = 0
            self._update_status()
        elif text == "/undo":
            if self.tui.engine:
                result = self.tui.engine._handle_command("/undo")
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
        elif text == "/permissions":
            app = self.tui.engine
            if app:
                if app._autonomy == AutonomyLevel.FULL_AUTO:
                    app._autonomy = AutonomyLevel.AUTO_EDIT
                    apply_autonomy_to_permissions(app.perms, get_policy(app._autonomy))
                    log.append_info("Permissions ON — will ask before running commands")
                else:
                    app._autonomy = AutonomyLevel.FULL_AUTO
                    apply_autonomy_to_permissions(app.perms, get_policy(app._autonomy))
                    log.append_info("Permissions OFF — full auto, no questions asked")
        elif text == "/search":
            self.action_toggle_search()
        elif text == "/model" or text.startswith("/model "):
            self._handle_model_command(text)
        elif text == "/thinking" or text.startswith("/thinking "):
            self._handle_thinking_command(text)
        elif text == "/status":
            self._handle_status_command()
        elif text == "/restart":
            log = self.query_one("#chat-log", ChatLog)
            log.append_info("Restarting model server...")
            self._restart_for_vision_change(reason="Server restarted")
        elif text == "/voice" or text.startswith("/voice "):
            self._handle_voice_command(text)
        elif text == "/audio" or text.startswith("/audio "):
            self._handle_audio_command(text)
        elif text == "/vision" or text.startswith("/vision "):
            self._handle_vision_command(text)
        elif text == "/sounds":
            cfg = self.tui.config
            cfg.ui.sounds_enabled = not cfg.ui.sounds_enabled
            try:
                from ...config import save_config
                save_config(cfg)
            except Exception:
                pass
            if cfg.ui.sounds_enabled:
                log.append_info("Sounds ON — completion + approval will chime")
                # Preview the completion sound so the user hears what's
                # being enabled, without having to wait for a turn to end.
                try:
                    from ...sounds import play_completion
                    play_completion(True)
                except Exception:
                    pass
            else:
                log.append_info("Sounds OFF")
        else:
            log.append_info(f"Unknown command: {text}")

    def _handle_thinking_command(self, text: str) -> None:
        """Handle `/thinking` policy changes for hidden reasoning.

        Supported values:
        - off: never use hidden thinking
        - auto: enable it selectively for harder turns
        """
        log = self.query_one("#chat-log", ChatLog)
        cfg = self.tui.config
        parts = text.strip().split(maxsplit=1)
        current = (cfg.runtime.internal_thinking_mode or "off").strip().lower() or "off"

        if len(parts) == 1:
            value = "auto" if current == "off" else "off"
            cfg.runtime.internal_thinking_mode = value
            try:
                from ...config import save_config
                save_config(cfg)
            except Exception:
                pass
            log.append_info(f"Thinking {value}.")
            return

        value = parts[1].strip().lower()
        valid = {"off", "auto"}
        if value not in valid:
            log.append_error(
                f"Unknown thinking policy '{value}'. Valid: off, auto."
            )
            return

        cfg.runtime.internal_thinking_mode = value
        try:
            from ...config import save_config
            save_config(cfg)
        except Exception:
            pass
        log.append_info(f"Thinking {value}.")

    def _handle_status_command(self) -> None:
        """Show what's running, what model is loaded, and where things live.

        Replaces the deleted `localcode status` CLI subcommand — same info,
        but accessible mid-conversation without quitting.
        """
        from pathlib import Path as _P
        from ...models_catalog import current as current_choice
        from ...runtime import LocalCodeRuntimeGateway
        log = self.query_one("#chat-log", ChatLog)
        config = self.tui.config

        # Server health
        try:
            gw = LocalCodeRuntimeGateway(config.runtime)
            ok, details = gw.healthcheck()
        except Exception as e:
            ok, details = False, str(e)
        status_str = "🟢 ok" if ok else f"🔴 unreachable ({details})"

        # Resolve the catalog entry behind the configured model path.
        choice = current_choice(config)
        model_disp = choice.name if choice is not None else _P(config.runtime.model or "—").name
        mmproj_on = "yes" if (choice and choice.mmproj_path and choice.mmproj_path.is_file()) else "no"

        # Voice state (lazy)
        vs = getattr(self.tui, "voice_state", None)
        voice_disp = (
            f"on (tts={vs.tts_engine}, speak={vs.tts_speak_mode})"
            if (vs is not None and vs.enabled) else "off"
        )

        # Plain text only — append_info uses `Text(line, style="dim")`
        # which does NOT parse [bold]/[/] markup. We section the output
        # with ALL-CAPS labels + leading blank lines instead.
        lines = [
            "RUNTIME",
            f"  server          {status_str}",
            f"  url             {config.runtime.base_url}",
            f"  provider        {config.runtime.provider}",
            f"  model file      {_P(config.runtime.model or '').name or '—'}",
            f"  model name      {model_disp}",
            f"  profile         {config.runtime.profile}",
            f"  vision (mmproj) {mmproj_on}",
            f"  voice           {voice_disp}",
            "",
            "PERFORMANCE",
            f"  gpu_layers      {getattr(config.runtime, 'llama_cpp_gpu_layers', '—')}",
            f"  threads         {getattr(config.runtime, 'llama_cpp_threads', '—')}",
            f"  kv_cache        {config.runtime.kv_cache_type_k} / {config.runtime.kv_cache_type_v}",
            f"  context         {getattr(config.runtime, 'cache_policy', '—')}",
            "",
            "PATHS",
            "  config          ~/.localcode/config.toml",
            "  models dir      ~/.local/share/localcode/models/",
            "  voice dir       ~/.local/share/localcode/voice/",
            "  server log      ~/.local/share/localcode/server.log",
        ]
        log.append_info("\n".join(lines))

    def _handle_voice_command(self, text: str) -> None:
        """`/voice` toggles voice mode on/off — same pattern as /sounds.

        First-time-on auto-downloads the Whisper STT model (~370 MB)
        with an inline confirm. Subcommands kept for power users:
          /voice setup  — download model only, don't enable
          /voice tts off|final|always
          /voice tts say|piper
          /voice status — verbose diagnostic
        """
        log = self.query_one("#chat-log", ChatLog)
        if not hasattr(self.tui, "voice_state"):
            from ...voice import VoiceState as _VS
            self.tui.voice_state = _VS()
        state = self.tui.voice_state
        from ...voice import stt_model_ready, ensure_stt_model

        parts = text.strip().split()
        sub = parts[1] if len(parts) >= 2 else None

        # ── bare /voice → toggle ──────────────────────────────────
        if sub is None:
            if state.enabled:
                state.enabled = False
                log.append_info("Voice mode OFF.")
                return
            # Hardware capability gate — bail BEFORE the user waits for
            # a 514 MB download that won't work because there's no mic,
            # no Info.plist mic descriptor on the host terminal, etc.
            try:
                from ...voice import detect_voice_capability
                ok, hint = detect_voice_capability()
                if not ok:
                    log.append_error(f"Voice unavailable: {hint}")
                    return
            except Exception:
                pass
            # Turning ON
            if not stt_model_ready(state):
                # Already-running guard so a frustrated user mashing
                # /voice doesn't kick off two concurrent downloads.
                if getattr(self, "_voice_download_in_flight", False):
                    log.append_info("Voice model is already downloading — please wait.")
                    return
                self._voice_download_in_flight = True
                log.append_info(
                    "Voice mode needs the Whisper STT model (~514 MB). "
                    "Downloading in background — UI stays responsive. "
                    "Resumable, will auto-retry on transient errors."
                )
                # Single-line dynamic progress — `Downloading: X% (Y/Z MB)`
                # is shown in the `#active-step` Static widget (already
                # designed for transient one-line status), not appended
                # to the chat log. That way the number updates in place
                # instead of dumping 70+ lines into the conversation.
                def _progress_from_thread(msg: str) -> None:
                    try:
                        is_progress = msg.startswith("Downloading:")
                        if is_progress:
                            # Mutate the single-line status widget in place
                            self.app.call_from_thread(self._set_download_line, msg)
                        else:
                            # Non-progress messages (retry, fallback, completion)
                            # still go into the chat log so the user has a
                            # durable record.
                            self.app.call_from_thread(log.append_info, msg)
                    except Exception:
                        pass

                def _worker() -> None:
                    ok, result = ensure_stt_model(state, on_progress=_progress_from_thread)
                    self.app.call_from_thread(self._finish_voice_download, ok, result)

                import threading as _t
                _t.Thread(target=_worker, daemon=True).start()
                return
            # Model already on disk — just toggle on synchronously.
            state.enabled = True
            log.append_info(
                f"Voice mode ON. Hold {state.ptt_key.upper()} to talk."
            )
            self._maybe_warn_terminal_mic_access(log)
            return

        # ── explicit subcommands (power users) ─────────────────────
        if sub in ("on", "off"):
            # Legacy explicit form — keep working but route through toggle logic
            if sub == "on" and not state.enabled:
                self._handle_voice_command("/voice")
                return
            if sub == "off" and state.enabled:
                state.enabled = False
                log.append_info("Voice mode OFF.")
            return

        if sub == "setup":
            log.append_info("Downloading Whisper STT model… (one-time, ~370 MB)")
            ok, result = ensure_stt_model(
                state, on_progress=lambda m: log.append_info(m),
            )
            if ok:
                from pathlib import Path as _P
                state.stt_model_path = _P(result)
                log.append_info("STT model ready.")
            else:
                log.append_error(f"Setup failed: {result}")
            return

        if sub == "status":
            ready = stt_model_ready(state)
            log.append_info(
                f"Voice: {'on' if state.enabled else 'off'} · "
                f"STT model {'ready' if ready else 'missing'} · "
                f"TTS={state.tts_engine}/{state.tts_speak_mode} · "
                f"PTT={state.ptt_key.upper()}"
            )
            return

        if sub == "tts" and len(parts) >= 3:
            mode = parts[2]
            if mode in ("off", "final", "always"):
                state.tts_speak_mode = mode
                log.append_info(f"TTS speak mode: {mode}")
            elif mode in ("say", "piper"):
                state.tts_engine = mode
                log.append_info(f"TTS engine: {mode}")
            else:
                log.append_info(
                    "Usage: /voice tts off|final|always   OR   /voice tts say|piper"
                )
            return

        log.append_info("Usage: /voice  (toggle)  ·  /voice setup  ·  /voice tts …  ·  /voice status")

    def _set_download_line(self, msg: str) -> None:
        """Update the single-line transient status widget with a download
        progress message. Called on the UI thread only."""
        try:
            self.query_one("#active-step", Static).update(f"[dim]{msg}[/]")
        except Exception:
            pass

    def _clear_download_line(self) -> None:
        try:
            self.query_one("#active-step", Static).update("")
        except Exception:
            pass

    def _handle_audio_command(self, text: str) -> None:
        """`/audio` toggles spoken responses (TTS) on/off.

        Independent of /voice — TTS uses macOS `say` (built-in, no
        download). State lives in voice_state.tts_speak_mode but the
        command is named `/audio` because users think of "voice" as
        "I talk" and "audio" as "it talks back".

        Subcommands:
          /audio              — toggle off ⇄ final
          /audio off          — silent
          /audio final        — speak only the final answer of each turn
          /audio always       — speak every assistant message
        """
        log = self.query_one("#chat-log", ChatLog)
        if not hasattr(self.tui, "voice_state"):
            from ...voice import VoiceState as _VS
            self.tui.voice_state = _VS()
        state = self.tui.voice_state

        parts = text.strip().split()
        sub = parts[1] if len(parts) >= 2 else None

        if sub is None:
            # Toggle off ⇄ final
            if state.tts_speak_mode != "off":
                state.tts_speak_mode = "off"
                log.append_info("Audio output OFF.")
            else:
                state.tts_speak_mode = "final"
                log.append_info(
                    "Audio output ON. Final answer of each turn will be read aloud."
                )
            return

        if sub in ("off", "final", "always"):
            state.tts_speak_mode = sub
            if sub == "off":
                log.append_info("Audio output OFF.")
            elif sub == "final":
                log.append_info("Audio output: final answer only.")
            else:
                log.append_info("Audio output: every assistant message.")
            return

        if sub == "voices":
            # List available macOS voices, highlighting Premium / Enhanced
            # ones that sound dramatically more natural than the default.
            import subprocess as _sp
            try:
                out = _sp.run(["say", "-v", "?"], capture_output=True, text=True, timeout=3).stdout
            except Exception as e:
                log.append_error(f"Couldn't enumerate voices: {e}")
                return
            premium, enhanced, basic = [], [], []
            for line in out.splitlines():
                if "(Premium)" in line:
                    premium.append(line)
                elif "(Enhanced)" in line:
                    enhanced.append(line)
                elif "en_" in line.lower() or " en_" in line:
                    basic.append(line)
            lines = ["MOST NATURAL — best quality (download via System Settings → Accessibility → Spoken Content → System Voice):"]
            lines += ["  " + v[:80] for v in premium[:8]] or ["  (none installed — go enable Premium voices in System Settings)"]
            lines.append("")
            lines.append("ENHANCED — still much better than default:")
            lines += ["  " + v[:80] for v in enhanced[:8]] or ["  (none installed)"]
            lines.append("")
            lines.append("Pick one with `/audio voice <Name>` (e.g. /audio voice Ava).")
            log.append_info("\n".join(lines))
            return

        if sub == "voice" and len(parts) >= 3:
            voice_name = " ".join(parts[2:])
            # `piper:<voice-id>` switches engine to Piper TTS (much more
            # natural). First use of a piper voice triggers an automatic
            # ~25-100 MB download of the .onnx voice model from
            # huggingface.co/rhasspy/piper-voices.
            if voice_name.startswith("piper:"):
                voice_id = voice_name[len("piper:"):].strip()
                if not voice_id:
                    log.append_info("Usage: /audio voice piper:<voice-id>  e.g. piper:en_US-amy-medium")
                    return
                state.tts_engine = "piper"
                state.tts_voice = voice_id
                log.append_info(
                    f"TTS engine: piper · voice: {voice_id}. "
                    "First use will download the model (~25-100 MB) into "
                    "~/.local/share/localcode/voice/piper/."
                )
            else:
                state.tts_engine = "say"
                state.tts_voice = voice_name
                log.append_info(
                    f"TTS voice: {voice_name} (macOS say). Next assistant "
                    "message will use it."
                )
            return

        log.append_info(
            "Usage: /audio (toggle) · /audio off|final|always · "
            "/audio voices · /audio voice <Name>"
        )

    def _finish_voice_download(self, ok: bool, result: str) -> None:
        """Called on the UI thread when the background voice-model download
        completes. Enables voice mode on success or surfaces an error +
        retry hint."""
        self._voice_download_in_flight = False
        self._clear_download_line()
        log = self.query_one("#chat-log", ChatLog)
        state = getattr(self.tui, "voice_state", None)
        if not ok or state is None:
            log.append_error(f"Couldn't enable voice: {result}")
            log.append_info(
                "Type [bold]/voice[/] again to retry — partial download is "
                "preserved on disk and will resume from where it stopped."
            )
            return
        from pathlib import Path as _P
        state.stt_model_path = _P(result)
        state.enabled = True
        log.append_info(
            f"Voice mode ON. Hold {state.ptt_key.upper()} to talk."
        )
        self._maybe_warn_terminal_mic_access(log)

    def _maybe_warn_terminal_mic_access(self, log) -> None:
        """If the host terminal can't request mic access (e.g. VS Code's
        integrated terminal), warn the user upfront so they don't waste
        time wondering why Space-hold does nothing."""
        try:
            from ...voice import host_terminal_supports_mic
            ok, hint = host_terminal_supports_mic()
            if not ok and hint:
                log.append_info(hint)
        except Exception:
            pass

    def _finish_vision_download(self, ok: bool, result: str, choice_key: str) -> None:
        """Called on the UI thread when the background mmproj download
        completes. Auto-restarts the server so --mmproj gets picked up —
        the user never has to type /model themselves."""
        self._vision_download_in_flight = False
        self._clear_download_line()
        log = self.query_one("#chat-log", ChatLog)
        if not ok:
            log.append_error(f"Couldn't enable vision: {result}")
            log.append_info(
                "Type [bold]/vision[/] again to retry — partial download is preserved."
            )
            return
        log.append_info("Vision projector ready — restarting server to activate...")
        self._restart_for_vision_change(reason="Vision ON")

    def _restart_for_vision_change(self, reason: str) -> None:
        """Force a server restart to pick up an mmproj change.

        Used both when vision turns ON (mmproj just downloaded; runtime
        needs to re-launch with --mmproj) and when it turns OFF (mmproj
        deleted; runtime needs to re-launch without --mmproj).

        The /model handler short-circuits when the same model is
        already loaded, so we bypass it and call _restart_server
        directly. Mirrors the same _server_restarting / queue-pending
        UX so user input during the swap is not lost.
        """
        log = self.query_one("#chat-log", ChatLog)
        self._server_restarting = True
        self._update_status()

        def _worker() -> None:
            # Auto-init the backend if it hasn't been touched yet — the
            # /vision toggle should never fail just because the user
            # hadn't sent their first message yet. ensure_backend is
            # idempotent + cheap when already up.
            try:
                self.app.call_from_thread(self.tui.ensure_backend)
                import time as _t
                # Give the call_from_thread a tick to land before we
                # check engine. 50ms is enough; engine init is in-process.
                _t.sleep(0.05)
            except Exception:
                pass
            engine = (
                self.tui.engine.engine
                if (self.tui.engine is not None and hasattr(self.tui.engine, "engine"))
                else self.tui.engine
            )
            if engine is None:
                self.app.call_from_thread(
                    self._on_server_restart_failed,
                    "Backend couldn't initialize. Check ~/.localcode/last_error.log.",
                )
                return
            try:
                ok = engine._restart_server()
            except Exception as e:
                self.app.call_from_thread(
                    self._on_server_restart_failed, f"Restart failed: {e}"
                )
                return
            if ok:
                self.app.call_from_thread(self._on_server_ready, reason)
            else:
                # Show the user the LAST lines of server.log so they
                # have a real reason instead of the generic "didn't come
                # back". Empirically this surfaces jetsam OOMs, port-
                # conflicts, and corrupt-config-flag errors directly to
                # the chat log.
                hint = "Server didn't come back up after restart."
                try:
                    from pathlib import Path as _P
                    log_p = _P.home() / ".local" / "share" / "localcode" / "server.log"
                    if log_p.is_file():
                        tail = log_p.read_text(errors="replace").splitlines()[-6:]
                        if tail:
                            hint += "\nlast server.log lines:\n  " + "\n  ".join(tail)
                except Exception:
                    pass
                hint += "\nType /restart to try again, or /status to inspect."
                self.app.call_from_thread(self._on_server_restart_failed, hint)

        self.run_worker(_worker, thread=True, exclusive=False)

    def _handle_vision_command(self, text: str) -> None:
        """`/vision` toggles vision on/off for the current model.

        Same UX as /voice — bare command toggles; first-time-on does
        the one-time mmproj download with progress. Power users still
        have /vision setup and /vision status for explicit control.
        """
        from ...models_catalog import current as current_choice
        log = self.query_one("#chat-log", ChatLog)
        config = self.tui.config
        choice = current_choice(config)
        if choice is None:
            log.append_info("No model selected — pick one via /model first.")
            return
        if not choice.supports_vision:
            log.append_info(
                f"{choice.name} doesn't support vision. Use a Gemma 4 or Qwen 3.6 model."
            )
            return
        # Hardware capability gate
        try:
            from ...voice import detect_vision_capability
            ok, hint = detect_vision_capability()
            if not ok:
                log.append_error(f"Vision unavailable: {hint}")
                return
        except Exception:
            pass
        mmproj = choice.mmproj_path
        has = bool(mmproj and mmproj.is_file())

        parts = text.strip().split()
        sub = parts[1] if len(parts) >= 2 else None

        # ── bare /vision → toggle ─────────────────────────────────
        if sub is None:
            if has:
                # Currently "on" → remove projector + restart server so
                # it relaunches without --mmproj. Frees RAM immediately.
                try:
                    mmproj.unlink()
                except Exception as e:
                    log.append_error(f"Couldn't remove projector: {e}")
                    return
                log.append_info("Vision OFF — restarting server to release projector memory…")
                self._restart_for_vision_change(reason="Vision OFF")
                return
            # Currently "off" → download + activate (async — keeps UI live)
            if getattr(self, "_vision_download_in_flight", False):
                log.append_info("Vision projector is already downloading — please wait.")
                return
            self._vision_download_in_flight = True
            log.append_info(
                f"Vision mode needs the projector (~{choice.mmproj_size_gb:.1f} GB) "
                f"for {choice.name}. Downloading in background — UI stays responsive."
            )

            def _progress_from_thread(msg: str) -> None:
                try:
                    if msg.startswith("Downloading:"):
                        self.app.call_from_thread(self._set_download_line, msg)
                    else:
                        self.app.call_from_thread(log.append_info, msg)
                except Exception:
                    pass

            def _worker() -> None:
                from ...bootstrap import download_mmproj as _dl
                ok, result = _dl(choice, on_progress=_progress_from_thread)
                self.app.call_from_thread(
                    self._finish_vision_download, ok, result, choice.key,
                )

            import threading as _t
            _t.Thread(target=_worker, daemon=True).start()
            return

        # ── power-user subcommands ───────────────────────────────
        if sub == "status":
            if has:
                mb = mmproj.stat().st_size // (1024 * 1024)
                log.append_info(f"Vision: on · {mmproj.name} · {mb} MB")
            else:
                log.append_info(
                    f"Vision: off · type /vision to enable "
                    f"(~{choice.mmproj_size_gb:.1f} GB)"
                )
            return

        if sub in ("on", "download", "setup"):
            if has:
                log.append_info("Vision already on.")
                return
            self._handle_vision_command("/vision")
            return

        if sub == "off":
            if has:
                self._handle_vision_command("/vision")
            else:
                log.append_info("Vision already off.")
            return

        log.append_info("Usage: /vision  (toggle)  ·  /vision status")

    def _handle_model_command(self, text: str) -> None:
        """Handle /model — open the visual picker or switch directly by key."""
        from ...models_catalog import CHOICES, by_key, current
        log = self.query_one("#chat-log", ChatLog)
        config = self.tui.config
        cur = current(config)
        parts = text.strip().split(maxsplit=1)

        # /model <key> — direct switch (keeps muscle-memory usage fast)
        if len(parts) == 2:
            key = parts[1].strip()
            choice = by_key(key)
            # Allow numeric selection too (/model 2)
            if choice is None and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(CHOICES):
                    choice = CHOICES[idx]
            if choice is None:
                valid = ", ".join(c.key for c in CHOICES)
                log.append_error(f"Unknown model '{key}'. Valid: {valid}")
                return
            self._apply_model_choice(choice)
            return

        # Bare /model — open the visual picker
        def _on_pick(choice):
            if choice is None:
                log = self.query_one("#chat-log", ChatLog)
                log.append_info("Model switch cancelled.")
                return
            self._apply_model_choice(choice)

        self.app.push_screen("model_picker", _on_pick)

    def _apply_model_choice(self, choice) -> None:
        """Persist a model choice, then restart the llama-server with the new
        model in a worker (so the TUI stays responsive) and healthcheck before
        letting the next request fire.
        """
        from ...models_catalog import current
        from ...config import save_config
        log = self.query_one("#chat-log", ChatLog)
        config = self.tui.config
        cur = current(config)
        if cur is not None and cur.key == choice.key:
            log.append_info(f"Already using {choice.name}.")
            return
        needs_download = not choice.local_path.is_file()
        config.runtime.model = str(choice.local_path)
        try:
            save_config(config)
        except Exception as e:
            log.append_error(f"Saved in-memory but couldn't persist config.toml: {e}")
        # Also update the runtime's in-memory config + engine gateway so the
        # next request reads the new model path, not the old one cached at
        # LocalCodeApp construction time.
        if self.tui.engine is not None:
            try:
                self.tui.engine.config.runtime.model = str(choice.local_path)
                self.tui.engine.runtime_model = str(choice.local_path)
                self.tui.engine.session.model = str(choice.local_path)
                self.tui.engine.engine.config.model = str(choice.local_path)
            except Exception:
                pass
        if needs_download:
            log.append_info(
                f"Downloading {choice.name} ({choice.size_gb:.1f} GB), then restarting server..."
            )
        else:
            log.append_info(f"Switching to {choice.name} — restarting server...")
        # Block input submission until the new server is fully loaded.
        # Anything the user types during the restart window is queued
        # (see on_input_submitted) and drained by `_on_server_ready`
        # once /health returns 200. Without this, messages typed during
        # the 20-40 s model load hit the still-loading server and come
        # back as 503 "Loading model" → user-facing E3102.
        self._server_restarting = True
        self._update_status()

        def _worker() -> None:
            if needs_download:
                try:
                    from ...bootstrap import download_model

                    ok, result = download_model(choice)
                except Exception as e:
                    self.app.call_from_thread(
                        self._on_server_restart_failed,
                        f"Model download failed: {e}",
                    )
                    return
                if not ok:
                    self.app.call_from_thread(
                        self._on_server_restart_failed,
                        str(result),
                    )
                    return
            engine = self.tui.engine.engine if self.tui.engine is not None else None
            if engine is None:
                self.app.call_from_thread(
                    self._on_server_restart_failed,
                    "Backend not initialized — can't restart server. Type a message to trigger a fresh start.",
                )
                return
            # _restart_server now returns the healthcheck result that
            # ServerManager.restart already waits for (up to 120 s for a
            # 10 GB model cold-load). The previous code threw away that
            # result and ran its own immediate healthcheck, which always
            # failed — hence the spurious "Server didn't come up in
            # time" toast even when the swap had succeeded.
            try:
                ok = engine._restart_server()
            except Exception as e:
                self.app.call_from_thread(
                    self._on_server_restart_failed, f"Server restart failed: {e}"
                )
                return
            if ok:
                self.app.call_from_thread(self._on_server_ready, choice.name)
            else:
                from ...errors import LocalCodeError, by_code
                code = by_code("E1002")
                msg = (
                    str(LocalCodeError(code=code, detail=choice.name))
                    if code is not None
                    else "Server didn't come up in time."
                )
                self.app.call_from_thread(self._on_server_restart_failed, msg)

        self.run_worker(_worker, thread=True, exclusive=False)

    def _on_server_ready(self, model_name: str) -> None:
        """UI-thread handler: restart succeeded, server is serving requests.

        Clears `_server_restarting` and drains any input the user typed
        while waiting. The drain path mirrors the end-of-turn drain in
        `_finish_turn`: pop the first queued message, append it as a
        user bubble, and kick off a real agent turn. Remaining queued
        messages wait for that turn to finish, same as normal.
        """
        self._server_restarting = False
        log = self.query_one("#chat-log", ChatLog)
        log.append_info(f"Server ready with {model_name}.")
        self._update_status()
        if self._pending_messages:
            next_msg = self._pending_messages.pop(0)
            self._update_queue()
            log.append_user(next_msg)
            log.scroll_end(animate=False)
            self._start_turn(next_msg)

    def _on_server_restart_failed(self, error_msg: str) -> None:
        """UI-thread handler: restart timed out or blew up.

        Clears `_server_restarting` so input is no longer blocked —
        the user may want to /model to a different model, or retry.
        Any messages queued during the restart are dropped with a
        breadcrumb so the user knows they weren't sent.
        """
        self._server_restarting = False
        log = self.query_one("#chat-log", ChatLog)
        log.append_error(error_msg)
        dropped = len(self._pending_messages)
        if dropped:
            self._pending_messages.clear()
            self._update_queue()
            log.append_info(
                f"Dropped {dropped} queued message{'s' if dropped != 1 else ''} — "
                f"server didn't come up. Re-type if you still want to send."
            )
        self._update_status()

    def _kick_backend_wait(self) -> None:
        """Poll llama-server's /health silently until it answers, then
        lazy-init the engine via `ensure_backend()` once. Drains the
        queue via `_on_server_ready`. Used on cold-start when the user
        types before llama-server has finished loading.

        Why probe `is_healthy()` directly instead of calling
        `ensure_backend()` in the loop: `ensure_backend()` toasts
        `notify("Backend error: …", severity="error")` on every
        failure, so a 1-Hz retry over 120 s would spam ~120 toasts.
        `ServerManager.is_healthy()` is a non-blocking, side-effect-
        free probe — exactly what a watcher loop wants.
        """
        import time as _time

        def _worker() -> None:
            try:
                from ...server_manager import ServerManager as _SM
                mgr = _SM.get()
            except Exception:
                mgr = None
            deadline = _time.time() + 120.0
            while _time.time() < deadline:
                if mgr is not None and mgr.is_healthy():
                    self.app.call_from_thread(self._finalize_backend_ready)
                    return
                _time.sleep(1.0)
            self.app.call_from_thread(
                self._on_server_restart_failed,
                "Backend didn't come up within 120 s — check `~/.localcode/logs/stderr.log`.",
            )

        self.run_worker(_worker, thread=True, exclusive=False)

    def _finalize_backend_ready(self) -> None:
        """UI-thread handler: server /health is OK, do the one-shot
        engine init and drain the queue. Mirrors `_on_server_ready`'s
        drain path but uses the actual configured model name.
        """
        if not self.tui.ensure_backend():
            # Server says healthy but engine init still threw. Treat
            # the same as a restart failure so queued messages aren't
            # silently lost.
            self._on_server_restart_failed(
                "Server is up but engine init failed. Try `/model` to pick a different model."
            )
            return
        cfg = getattr(self.tui, "config", None)
        rt = getattr(cfg, "runtime", None) if cfg else None
        model_name = getattr(rt, "model", None) or "model"
        self._on_server_ready(model_name)

    # ── Search ──

    def action_ptt_space(self) -> None:
        """Space handler with dual behavior:

        * Voice mode OFF → insert a space character into the focused input
          box (preserves normal typing).
        * Voice mode ON, not recording → start recording + spawn the
          streaming transcription loop (re-transcribes every 1.5 s so
          text appears in the input box while the user is still talking).
          Track the press timestamp.
        * Voice mode ON, recording → treat as a key-repeat (user is
          still holding Space). Reset `_ptt_last_key_ts`. The watchdog
          stops the recording when no Space event arrives for 350 ms,
          which is how we detect "released" in a terminal that can't
          send key-release events.
        """
        # Lazy-create voice state
        if not hasattr(self.tui, "voice_state"):
            from ...voice import VoiceState as _VS
            self.tui.voice_state = _VS()
        state = self.tui.voice_state

        # ── Voice OFF: space is just a space ────────────────────
        if not state.enabled:
            try:
                inp = self.query_one("#chat-input", Input)
                inp.insert_text_at_cursor(" ")
            except Exception:
                pass
            return

        import time as _t
        now = _t.time()

        # ── Voice ON, recording: this is a key-repeat — reset watchdog ─
        if getattr(self, "_ptt_recorder", None) is not None:
            self._ptt_last_key_ts = now
            return

        # ── Voice ON, not recording: start ──────────────────────
        log = self.query_one("#chat-log", ChatLog)
        from ...voice import Recorder
        try:
            rec = Recorder(state)
            rec.start()
            self._ptt_recorder = rec
            # Session counter — increments per recording so in-flight
            # streaming workers from a PREVIOUS session can detect they
            # were orphaned (and not overwrite the input box that the
            # user is now typing into).
            self._ptt_session = getattr(self, "_ptt_session", 0) + 1
            self._ptt_start_ts = now
            self._ptt_last_key_ts = now
            # Save what's currently in the input — could be user-typed
            # text, a previous transcription, or both. Streaming /
            # final transcripts APPEND to this prefix so consecutive
            # recordings stack ("hello" + hold-Space + "how are you"
            # → "hello how are you") instead of overwriting.
            try:
                existing = self.query_one("#chat-input", Input).value or ""
                # Ensure exactly one trailing space so the join is clean.
                self._ptt_input_prefix = existing.rstrip() + " " if existing.strip() else ""
            except Exception:
                self._ptt_input_prefix = ""
            # Don't log "Recording — release Space" anymore — it spammed
            # the chat log if the watchdog mis-fired. The visualizer bar
            # next to the input is now the only recording indicator.
            # Visualizer widget — colored block right of the input,
            # animated by its own 30 FPS timer.
            try:
                from ..widgets.voice_visualizer import VoiceVisualizer
                vis = self.query_one("#voice-visualizer", VoiceVisualizer)
                vis.activate(rec)
            except Exception:
                pass

            def _hold_watchdog() -> None:
                """Detect Space release in a terminal that doesn't send key-up
                events. We watch the gap between consecutive Space events.

                Three-phase logic to avoid the false-stop bug where the
                watchdog fired BEFORE the first key-repeat could arrive:

                1. WAIT_FOR_FIRST_REPEAT (0 to ~1.2 s): hold open. macOS
                   default initial-key-repeat delay is 500 ms but it's
                   user-configurable up to 2 s. We give it 1.2 s of grace.
                   During this window, EITHER a repeat lands (→ phase 2)
                   OR no repeat lands (→ user tapped, fall back to
                   silence-auto-stop, phase 3).
                2. STEADY_HOLD: repeats arriving regularly (every ~33 ms
                   after initial). Stop when gap > 350 ms.
                3. TAP_FALLBACK: no repeat arrived — user tapped Space.
                   Stop on 1.5 s of silence (from the Recorder).
                """
                r = self._ptt_recorder
                if r is None:
                    return
                now = _t.time()
                elapsed = now - self._ptt_start_ts
                # We learn the user is HOLDING when last_key_ts advances
                # past start_ts (a key-repeat arrived).
                got_repeat = self._ptt_last_key_ts > self._ptt_start_ts + 0.01

                # Phase 1: wait up to 1.2 s for the first repeat
                if not got_repeat and elapsed < 1.2:
                    return
                # Phase 3: user tapped, no hold → use silence as stop
                if not got_repeat:
                    try:
                        if r.silence_seconds > 1.5:
                            self._ptt_stop_and_finalize()
                    except Exception:
                        pass
                    return
                # Phase 2: steady hold — release when gap > 350 ms
                gap = now - self._ptt_last_key_ts
                if gap > 0.35:
                    self._ptt_stop_and_finalize()
            self._ptt_hold_timer = self.set_interval(0.08, _hold_watchdog)

            # Streaming transcription loop — every 1.5 s, snapshot the
            # audio captured so far, transcribe it on a worker thread,
            # and push the result to the input box.
            self._ptt_streaming_busy = False
            def _stream_tick() -> None:
                r = self._ptt_recorder
                if r is None or self._ptt_streaming_busy:
                    return
                snap = r.snapshot_wav()
                if snap is None:
                    return
                self._ptt_streaming_busy = True
                session_at_start = self._ptt_session
                def _worker():
                    try:
                        from ...voice import transcribe as _trans
                        ok, text = _trans(state, snap)
                        if ok and text:
                            # Capture session at worker spawn; only apply
                            # if the recording is still ours. Stops in-
                            # flight workers from an old recording from
                            # overwriting the input the user is now
                            # typing into.
                            self.app.call_from_thread(
                                self._apply_partial_transcript, text, session_at_start
                            )
                    finally:
                        try:
                            snap.unlink(missing_ok=True)
                        except Exception:
                            pass
                        self._ptt_streaming_busy = False
                import threading as _t2
                _t2.Thread(target=_worker, daemon=True).start()
            # 0.35 s re-decode for true word-by-word feel. Whisper
            # distil-medium.en runs ~15× realtime on M5 Max so re-decoding
            # ≤10 s of audio takes ~700 ms — still under the next tick
            # because of the busy guard.
            self._ptt_stream_timer = self.set_interval(0.35, _stream_tick)
        except Exception as e:
            log.append_error(f"Couldn't start mic: {e}")
            # Disable voice mode so subsequent Space presses don't
            # spam the same error 6 times in a row. User can /voice
            # again to re-enable once they've fixed the underlying
            # issue (granted permission, plugged in mic, etc.).
            state.enabled = False
            log.append_info(
                "Voice mode turned OFF. Fix the issue above, then "
                "type /voice to re-enable."
            )

    def _apply_partial_transcript(self, text: str, session: int = -1) -> None:
        """Push a partial transcript into the input field.

        Word-by-word stream: compare to last applied transcript, find
        the longest common prefix in word units, and only APPEND the
        new tail words. This stops the visible "rewind and rewrite"
        feel where each tick would replace the entire string and the
        cursor would briefly jump backwards.

        `session` is the recording id this transcript was produced for.
        Stale workers from a previous recording carry an old id; if it
        doesn't match the current `_ptt_session`, drop silently.
        """
        if session >= 0 and session != getattr(self, "_ptt_session", -1):
            return
        try:
            inp = self.query_one("#chat-input", Input)
            prefix = getattr(self, "_ptt_input_prefix", "") or ""
            # Compare to whatever we last wrote during THIS session
            # (per-session so a new recording starts fresh).
            last_full = getattr(self, "_ptt_last_transcript", "")
            last_session = getattr(self, "_ptt_last_transcript_session", -2)
            if last_session != getattr(self, "_ptt_session", -1):
                last_full = ""
            # Word-level diff: find common leading words, only append tail.
            old_words = last_full.split()
            new_words = text.split()
            i = 0
            while i < len(old_words) and i < len(new_words) and old_words[i] == new_words[i]:
                i += 1
            # New tail = words we haven't shown yet.
            tail = " ".join(new_words[i:])
            self._ptt_last_transcript = text
            self._ptt_last_transcript_session = getattr(self, "_ptt_session", -1)
            joined = (prefix + text).strip() if prefix else text
            # Single authoritative write each tick — `joined` is what
            # the input box should hold right now. The diff above is
            # purely so we COULD do something fancier (e.g. cursor
            # animation) on `tail` if we wanted; setting the full
            # `joined` matches what's been transcribed so far.
            inp.value = joined
            inp.cursor_position = len(joined)
            inp.focus()
        except Exception:
            pass

    def _ptt_stop_and_finalize(self) -> None:
        """Called when the hold watchdog detects Space was released, or
        from action_ptt_cancel when user hits Esc. Tears down timers,
        stops the recorder, does one final transcription pass so the
        input field has the complete text, then cleans up."""
        rec = getattr(self, "_ptt_recorder", None)
        if rec is None:
            return
        # Tear down timers immediately so they don't re-fire mid-finalize
        for attr in ("_ptt_hold_timer", "_ptt_stream_timer", "_ptt_cursor_timer"):
            try:
                t = getattr(self, attr, None)
                if t is not None:
                    t.stop()
            except Exception:
                pass
            setattr(self, attr, None)
        # Tear down visualizer widget.
        try:
            from ..widgets.voice_visualizer import VoiceVisualizer
            vis = self.query_one("#voice-visualizer", VoiceVisualizer)
            vis.deactivate()
        except Exception:
            pass
        try:
            wav_path = rec.stop()
        except Exception:
            wav_path = None
        self._ptt_recorder = None
        if wav_path is None:
            return
        # Final transcription on the full audio (overrides any streaming
        # partial). Done on a worker so we don't block the UI thread.
        # We pass the FINAL session (current session id) so the apply
        # only lands if the user hasn't already started a new recording.
        state = self.tui.voice_state
        session_final = getattr(self, "_ptt_session", 0)
        def _final_worker():
            try:
                from ...voice import transcribe as _trans
                ok, text = _trans(state, wav_path)
                if ok and text:
                    self.app.call_from_thread(
                        self._apply_partial_transcript, text, session_final
                    )
            finally:
                try:
                    from pathlib import Path as _P
                    _P(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass
        import threading as _t3
        _t3.Thread(target=_final_worker, daemon=True).start()

    def action_ptt_cancel(self) -> None:
        """Cancel an active recording — throws away audio, no transcription.

        Only fires when a recording is actually in flight. When idle,
        Esc keeps its normal behavior (e.g. closing the search bar)
        because we early-return without consuming the event.
        """
        if not getattr(self, "_ptt_recorder", None):
            return  # let other Esc handlers run
        log = self.query_one("#chat-log", ChatLog)
        # Tear down watchdog + visualizer + streaming timer
        for attr in ("_ptt_silence_timer", "_ptt_hold_timer", "_ptt_stream_timer", "_ptt_cursor_timer"):
            try:
                t = getattr(self, attr, None)
                if t is not None:
                    t.stop()
            except Exception:
                pass
            setattr(self, attr, None)
        # Tear down visualizer widget.
        try:
            from ..widgets.voice_visualizer import VoiceVisualizer
            vis = self.query_one("#voice-visualizer", VoiceVisualizer)
            vis.deactivate()
        except Exception:
            pass
        # Stop the recorder, discard the wav
        try:
            wav_path = self._ptt_recorder.stop()
            if wav_path is not None:
                from pathlib import Path as _P
                try:
                    _P(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass
        self._ptt_recorder = None
        log.append_info("Recording cancelled.")

    def action_toggle_search(self) -> None:
        """Toggle the search bar with Ctrl+F."""
        search_input = self.query_one("#search-input", Input)
        search_bar = self.query_one("#search-bar", Static)
        if self._search_active:
            self._search_active = False
            search_input.remove_class("active")
            search_bar.remove_class("active")
            search_bar.update("")
            search_input.value = ""
            self._search_results = []
            self._search_idx = 0
            self.query_one("#chat-input", Input).focus()
        else:
            self._search_active = True
            search_input.add_class("active")
            search_bar.add_class("active")
            search_bar.update("[dim]Ctrl+F close · Enter next · Shift+Enter prev[/]")
            search_input.focus()

    def _do_search(self, query: str) -> None:
        """Run search and show results."""
        log = self.query_one("#chat-log", ChatLog)
        bar = self.query_one("#search-bar", Static)
        self._search_results = log.search_text(query)
        self._search_idx = 0
        if not query:
            bar.update("[dim]Ctrl+F close · Enter next · Shift+Enter prev[/]")
            return
        n = len(self._search_results)
        if n == 0:
            bar.update(f"[dim]No results for[/] [bold]\"{query}\"[/]")
        else:
            self._show_search_result()

    def _show_search_result(self) -> None:
        """Display current search result and scroll to it."""
        n = len(self._search_results)
        if n == 0:
            return
        idx = self._search_idx % n
        hist_idx, snippet = self._search_results[idx]
        bar = self.query_one("#search-bar", Static)
        # Escape markup in snippet
        safe_snippet = snippet.replace("[", "\\[")
        bar.update(f"[dim]{idx + 1}/{n}[/]  {safe_snippet}")
        log = self.query_one("#chat-log", ChatLog)
        log.scroll_to_history_index(hist_idx)

    # ── Agent turn (uses full agent loop with tool execution) ──

    def _start_turn(self, text: str) -> None:
        if not self.tui.ensure_backend():
            # Cold-start race: server is still loading, so backend init
            # 503'd. Don't drop the message — queue it, flip the restart
            # flag so any further input also queues, and spin a worker
            # that retries until the backend is ready (or times out).
            # Drains via the same `_on_server_ready` path the model
            # switch uses.
            log = self.query_one("#chat-log", ChatLog)
            self._pending_messages.insert(0, text)
            if not self._server_restarting:
                self._server_restarting = True
                self._update_status()
                log.append_info(
                    "Server still loading — message queued, will send when ready."
                )
                self._kick_backend_wait()
            self._update_queue()
            return
        self._agent_busy = True
        self._stream_buf.clear()
        self._last_assistant_text = ""
        self._tools_used.clear()
        self._turn_start = time.time()
        # Reset the cancel flag — last turn may have ended via user cancel.
        if self.tui.engine is not None:
            self.tui.engine.cancel_requested = False
        # Open a new telemetry trace for this turn — mutated by
        # on_agent_event as tool calls fire, written to JSONL at
        # turn end in _on_turn_done.
        from ...telemetry import TurnTrace, current_git_branch, current_app_version
        cfg = getattr(self.tui, "config", None)
        rt = getattr(cfg, "runtime", None) if cfg else None
        cwd_str = str(Path.cwd())
        self._telemetry_turn = TurnTrace(
            session_id=self._telemetry_session_id,
            user_message=text,
            cwd=cwd_str,
            git_branch=current_git_branch(cwd_str),
            app_version=current_app_version(),
            model=getattr(rt, "model", "") if rt else "",
            mode=getattr(rt, "laptop_26b_runtime_mode", "") if rt else "",
            context_window=self._context_max,
        )
        # Single-source accounting — recompute from current session state.
        # (Used to add a delta here, then overwrite later, then add again —
        # triple-counted and still missed the system prompt. Now: one path.)
        self._context_used = self._recompute_context_used()
        log = self.query_one("#chat-log", ChatLog)
        log.reset_steps()
        self._show_thinking()
        self._update_status()
        self.run_agent_turn(text)

    @work(exclusive=True, thread=True)
    def run_agent_turn(self, user_text: str) -> None:
        """Run FULL agent loop on background thread.

        Uses LocalCodeApp.ask() which handles:
        - Context gathering (repo structure, retrieval, cartridge)
        - System prompt composition
        - Full agent loop with tool execution (read, write, bash, grep, etc.)
        - Multi-round tool calls until model is done
        - All events emitted via OutputManager → bridge → TUI
        """
        if not self.tui.engine:
            return
        app = self.tui.engine
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
            # Wrap unknown exceptions in the error-code system so the
            # user sees `[E9001] Unhandled exception... — fix: ...`
            # instead of a bare opaque string. The formatter preserves
            # the exception type + message in the detail field so we
            # don't lose any diagnostic info.
            from ...errors import format_for_user
            bridge.on_event("error", message=format_for_user(e, fallback_code="E9001"))
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

        # Notification sound — opt-in via /sounds. Non-blocking afplay;
        # silent no-op when disabled / non-mac / missing system sound.
        try:
            from ...sounds import play_completion
            play_completion(self.tui.config.ui.sounds_enabled)
        except Exception:
            pass

        # TTS — opt-in via /audio (off|final|always). Independent of
        # /voice — TTS uses macOS `say`, no Whisper dependency. Runs
        # in a background thread so the turn doesn't pause for audio.
        try:
            vs = getattr(self.tui, "voice_state", None)
            if vs is not None and vs.tts_speak_mode != "off":
                spoken = (self._last_assistant_text or "").strip()
                if spoken:
                    from ...voice import speak as _speak
                    _speak(spoken, vs)
        except Exception:
            pass

        log = self.query_one("#chat-log", ChatLog)

        # Capture the final assistant text BEFORE clearing _stream_buf —
        # telemetry needs it for the trace record.
        final_assistant_text = "".join(self._stream_buf) or self._last_assistant_text

        # Finalize any remaining streamed content
        if not self._response_shown:
            log.finish_stream(final_assistant_text)

        self._stream_buf.clear()

        # Telemetry: write the completed turn to JSONL. Wrapped in try
        # because we never want logging to break a chat turn.
        if self._telemetry_turn is not None:
            try:
                from ...telemetry import log_turn
            except Exception:
                log_turn = None  # type: ignore[assignment]

        # Turn summary — PER-TURN values now (was cumulative across
        # session). User feedback: cumulative numbers were confusing
        # ("why does the second turn say 1.4k when I only got 600
        # tokens?"). Per-turn answers "what did THIS response cost."
        # Real token counts come from llama-server's `usage` field via
        # the turn_tokens event; fall back to streamed-char estimate
        # when usage isn't populated by the backend.
        elapsed = time.time() - self._turn_start
        # Prefer real completion tokens from llama-server; fall back to
        # the streaming char-count estimate if backend didn't report.
        out_tokens = self._turn_completion_tokens or self._turn_tokens
        in_tokens = self._turn_prompt_tokens
        total_tokens = self._turn_total_tokens or (
            in_tokens + out_tokens if (in_tokens or out_tokens) else 0
        )
        self._total_tokens += total_tokens
        log.append_turn_summary(
            elapsed, self._tools_used,
            tokens_in=in_tokens,
            tokens_out=out_tokens,
            tokens_total=total_tokens,
        )
        if self._telemetry_turn is not None and log_turn is not None:
            try:
                self._telemetry_turn.assistant_message = final_assistant_text
                self._telemetry_turn.thinking_text = self._thinking_text
                self._telemetry_turn.tokens_in = in_tokens
                self._telemetry_turn.tokens_out = out_tokens
                self._telemetry_turn.tokens_total = total_tokens
                log_turn(self._telemetry_turn)
            except Exception:
                pass
            self._telemetry_turn = None
        # Reset per-turn token counters for the next turn.
        self._turn_prompt_tokens = 0
        self._turn_completion_tokens = 0
        self._turn_total_tokens = 0
        # Refresh the context counter from current session state.
        # Automatically picks up anything compaction shrank.
        self._context_used = self._recompute_context_used()
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
            # Stream tokens to display in real time
            log.stream_token(chunk)
            toks = max(1, len(chunk) // 4)
            self._turn_tokens += toks
            # Don't add to _context_used here — the authoritative value
            # comes from _recompute_context_used() at turn-start and
            # turn-end. Mid-stream delta would drift from reality.
            self._thinking_phase = "generating"
        elif t == "response_done":
            # Finalize streaming display — text was already shown line by line
            full_text = "".join(self._stream_buf)
            self._last_assistant_text = full_text
            log.finish_stream(full_text)
            # Rewrite the just-streamed assistant text through the Markdown
            # renderer. `stream_token` writes raw characters (fast path,
            # no per-token Markdown parse), so during the turn the user
            # sees literal `**bold**`, ungrouped `1.` lists, and raw
            # backticks. `_render_assistant` (used by `_rerender`) runs
            # the same text through `rich.markdown.Markdown` — bold
            # renders, lists indent, code blocks get syntax highlighting.
            # Calling `_rerender` here applies that treatment the moment
            # the turn ends, instead of waiting for the next user input /
            # resize to incidentally trigger a rerender (which is what
            # caused the "rendering only works after I type again" bug).
            # Safe to call now: stream is no longer in progress, so the
            # "don't rerender mid-stream" guard in `on_resize` doesn't
            # apply — and `finish_stream` already recorded this round
            # into `_history`, which `_rerender` replays.
            log._rerender()
            self._response_shown = True
            self._stream_buf.clear()
            # Refresh the context counter on every round_end so the
            # status bar reflects the conversation growing during a
            # multi-round turn. Earlier this only updated on turn_start
            # / turn_end, which meant `context: 99% free` stayed pinned
            # at its turn-start value for the entire build — through
            # 5+ min app.py writes — and only ticked when the user
            # submitted again. _recompute_context_used reads
            # session.messages, which has already been updated with
            # this round's assistant message + any tool_results.
            self._context_used = self._recompute_context_used()
            self._update_status()
        elif t == "turn_tokens":
            # Real prompt/completion token counts forwarded from
            # runtime.py via output.update_turn_tokens. Accumulate per
            # turn (multi-tool rounds fire this multiple times) and
            # let the response_done handler read them for the summary.
            try:
                self._turn_prompt_tokens += int(p.get("prompt_tokens", 0) or 0)
                self._turn_completion_tokens += int(p.get("completion_tokens", 0) or 0)
                total = int(p.get("total_tokens", 0) or 0)
                if total > 0:
                    self._turn_total_tokens += total
            except (TypeError, ValueError):
                pass
        elif t == "tool_preview":
            # Mid-stream signal: model has committed to a tool name and is
            # streaming its arguments. Show the floating header right now so
            # the user knows what's coming, instead of staring at a generic
            # "thinking…" indicator while a 5K-line file write streams.
            # The real `tool_start` arrives after the full call is parsed
            # and replaces this preview with the final args.
            name = p.get("name", "")
            chars = p.get("chars", "0")
            snippet = p.get("snippet", "") or ""
            tool_idx = p.get("index", -1)
            try:
                size_kb = int(chars) // 1024
            except (TypeError, ValueError):
                size_kb = 0
            # Live output-token estimate: tool args stream is part of
            # the model's decode output. `chars` is cumulative for this
            # tool_idx, so add the delta only.
            try:
                cur_chars = int(chars)
                idx_key = int(tool_idx)
                seen = self._tool_args_seen.get(idx_key, 0)
                if cur_chars > seen:
                    self._turn_tokens += (cur_chars - seen) // 4
                    self._tool_args_seen[idx_key] = cur_chars
            except (TypeError, ValueError):
                pass
            # Pull the primary identifier from the partial JSON args. Tool
            # schemas put `path` / `url` / `command` / `pattern` first so
            # the leading bytes almost always contain a complete
            # `"key": "value"` pair we can regex-extract before the bulky
            # `content` field starts streaming.
            primary = ""
            if snippet:
                for key in ("path", "url", "command", "pattern", "query"):
                    # Match either a complete `"key": "value"` pair OR a
                    # partial value where the closing quote hasn't streamed
                    # yet (common for URLs that exceed our snippet window).
                    m = re.search(rf'"{key}"\s*:\s*"([^"\\]{{1,120}})(?:"|$)', snippet)
                    if m:
                        primary = m.group(1)
                        if len(primary) > 50:
                            primary = primary[:47] + "…"
                        break
            if primary and size_kb >= 1:
                label = f"{primary}… {size_kb} KB"
            elif primary:
                label = f"{primary}…"
            elif size_kb >= 1:
                label = f"streaming… {size_kb} KB"
            else:
                label = "streaming…"
            self._show_active_step(name, label)
        elif t == "tool_start":
            name = p.get("name", "")
            args = p.get("args", "")
            # Commit any pending streamed text BEFORE the tool appears, so the
            # round's explanation renders as Markdown above the tool calls
            # (without this, between-round text gets buried under tool output)
            if hasattr(log, '_stream_started') and log._stream_started:
                log.finish_stream()
            # Add spacing before first tool in a turn
            if not self._tools_used:
                log.write(RichText(""))
            self._tools_used.append(name)
            self._thinking_phase = name
            # Only show in floating #active-step widget (NOT in chat log)
            self._show_active_step(name, args)
            # Telemetry: begin a tool-duration measurement for this call.
            if self._telemetry_turn is not None:
                self._telemetry_turn.tool_started(name, args)
        elif t == "tool_result":
            result = p.get("result", "")
            error = p.get("error", "")
            is_error = str(error).lower() == "true"
            # Hide floating animation
            self._hide_active_step()
            # Use name from event payload (reliable) instead of _active_tool_name (can go stale)
            name = p.get("name", "") or self._active_tool_name
            args = p.get("args", "") or self._active_tool_args
            # Always render ✓ green for success, ● + error for failure
            if is_error:
                log.append_tool(name, args)
                log.append_tool_result(result, error=True)
            else:
                lines = result.strip().splitlines()
                summary = lines[0][:80] if lines else ""
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
                    log.append_tool_done(name, args, summary)
            # Telemetry: close out the most recent tool-started event.
            if self._telemetry_turn is not None:
                self._telemetry_turn.tool_finished(len(str(result)), is_error)
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
            self._turn_tokens += max(1, len(chunk) // 4) if chunk else 0
            self._thinking_phase = "thinking"
            # Update active step with thinking preview
            preview = self._thinking_text.strip().replace("\n", " ")[:60]
            if preview:
                if self._active_mode != "thinking":
                    self._show_active_thinking(preview)
                else:
                    self._active_step_text = preview
                    self._tick_active()
        elif t == "thinking_peek":
            self._thinking_phase = "thinking"
            peek = (p.get("text", "") or "").strip().replace("\n", " ")[:60]
            if peek:
                if self._active_mode != "thinking":
                    self._show_active_thinking(peek)
                else:
                    self._active_step_text = peek
                    self._tick_active()
        elif t == "thinking_done":
            text = p.get("text", "")
            self._thinking_text = text
            self._hide_active_step()
            if text.strip():
                log.append_thinking(text, expanded=self._thinking_expanded)
        elif t == "stream_start":
            self._thinking_phase = "generating"
            # Show localcode-themed animation while generating response
            self._show_active_thinking("generating")
            # Hide any active tool step animation
            if self._active_mode == "tool":
                self._hide_active_step()
        elif t == "error":
            msg = p.get("message", "Unknown error")
            log.append_error(msg)
        elif t == "stage":
            stage = p.get("stage", "")
            if stage:
                self._thinking_phase = stage
                if getattr(self, "_last_announced_stage", "") != stage:
                    self._last_announced_stage = stage
                # Show animation for stage changes (between tool rounds)
                if not self._active_mode:
                    self._show_active_thinking(stage)
        elif t == "done":
            pass  # handled by on_worker_state_changed
        elif t == "_approval_request":
            log.append_approval(p.get("tool_name", ""), p.get("command", ""))
            self._awaiting_approval = True
            inp = self.query_one("#chat-input", Input)
            inp.disabled = True
            self.focus()

    # ── Tool approval (inline, like terminal coding tools) ──

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

        # Notification sound — opt-in via /sounds. Distinct tone from
        # completion so the user can tell ear-only that input is needed.
        try:
            from ...sounds import play_approval
            play_approval(self.tui.config.ui.sounds_enabled)
        except Exception:
            pass

    def on_key(self, event) -> None:
        """Handle slash menu navigation, search nav, and inline approval."""
        # Search navigation — Escape closes, Shift+Enter goes to previous result
        if self._search_active:
            if event.key == "escape":
                self.action_toggle_search()
                event.prevent_default()
                event.stop()
                return
            elif event.key == "shift+enter" and self._search_results:
                self._search_idx = (self._search_idx - 1) % len(self._search_results)
                self._show_search_result()
                event.prevent_default()
                event.stop()
                return

        # Slash menu navigation (arrow keys, tab, enter)
        if self._slash_matches:
            key = event.key
            if key in ("down", "tab"):
                self._slash_selected = (self._slash_selected + 1) % len(self._slash_matches)
                self._render_slash_menu()
                event.prevent_default()
                event.stop()
                return
            elif key == "up":
                self._slash_selected = (self._slash_selected - 1) % len(self._slash_matches)
                self._render_slash_menu()
                event.prevent_default()
                event.stop()
                return
            elif key == "enter":
                # Select the highlighted command
                cmd = self._slash_matches[self._slash_selected][0]
                inp = self.query_one("#chat-input", Input)
                inp.value = cmd
                inp.cursor_position = len(cmd)
                # Don't prevent default — let Input.Submitted fire
                return
            elif key == "escape":
                self._slash_matches = []
                self._slash_selected = 0
                self.query_one("#slash-menu", Static).remove_class("active")
                event.prevent_default()
                event.stop()
                return

        if not self._awaiting_approval:
            return
        key = event.key
        log = self.query_one("#chat-log", ChatLog)
        verdict: str | None = None
        label = ""
        if key in ("1", "y"):
            verdict, label = "once", "allowed"
        elif key == "2":
            verdict, label = "always", "always-allowed for this session"
        elif key in ("3", "n", "escape"):
            verdict, label = "deny", "denied"

        if verdict is not None:
            self._awaiting_approval = False
            log.append_info(f"  └ {label}")
            self.tui.bridge.set_approval(verdict)
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()

        # Block ALL other keys during approval
        event.prevent_default()
        event.stop()
