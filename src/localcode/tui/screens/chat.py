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
from ..paste_collapse import PasteBuffer, is_large_paste

# Sentinel: lets _model_supports_thinking accept a pre-resolved ModelChoice
# (the 2 s status tick already has one) vs. resolving it itself.
_UNSET = object()


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
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Input, Static, TextArea
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
        background: ansi_default;
        &:focus {
            background-tint: transparent 0%;
            background: ansi_default;
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
        # Preserve newlines so pasted code / JSON keeps its structure when
        # submitted (previously they were squashed to single spaces, which
        # destroyed any block the user pasted). The Input renders on one
        # line, but the screen's #input-overflow preview shows the full
        # multi-line value wrapped, and the submitted value carries the
        # real newlines through to the model. Normalize CRLF/CR → LF only.
        joined = text.replace("\r\n", "\n").replace("\r", "\n")
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
        # Simple rule: PTT only fires when input is EMPTY (or already
        # recording — key-repeat case). Once the user has typed/
        # dictated anything, Space types a space normally so they can
        # add words between things without hijacking. Previous "ends
        # with space" heuristic broke as soon as the user typed a word
        # like "capital" — pressing Space then triggered recording
        # instead of letting them continue the sentence.
        if event.key == "space":
            vs = getattr(self.app, "voice_state", None)
            if vs is not None and getattr(vs, "enabled", False):
                already_recording = getattr(
                    self.screen, "_ptt_recorder", None
                ) is not None
                input_empty = not (self.value or "").strip()
                # Voice-filled but untouched: user dictated, the
                # transcript landed in the input, and they haven't
                # typed since. Pressing Space again starts a new
                # recording that APPENDS — the alternative (typing
                # a literal space) wastes the only natural way to
                # continue a voice-only session without going to the
                # keyboard. Tracked via _ptt_last_input_value, which
                # is set in _apply_voice_transcript and cleared the
                # moment on_input_changed sees a divergence.
                last_voice_fill = getattr(self.screen, "_ptt_last_input_value", None)
                voice_filled_untouched = (
                    last_voice_fill is not None
                    and (self.value or "") == last_voice_fill
                )
                if already_recording or input_empty or voice_filled_untouched:
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

    @property
    def _active_cursor_glyph(self) -> str:
        # Cursor is ALWAYS the thin "▏" bar. The voice indicator is the
        # VoiceVisualizer sibling widget on the right of the input row —
        # NOT the Input's own cursor. Keeping the cursor thin removes
        # the "bottom-gray" artifact from partial-fill chars and avoids
        # a confusing second bar appearing in the middle of the text.
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
        # the cursor glyph. Then — IF recording — append an extra
        # colored "█" block to the end of the rendered text so the user
        # sees a pulsing bar right after the dictated text. We append
        # the bar BEFORE rendering to segments so its color sticks
        # (apply_style at the end only touches cells that don't already
        # have a style — our explicit color span does).
        cursor_style = self.get_component_rich_style("input--cursor")
        result = self._value.copy()
        if not self.selection.is_empty:
            start, end = self.selection
            start, end = sorted((start, end))
            result.stylize_before(
                self.get_component_rich_style("input--selection"), start, end,
            )
        # While recording, SUPPRESS the cursor glyph entirely. The
        # colored bar appended right after the text is the only
        # indicator. Otherwise users see "text + ▏ + colored bar" —
        # the thin cursor floats awkwardly between the dictation and
        # the bar. With this branch, layout is just "text + colored bar".
        recording_now = False
        try:
            recording_now = getattr(self.screen, "_ptt_recorder", None) is not None
        except Exception:
            pass
        if recording_now:
            pass  # no cursor cell — let the colored bar serve as caret
        elif cursor_pos >= len(result.plain):
            # Cursor past the last char — there's no glyph to highlight,
            # so append the thin bar caret to mark insertion point.
            result.append(self._active_cursor_glyph, style=cursor_style)
        else:
            # Cursor over an existing character. Previously we REPLACED
            # the character with the bar glyph (▏) so the cell read as
            # a caret — but Textual's blink toggle alternates between
            # this render path (glyph-replaced) and the stock render
            # path (original char). The user perceived the swap as
            # "cursor deletes the letter" because on every blink the
            # 'b' in 'alombasi' alternately vanished and reappeared.
            # Instead, KEEP the character visible and apply the cursor
            # style (usually reverse-video) to that cell — the cell
            # now reads as a block cursor that doesn't erase the glyph,
            # and the blink-off path renders the same char unchanged
            # so there's no visual jump.
            result.stylize(cursor_style, cursor_pos, cursor_pos + 1)
        # Voice recording? Append a colored block character right after
        # the text. NO background — the cell stays terminal-native so
        # the bar character IS the visual (no "gray box" feel from a
        # dim bg behind it). Amplitude uses a sqrt curve so quiet
        # speech registers as ▂/▃ instead of stuck on ▁.
        try:
            screen = self.screen
            rec = getattr(screen, "_ptt_recorder", None)
            if rec is not None:
                from rich.style import Style as _RichStyle
                import time as _t
                import math as _m
                _RAINBOW = (
                    "#ff5470", "#ff8a5b", "#ffd166", "#9bff8a", "#5bffc1",
                    "#5bd1ff", "#5b96ff", "#9b5bff", "#e75bff", "#ff5bd1",
                )
                _FILL = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
                peak = float(getattr(rec, "peak", 0.0) or 0.0)
                # sqrt curve + 14x boost makes the bar responsive at
                # whisper volumes. Earlier linear * 8x sat on ▁ until
                # the user nearly shouted.
                amp = min(1.0, _m.sqrt(max(0.0, peak * 14.0)))
                idx = max(1, min(len(_FILL) - 1, int(amp * len(_FILL))))
                bar_char = _FILL[idx]
                phase = int(_t.time() * 4) + int(amp * 4)
                color = _RAINBOW[phase % len(_RAINBOW)]
                bar_text = _RichText(bar_char, end="")
                bar_text.stylize(_RichStyle(color=color, bold=True), 0, 1)
                result = result + bar_text
        except Exception:
            pass
        segments = list(console.render(
            result, console_options.update_width(self.content_width),
        ))
        strip = Strip(segments)
        scroll_x, _ = self.scroll_offset
        strip = strip.crop(scroll_x, scroll_x + max_content_width + 1)
        strip = strip.extend_cell_length(max_content_width + 1)
        return strip.apply_style(self.rich_style)


class _ChatTextArea(TextArea):
    """Multi-line, scrollable chat input.

    Replaces the old single-line `_NoTintInput`. Standard chat submit
    semantics: plain **Enter submits** the message (routed to the
    screen's `_submit_message`), **Shift+Enter / Ctrl+J insert a
    newline**. TextArea's stock behaviour is Enter=newline, so we
    override it in `_on_key` below.

    The widget auto-grows with its content (CSS `height: auto`) up to a
    max height, after which it scrolls internally instead of pushing the
    rest of the layout. Arrow / Home / End / PageUp / PageDown all do
    real cursor + scroll navigation through long pasted content because
    that is TextArea's native behaviour — the old `#input-overflow`
    wrap-preview Static is gone (it only existed to fake multi-line for
    the single-line Input).

    Voice-recording indication: while the screen's `_ptt_recorder` is
    set we add the `recording` CSS class (red border, see screen CSS).
    The old inline colored-block glyph rendered in `Input.render_line`
    does not port cleanly to TextArea's document renderer, so we use a
    border-colour indicator instead.
    """

    # No `:focus` highlight; keep the terminal-native background and stay
    # full-brightness while disabled (during the approval prompt) — same
    # rationale as the old _NoTintInput CSS.
    # Force the caret to blink. TextArea defaults to blink=True, but making it
    # explicit guards against a theme/terminal resetting it.
    cursor_blink = True

    DEFAULT_CSS = """
    _ChatTextArea {
        background: ansi_default;
        /* Scrollbar shows once the input grows past its cap. Match the rest of
           the app's grey scrollbars (see styles/app.tcss) instead of a brand
           blue — otherwise it also inherits the textual-ansi theme's
           `scrollbar: ansi_blue` navy, which clashes. */
        scrollbar-color: #333333;
        scrollbar-color-hover: #555555;
        scrollbar-color-active: #666666;
        scrollbar-background: ansi_default;
        scrollbar-size-vertical: 1;
        & .text-area--cursor-line {
            background: ansi_default;
        }
        /* The caret. textual-ansi reports dark=False, so TextArea's own
           `&:light .text-area--cursor` rule (background: $foreground 70%) wins
           over a plain override and renders a dark/black block. Force it with
           !important: color+background = ansi_default (terminal fg/bg) plus
           `reverse` swaps them, giving a light block on a dark terminal and a
           dark block on a light one — terminal-adaptive, no hardcoded hex. */
        & .text-area--cursor {
            color: ansi_default !important;
            background: ansi_default !important;
            text-style: reverse !important;
        }
        &:focus {
            background-tint: transparent 0%;
            background: ansi_default;
        }
        &:disabled, &:disabled:can-focus {
            opacity: 1;
            background: ansi_default;
            background-tint: transparent 0%;
        }
    }
    """

    # ── value compatibility shim ──
    # The old call sites used Input's `.value`; TextArea exposes `.text`.
    # We migrated those call sites to `.text`, but keep a `value` alias so
    # any stray access keeps working and reads identically.
    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, new: str) -> None:
        self.text = new or ""

    def insert_text_at_cursor(self, text: str) -> None:
        """Input-API compatibility: TextArea uses `insert()`."""
        self.insert(text)
        self.autosize_height()

    # ── auto-grow (Codex-style) ──
    # Textual's TextArea does NOT grow from `height: auto` alone — it keeps a
    # fixed height and scrolls. So a big pasted prompt showed as a single row.
    # We drive the row count from the wrapped document: grow up to the cap,
    # then it scrolls internally instead of pushing the layout. Called on every
    # content change + on mount.
    #
    # The cap is terminal-relative, mirroring the reference Codex input
    # (`maxVisibleLines = max(MIN, floor(rows/2) - footer)`): roughly half the
    # terminal height, leaving room for the chat log + status bar, but never
    # smaller than _MIN_INPUT_CAP nor larger than _MAX_INPUT_CAP. On the typical
    # ~24-row terminal this lands at ~10 rows, matching the old fixed value; on
    # a tall window the box can grow further before it starts scrolling, and on
    # a short one it stays modest so it never swallows the conversation.
    _MIN_INPUT_CAP = 3   # floor — always show at least a few lines
    _MAX_INPUT_CAP = 20  # ceiling — don't let the box dominate a huge terminal
    _FOOTER_RESERVE = 6  # rows kept for status bar + active-step + breathing room

    def _max_input_lines(self) -> int:
        """Terminal-relative cap on the visible input height (Codex-style)."""
        try:
            term_rows = self.app.size.height
        except Exception:
            term_rows = 0
        if term_rows <= 0:
            # Pre-mount / unknown size — fall back to the historical fixed cap.
            return 10
        cap = (term_rows // 2) - self._FOOTER_RESERVE
        return max(self._MIN_INPUT_CAP, min(self._MAX_INPUT_CAP, cap))

    def autosize_height(self) -> None:
        try:
            rows = self.wrapped_document.height  # visual rows incl. soft-wrap
        except Exception:
            try:
                rows = self.document.line_count
            except Exception:
                rows = 1
        cap = self._max_input_lines()
        self.styles.height = max(1, min(cap, int(rows or 1)))
        # Keep the CSS scroll clamp in lockstep with the computed cap so that
        # once content exceeds it the TextArea scrolls internally rather than
        # being cut off (the CSS `max-height` was a static 10).
        self.styles.max_height = cap

    def on_text_area_changed(self, event: "TextArea.Changed") -> None:
        # Re-measure on EVERY content change — typing, backspace, cut — not
        # just paste/history. A long typed line soft-wraps in the document;
        # without this the widget stayed one row tall and only the last
        # wrapped row (the "end") was visible, so mid-line edits looked
        # impossible. Growing the box exposes the whole wrapped input.
        self.autosize_height()

    # ── input history (per-session, in-memory) ──
    # Same shell-style behaviour as the old Input: ↑ recalls older
    # submissions, ↓ walks newer / clears the draft. For a multi-line
    # field we only hijack the arrow for history when the cursor can't
    # move further in that direction (↑ on the first line, ↓ on the last
    # line) so normal multi-line cursor movement is preserved.
    def _hist_init(self) -> None:
        if not hasattr(self, "_input_history"):
            self._input_history: list[str] = []
            self._input_history_pos: int = -1  # -1 = not browsing
            self._input_history_draft: str = ""

    def history_push(self, text: str) -> None:
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

    def _set_text_and_cursor_end(self, text: str) -> None:
        self.text = text or ""
        self.move_cursor(self.document.end)
        self.autosize_height()

    def _hist_navigate(self, direction: int) -> bool:
        self._hist_init()
        if not self._input_history and direction < 0:
            return False
        pos = self._input_history_pos
        if direction < 0:  # up = older
            if pos == -1:
                self._input_history_draft = self.text
                pos = len(self._input_history) - 1
            elif pos > 0:
                pos -= 1
            else:
                return False
        else:  # down = newer
            if pos == -1:
                if not self.text:
                    return False
                self._set_text_and_cursor_end("")
                self._input_history_draft = ""
                return True
            if pos < len(self._input_history) - 1:
                pos += 1
            else:
                self._input_history_pos = -1
                self._set_text_and_cursor_end(self._input_history_draft)
                return True
        self._input_history_pos = pos
        self._set_text_and_cursor_end(self._input_history[pos])
        return True

    def _on_paste(self, event) -> None:  # type: ignore[override]
        # Insert the paste ourselves (normalizing CRLF/CR → LF) and grow the
        # box. We do NOT defer to TextArea's stock paste: that path skipped our
        # autosize, so a long single-line paste that should soft-wrap stayed one
        # row in a live terminal.
        text = getattr(event, "text", "") or ""
        if not text:
            # A terminal pastes ONLY text; an image copied to the OS
            # clipboard arrives here as an EMPTY paste. This is exactly
            # where we hook image capture: read the clipboard for a PNG
            # and, if present, attach it to the next message.
            try:
                handler = getattr(self.screen, "_attach_clipboard_image", None)
                if callable(handler) and handler():
                    event.stop()
            except Exception:
                pass
            return
        joined = text.replace("\r\n", "\n").replace("\r", "\n")
        prevent_default = getattr(event, "prevent_default", None)
        if callable(prevent_default):
            prevent_default()
        # Collapse a LARGE paste to a single deletable chip so the composer
        # doesn't flood (every peer does this: claude-code
        # `[Pasted text #1 +400 lines]`, pi `[paste #1 +123 lines]`,
        # opencode `[Pasted ~N lines]`). The real text is stashed in the
        # per-widget PasteBuffer and spliced back at submit
        # (ChatScreen._submit_message). Small pastes stay inline.
        if is_large_paste(joined):
            if getattr(self, "_paste_buffer", None) is None:
                self._paste_buffer = PasteBuffer()
            chip = self._paste_buffer.add(joined)
            self.insert(chip)
        else:
            self.insert(joined)
        event.stop()
        # Grow now AND after the next render: a single-line paste only knows its
        # wrapped row-count once the document re-wraps on the following frame,
        # so the synchronous measure can read a stale height of 1.
        self.autosize_height()
        try:
            self.call_after_refresh(self.autosize_height)
        except Exception:
            pass

    async def _on_key(self, event) -> None:  # type: ignore[override]
        # Runs BEFORE TextArea's own _on_key insertion (we call super at
        # the end for the non-intercepted keys). Intercept Enter to
        # submit; Shift+Enter / Ctrl+J insert a literal newline.
        # NOTE: TextArea._on_key is a coroutine, so this override is async
        # and awaits super() for the keys we don't handle ourselves.
        key = event.key

        # ── Enter submits ──
        if key == "enter":
            # When the slash menu is open, Enter is handled by the
            # screen's on_key (selects the highlighted command). We must
            # prevent_default (else TextArea inserts a literal newline)
            # but NOT stop propagation, so the event bubbles up to
            # ChatScreen.on_key which handles the selection + submit.
            if getattr(self.screen, "_slash_matches", None):
                event.prevent_default()
                return
            event.prevent_default()
            event.stop()
            submit = getattr(self.screen, "_submit_message", None)
            if callable(submit):
                submit(self.text)
            return

        # ── Shift+Enter / Ctrl+J insert a newline ──
        if key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        # ── Space → push-to-talk when voice mode is on (empty input) ──
        if key == "space":
            vs = getattr(self.app, "voice_state", None)
            if vs is not None and getattr(vs, "enabled", False):
                already_recording = getattr(
                    self.screen, "_ptt_recorder", None
                ) is not None
                input_empty = not (self.text or "").strip()
                last_voice_fill = getattr(self.screen, "_ptt_last_input_value", None)
                voice_filled_untouched = (
                    last_voice_fill is not None
                    and (self.text or "") == last_voice_fill
                )
                if already_recording or input_empty or voice_filled_untouched:
                    ptt = getattr(self.screen, "action_ptt_space", None)
                    if callable(ptt):
                        ptt()
                        event.prevent_default()
                        event.stop()
                        return

        # ── Arrow-key history navigation ──
        # When the slash menu is open, up/down navigate the menu (screen
        # on_key) — prevent_default so TextArea doesn't move the cursor,
        # but let the event bubble to ChatScreen.on_key.
        if key in ("up", "down"):
            if getattr(self.screen, "_slash_matches", None):
                event.prevent_default()
                return
        if key == "up" and self.cursor_at_first_line:
            if self._hist_navigate(-1):
                event.prevent_default()
                event.stop()
                return
        elif key == "down" and self.cursor_at_last_line:
            if self._hist_navigate(+1):
                event.prevent_default()
                event.stop()
                return

        # Everything else: TextArea's native handling (typing, cursor
        # movement within the document, scrolling, etc.).
        await super()._on_key(event)


from ..bridge import AgentEvent, ApprovalRequest
from ..widgets.chat_log import ChatLog
from ...autonomy import AutonomyLevel, apply_autonomy_to_permissions, get_policy

_SLASH_COMMANDS = [
    ("/permissions", "Toggle command approvals on/off"),
    ("/status", "Show runtime: server health, current model, perf config"),
    ("/restart", "Restart the model server (use when /status shows 'unreachable')"),
    ("/mcp", "List or reload MCP servers from ~/.localcode/mcp.json"),
    ("/skills", "List loaded skills and where they're from"),
    ("/model", "List available models / switch (e.g. /model qwen)"),
    ("/delete", "Delete a downloaded model to free disk space (asks first)"),
    ("/hooks", "Show this repo's .localcode/hooks.toml and trust it (runs shell)"),
    ("/paste", "Attach an image/screenshot from the clipboard (or press Ctrl+G)"),
    ("/thinking", "Show / set hidden reasoning policy (off|auto)"),
    ("/sounds", "Toggle completion + approval notification sounds"),
    ("/voice", "Toggle voice mode (push-to-talk dictation into the input box)"),
    ("/audio", "Toggle audio output (assistant reads replies aloud via macOS say)"),
    ("/vision", "Toggle vision mode (let the model see images)"),
    ("/undo", "Revert the last file change the agent made (/undo all for every change)"),
    ("/clear", "Clear conversation history"),
    ("/exit", "Exit LocalCode"),
]
# /search dropped from the palette — Ctrl+F is the canonical entry. The
# command handler still treats it as a no-op alias to avoid surprising
# anyone who typed it before.

# Every recognized slash command (palette names + aliases the handler
# accepts). Input starting with "/" is only treated as a command when its
# first token is one of these; anything else — e.g. a filesystem path like
# /Users/you/project — is sent to the model as a normal message instead of
# being rejected as an "Unknown command". Only `!` enters shell mode.
_KNOWN_COMMANDS = {name for name, _desc in _SLASH_COMMANDS} | {
    "/quit", "/search", "/copy", "/image",
}


def _is_known_command(text: str) -> bool:
    """True only when `text`'s leading token is a recognized slash command."""
    if not text.startswith("/"):
        return False
    head = text.split(None, 1)[0].lower()
    return head in _KNOWN_COMMANDS

if TYPE_CHECKING:
    from ..app import LocalCodeTUI
    from ...telemetry import TurnTrace

# Cycling thinking indicator — icons + labels rotate per tick.
_THINK_ICONS = ["·", "•", "●"]
# Playful generic gerund placeholders for the streaming spinner. These are
# the ONLY labels the spinner may show while the model is reasoning — the
# model's real thinking text must never leak into the spinner (it belongs
# in the expandable thinking section). Rotating between these is fine.
_SPINNER_GERUNDS = [
    "mining", "digging", "crunching", "pondering", "brewing",
    "thinking", "reasoning", "cooking", "noodling", "scheming",
    "percolating", "tinkering",
]


# How long (seconds) each spinner word stays before rotating. Deliberately
# slow: the word is a mood indicator, not a data readout. Rotating it per
# token (at 50-100 tok/s) makes it strobe unreadably — the reference CLI
# agents keep the word steady for a couple seconds and let only the little
# frame glyph + timer animate quickly. ~2.4 s reads as "alive but calm".
_SPINNER_WORD_PERIOD = 2.4


def _spinner_label(tick: int = 0) -> str:
    """Return a generic playful gerund for the streaming spinner.

    Deliberately ignores any model output: callers must NEVER pass the
    model's reasoning text here. `tick` rotates the placeholder so the
    label feels alive without ever revealing real thinking content.
    """
    return _SPINNER_GERUNDS[tick % len(_SPINNER_GERUNDS)]


def _spinner_label_for_elapsed(elapsed: float) -> str:
    """Pick the gerund by WALL-CLOCK time, not token/tick count.

    The word advances once every `_SPINNER_WORD_PERIOD` seconds regardless of
    how fast tokens stream, so a fast decode no longer flickers the label.
    """
    idx = int(max(0.0, elapsed) / _SPINNER_WORD_PERIOD)
    return _SPINNER_GERUNDS[idx % len(_SPINNER_GERUNDS)]


def _fmt_tool_duration(ms: int) -> str:
    """Compact per-tool duration for the done row: `3s`, `1m04s` (matches the
    turn-summary formatter). Only shown when ≥1s so quick calls stay quiet."""
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"

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


def reconcile_live_tokens(live_estimate: int, real_completion_cumulative: int) -> int:
    """Pick the value for the live "↓ N tokens" badge.

    `live_estimate` is the running char/4 estimate of streamed output for the
    turn so far; `real_completion_cumulative` is llama-server's real completion
    count summed over every round that has closed. The estimate undercounts a
    multi-round turn badly (it only really tracks the round currently
    decoding), so once real usage is available we snap up to it. max() keeps
    whichever is larger so (a) the badge never regresses and (b) the in-flight
    char estimate for the round still decoding — for which no real usage has
    arrived yet — keeps the badge advancing live between rounds.
    """
    return max(int(live_estimate or 0), int(real_completion_cumulative or 0))


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
        background: ansi_default;
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
        background: ansi_default;
    }
    #active-step {
        background: ansi_default;
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
        background: ansi_default;
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
        background: ansi_default;
        height: auto;
        max-height: 10;
        padding: 0 1;
        display: none;
    }
    #slash-menu.active {
        display: block;
    }
    #search-bar {
        background: ansi_default;
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
        background: ansi_default;
        display: none;
    }
    #search-input.active {
        display: block;
    }
    /* Input box + row and children — explicit terminal-default so
       Textual's surface color doesn't bleed through. The box is one
       bordered multi-line field: the `›` prompt on the left and the
       scrollable TextArea filling the rest of the row. */
    #input-box {
        background: ansi_default;
        height: auto;
    }
    #input-row {
        background: ansi_default;
        /* Grow with the TextArea's content (auto) rather than a fixed
           single row, so multi-line input expands the box. */
        height: auto;
    }
    #input-prompt {
        background: ansi_default;
        color: #5f87ff;
        width: 2;
        /* Pin the prompt glyph to the top of a multi-line input so it
           sits beside the first line, terminal-coding-tools style. */
        height: 1;
    }
    /* The multi-line chat input (TextArea). Auto-grows with content up
       to max-height, then scrolls internally so the rest of the layout
       (status bar, slash menu) stays put. */
    #chat-input {
        background: ansi_default;
        width: 1fr;
        height: auto;
        /* Initial scroll clamp; `_ChatTextArea.autosize_height` overrides
           this each change with a terminal-relative cap (~half the height).
           10 is the pre-mount fallback so the first frame looks right. */
        max-height: 10;
        border: none;
        padding: 0;
        scrollbar-size-vertical: 1;
    }
    /* Voice-recording indicator: red left-border accent while
       _ptt_recorder is set (toggled via the `recording` class in the
       PTT start/stop handlers). Replaces the old inline colored-block
       glyph that Input.render_line painted — that doesn't port to
       TextArea's document renderer. */
    #chat-input.recording {
        border-left: thick #ff5470;
    }
    #status-bar {
        dock: bottom;
        /* 4 rows tall: 2 blank rows above the text, the text on row 3,
           1 row of bottom breathing space. */
        height: 4;
        padding: 2 1 1 1;
        color: $text-muted;
        background: ansi_default;
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
        # model_keys for background downloads we've already surfaced a
        # terminal (done/failed) toast for, so the periodic poller fires
        # App.notify exactly once per finished/failed download.
        self._dl_notified: set[str] = set()
        self._pending_messages: list[str] = []
        # Base64-encoded images captured from an empty (image) clipboard
        # paste, attached to the FIRST user message of the next turn.
        self._pending_images: list[str] = []
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
        self._last_round_prompt_tokens: int = 0  # peak window occupancy / turn
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
        self._thinking_streamed: bool = False  # live reasoning shown this turn
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
        # One bordered input BOX containing the `›` prompt and the
        # multi-line, scrollable chat TextArea. The old single-line Input
        # needed a separate `#input-overflow` wrap-preview Static to fake
        # multi-line display; the TextArea wraps + scrolls natively, so
        # that preview (and all the on_input_changed wrap logic) is gone.
        with Vertical(id="input-box"):
            with Horizontal(id="input-row"):
                yield Static("›", id="input-prompt")
                # Multi-line chat input. Enter submits, Shift+Enter
                # inserts a newline (see _ChatTextArea). Voice-recording
                # is shown via the `recording` CSS class on this widget.
                yield _ChatTextArea(
                    id="chat-input", soft_wrap=True,
                    show_line_numbers=False, tab_behavior="focus",
                )
        # Slash command palette appears BELOW the input (terminal coding tools style),
        # not above. Visually it reads as a dropdown extending downward from
        # the prompt the user is typing in.
        yield Static("", id="slash-menu")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self._start_status_probe()
        self._update_status()
        self.query_one("#chat-input", _ChatTextArea).focus()
        log = self.query_one("#chat-log", ChatLog)
        # Warn if this repo ships hooks that shell out but haven't been trusted.
        # They are NOT loaded (see hooks.py) — this just tells the user they
        # exist and how to enable them after review.
        try:
            if getattr(getattr(self.tui, "engine", None), "hooks", None) is not None and \
                    getattr(self.tui.engine.hooks, "untrusted_project_hooks", False):
                log.append_info(
                    "⚠ This repo has .localcode/hooks.toml (runs shell commands). "
                    "It is disabled until you review it with /hooks and run /hooks trust."
                )
        except Exception:
            pass
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
                # Defense-in-depth: an ephemeral "SYSTEM:" nudge is a role:user
                # message that is normally stripped before persistence — but if a
                # turn was interrupted before stripping ran, it can survive. Never
                # replay it as "you: SYSTEM: …" (it was never the user's words).
                if text.startswith("SYSTEM:") or text.startswith("SYSTEM —"):
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
        # Watch background model downloads (kicked off elsewhere, e.g. the
        # model picker / a prior /model switch). When one finishes we toast
        # the user once so they know they can /model to it; on failure we
        # surface the error once. 1 s is responsive without being chatty.
        self.set_interval(1.0, self._poll_active_downloads)

    def _poll_active_downloads(self) -> None:
        """Surface terminal background-download outcomes as one-shot toasts.

        Runs on the UI thread (1 s interval). Walks every in-flight entry
        from `bootstrap.list_active_downloads()`; once a key we've seen
        leaves the active set, we look it up via `bootstrap.download_status`
        and, if it reached a terminal state we haven't announced yet,
        `App.notify` exactly once (tracked in `_dl_notified`).
        """
        from ... import bootstrap
        # Keys currently active (downloading/queued) this tick.
        try:
            active = {e["model_key"] for e in bootstrap.list_active_downloads()}
        except Exception:
            return
        # Remember active keys so we know which ones to check once they
        # drop out of the active set.
        self._dl_active_seen = getattr(self, "_dl_active_seen", set()) | active
        for key in list(self._dl_active_seen):
            if key in active:
                continue  # still in flight — nothing to announce yet
            if key in self._dl_notified:
                self._dl_active_seen.discard(key)
                continue
            entry = bootstrap.download_status(key)
            if entry is None:
                self._dl_active_seen.discard(key)
                continue
            status = entry.get("status")
            if status == "done":
                self._dl_notified.add(key)
                self._dl_active_seen.discard(key)
                name = entry.get("name") or key
                self.app.notify(f"Ready: {name}, type /model to switch")
            elif status == "failed":
                self._dl_notified.add(key)
                self._dl_active_seen.discard(key)
                name = entry.get("name") or key
                err = entry.get("error") or "download failed"
                self.app.notify(
                    f"Download failed: {name} — {err}", severity="error"
                )

    def _start_status_probe(self) -> None:
        """Start the daemon thread that feeds `_update_status` its slow data.

        `_update_status` runs ON THE UI THREAD (2 s interval, plus every
        resize and turn boundary), so it must never block. The two slow
        inputs it needs — the llama-server liveness probe (an HTTP health
        check with a 1 s timeout when the process isn't ours) and the
        one-time `git rev-parse` for the build tag (up to ~1 s on a cold
        disk) — are computed here and cached on plain attributes the UI
        thread just reads. Before this, a loaded/hung server froze the
        whole TUI for up to 1 s out of every 2.
        """
        import threading
        # Optimistic default — matches the old behaviour when the probe
        # couldn't run.
        self._server_alive = True
        self._status_probe_stop = threading.Event()
        t = threading.Thread(target=self._status_probe_loop, daemon=True,
                             name="status-probe")
        t.start()

    def _status_probe_loop(self) -> None:
        # One-time build info (version is cheap; git subprocess is not).
        try:
            from importlib.metadata import version as _pkgver
            self._build_version = _pkgver("localcode")
        except Exception:
            self._build_version = ""
        try:
            import subprocess as _sp
            from pathlib import Path as _Path
            # Best-effort — dev installs have git metadata, installed
            # builds usually don't; either branch is fine.
            r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, timeout=2,
                        cwd=str(_Path(__file__).resolve().parent))
            self._build_commit = r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            self._build_commit = ""
        # Liveness loop. Probe IMMEDIATELY, then every 2 s — same cadence
        # the status bar refreshes at. The cwd git branch (a `git` subprocess,
        # too slow for the UI thread) is refreshed here too so the status bar's
        # directory segment stays current if the user checks out a branch
        # mid-session — but only every ~30 s, not every 2 s tick: a branch
        # switch is rare and a fork per tick for the whole session is wasteful.
        self._cwd_branch = ""
        stop = self._status_probe_stop
        _tick = 0
        while True:
            try:
                from ...server_manager import ServerManager as _SM
                _mgr = _SM.get()
                # `is_running()` only knows about processes we spawned; a
                # llama-server launched by the user (or a prior session) on
                # the same port is serving fine, hence the HTTP fallback.
                self._server_alive = _mgr.is_running() or _mgr.is_healthy()
            except Exception:
                self._server_alive = True  # can't probe → don't cry wolf
            if _tick % 15 == 0:  # first tick, then every ~30 s
                try:
                    from ...telemetry import current_git_branch as _cgb
                    from pathlib import Path as _P
                    self._cwd_branch = _cgb(str(_P.cwd())) or ""
                except Exception:
                    self._cwd_branch = ""
            _tick += 1
            if stop.wait(2.0):
                return

    def on_unmount(self) -> None:
        ev = getattr(self, "_status_probe_stop", None)
        if ev is not None:
            ev.set()

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
            log.reflow_if_needed()
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

        # While the model is REASONING, rotate the playful gerund on a slow
        # wall-clock cadence here — never per streamed chunk. This is the one
        # place the word is chosen, so token speed can't strobe it. Fixed
        # phases ("generating") keep their word.
        if self._active_mode == "thinking" and getattr(self, "_thinking_phase", "") == "thinking":
            elapsed = time.time() - (self._turn_start or time.time())
            text = _spinner_label_for_elapsed(elapsed)

        timer = self._elapsed_str()

        # Esc-to-interrupt affordance — spliced inside the badge parens the
        # way codex renders `(3s • esc to interrupt)` and claude-code /
        # opencode / pi all surface "esc to interrupt". Only while a turn is
        # actually cancellable (`_agent_busy`); `_elapsed_str` always wraps
        # in parens so we inject just before the closing one.
        def _with_interrupt(badge: str) -> str:
            if not self._agent_busy:
                return badge
            hint = "esc to interrupt"
            if badge.endswith(")"):
                return f"{badge[:-1]} · {hint})"
            return f"{badge} · {hint}"

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
            # Only append the interrupt hint on a roomy-enough narrow
            # terminal — otherwise the badge would overflow the row.
            badge = _with_interrupt(timer) if width >= 56 else timer
            max_label = max(8, width - (len(badge) + 4 if width >= 28 else 2))
            if len(label) > max_label:
                label = label[: max(3, max_label - 1)] + "…"
            pos = self._scan_pos % max(len(label), 1)
            line = RichText()
            line.append("  ")
            line.append(label[:pos + 1], style=f"bold {C.primary}")
            line.append(label[pos + 1:], style="dim italic")
            if width >= 28:
                line.append(f" {badge}", style="dim")
            self.query_one("#active-step", Static).update(line)
            return

        badge = _with_interrupt(timer)
        pos = self._scan_pos
        bright = label[:pos + 1]
        dim = label[pos + 1:]
        # Escape markup characters in the text
        bright = bright.replace("[", "\\[")
        dim = dim.replace("[", "\\[")
        line = f"  [bold]{bright}[/][dim italic]{dim}[/]  [dim]{badge}[/]"

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

    def flash_exit_hint(self) -> None:
        """Show a transient grey "Press Ctrl+C again to exit" in the status
        bar (first Ctrl+C). _update_status renders the hint while the window
        is open; the timer below restores the normal bar when it lapses."""
        import time as _t
        self._exit_hint_until = _t.monotonic() + 3.0
        self._update_status()
        try:
            self.set_timer(3.1, self._update_status)
        except Exception:
            pass

    def _update_status(self) -> None:
        import time as _t
        if _t.monotonic() < getattr(self, "_exit_hint_until", 0.0):
            try:
                self.query_one("#status-bar", Static).update(
                    RichText("  Press Ctrl+C again to exit", style="dim")
                )
                return
            except Exception:
                pass
        config = self.tui.config
        from ...models_catalog import current as current_choice
        cur = current_choice(config)
        # Thinking indicator: reflects the `/thinking` policy
        # (internal_thinking_mode), NOT the perf runtime mode. When the active
        # model can't produce hidden reasoning (diffusion arch), show "n/a"
        # rather than a misleading on/off — single source of truth:
        # _model_supports_thinking. Reuse the `cur` resolved just above so the
        # 2 s tick doesn't resolve the current model twice.
        if not self._model_supports_thinking(cur):
            thinking_label = "n/a"
        else:
            _tm = (config.runtime.internal_thinking_mode or "off").strip().lower()
            thinking_label = "off" if _tm in (
                "", "off", "none", "false", "0", "no"
            ) else "on"
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
        # Server status — short, plain-English action label ("ready",
        # "loading", "stopped", "not connected") — the value word IS the state.
        provider = (config.runtime.provider or "").lower()
        # Plain `key: value` — the value word ("ready", "loading",
        # "degraded", "not connected") IS the state; a leading glyph
        # would be decoration. Same shape as the other status fields.
        if self._server_restarting:
            server_label = "server: loading model"
        elif cur is not None and str(
            getattr(cur, "architecture", "")
        ).startswith("diffusion"):
            # Block-diffusion models have no llama-server — each turn
            # spawns the one-shot diffusion runner. Probing the HTTP
            # port would always say "stopped", which reads as broken.
            server_label = "server: diffusion runner"
        elif provider == "llama_cpp":
            # Liveness rather than hardcoded "ready" whenever the provider
            # config is llama_cpp. Prior bug (2026-04-27): after
            # pressure_kill SIGTERM'd llama-server, the status bar still
            # showed "ready" because nothing in this branch checked
            # liveness — user typed a message, got "Backend not ready"
            # from the gateway, and saw a status bar that contradicted
            # the error. The probe itself (process check + HTTP health
            # check with a 1 s timeout) runs in the `_status_probe_loop`
            # daemon thread — reading it here keeps this UI-thread method
            # non-blocking.
            _alive = getattr(self, "_server_alive", True)
            server_label = "server: ready" if _alive else "server: stopped"
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
        # checkout with no git info. Computed once per session by the
        # `_status_probe_loop` daemon thread (the `git rev-parse`
        # subprocess can take ~1 s on a cold disk — too slow for this
        # UI-thread method); until the thread delivers, the getattr
        # defaults below render a plain "dev" tag for a tick or two.
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
        # Compact the parenthetical quant tag: keep the part that
        # disambiguates ("Q4", "BF16", "IQ2_M"), drop the filler. The
        # earlier string-replace version only handled the
        # "(Unsloth UD-…)" shape — on "Gemma 4 12B (BF16, full)" it
        # stripped just the CLOSING paren and the bar showed the
        # broken "Gemma 4 12B (BF16, full".
        _m = re.match(r"^(.*?)\s*\((.*)\)\s*$", model)
        if _m:
            _tag = _m.group(2).replace("Unsloth UD-", "").split(",")[0].strip()
            short_model = f"{_m.group(1)} {_tag}".strip()
        else:
            short_model = model
        short_model = short_model.replace("-A3B", "")
        # Brand prefix + status (LEFT) | version (RIGHT). Like a tmux /
        # vim statusline — left content pinned to the left edge, right
        # content pinned to the right edge, padded with spaces to fill
        # the terminal width. Without this the version sat awkwardly
        # mid-row on wide terminals with empty space trailing it.
        # One quiet line, uniform ` · ` separators. The earlier version
        # opened with an emoji + box-drawing divider and double-spaced
        # separators — decoration that ate ~10 columns and made the bar
        # the loudest thing on screen. Brand color on the name is enough.
        # Working-directory segment (Zero-inspired orientation cue). The
        # `git` branch is computed off the UI thread by the status probe;
        # the path itself is cheap (no subprocess). Collapse $HOME to `~`,
        # and show the full path only on wide terminals — on narrow ones
        # the basename is enough and the rest of the row matters more.
        import os as _os
        try:
            _cwd = _os.getcwd()
            _home = _os.path.expanduser("~")
            if _cwd == _home:
                _cwd_disp = "~"
            elif _cwd.startswith(_home + _os.sep):
                _cwd_disp = "~" + _cwd[len(_home):]
            else:
                _cwd_disp = _cwd
            if term_cols and term_cols < 100:
                _cwd_disp = _os.path.basename(_cwd.rstrip(_os.sep)) or _cwd_disp
        except Exception:
            _cwd_disp = ""
        _branch = getattr(self, "_cwd_branch", "") or ""
        if _cwd_disp:
            from rich.markup import escape as _mesc
            cwd_seg = _mesc(_cwd_disp + (f" ({_branch})" if _branch else ""))
        else:
            cwd_seg = ""
        left = RichText.from_markup(
            f"[{C.primary}]LocalCode[/]"
            + (f" · [dim]{cwd_seg}[/]" if cwd_seg else "")
            + f" · {server_label} · permissions: {self._permissions_label()} · "
            f"context: {pct_remaining}% free · "
            f"thinking: {thinking_label}"
            + f" · model: {short_model}"
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
            # If even the left side overflows, end it with an ellipsis
            # instead of letting the renderer hard-clip mid-word —
            # "…model: Gemma 4 12B (BF16, fu" reads as a bug, "…" reads
            # as intentional.
            if bar_width and left.cell_len > bar_width:
                left.truncate(bar_width, overflow="ellipsis")
            bar.update(left)

    def _permissions_label(self) -> str:
        engine = self.tui.engine
        if engine is not None and engine._autonomy == AutonomyLevel.FULL_AUTO:
            return "off"
        return "on"

    def _update_queue(self) -> None:
        q = self.query_one("#queue-line", Static)
        if self._pending_messages:
            n = len(self._pending_messages)
            preview = self._pending_messages[0][:40]
            from rich.markup import escape as _mesc
            q.update(f" ↻ {n} queued: \"{_mesc(preview)}\"{'…' if len(self._pending_messages[0]) > 40 else ''}")
            q.add_class("active")
        else:
            q.remove_class("active")

    # ── Input handling ──

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live search as the user types in the (single-line) search box.

        The chat input is now a TextArea, not an Input, so its changes
        arrive via `on_text_area_changed` below — this handler only sees
        the `#search-input` Input now.
        """
        if event.input.id == "search-input":
            self._do_search(event.value)

    def on_text_area_changed(self, event) -> None:
        """Chat TextArea content changed — drive slash menu + voice state.

        The TextArea wraps + scrolls natively, so the old
        `#input-overflow` wrap-preview logic is gone; we only need to
        maintain the slash-command menu and the voice "untouched"
        snapshot here.
        """
        try:
            if event.text_area.id != "chat-input":
                return
        except Exception:
            return
        try:
            event.text_area.autosize_height()  # grow/shrink with content
            # Re-measure after the next render too: a long single-line paste
            # only knows its soft-wrapped row count once the document re-wraps
            # on the following frame, so the synchronous call above can read a
            # stale height of 1 in a live terminal.
            event.text_area.call_after_refresh(event.text_area.autosize_height)
        except Exception:
            pass
        self._on_chat_text_changed(event.text_area.text)

    def _on_chat_text_changed(self, text: str) -> None:
        # Invalidate the "voice-filled but untouched" snapshot the
        # moment the value diverges from what we last wrote — past
        # that point Space should type a literal space, not re-trigger
        # PTT, because the user is actively editing.
        last_voice = getattr(self, "_ptt_last_input_value", None)
        if last_voice is not None and text != last_voice:
            self._ptt_last_input_value = None
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
                # Don't leave the highlight on a greyed-out command (e.g.
                # /thinking on a non-reasoning model): Enter would no-op. Land
                # on the first selectable entry, matching arrow/Tab nav.
                if self._slash_cmd_disabled(self._slash_matches[self._slash_selected][0]):
                    self._slash_selected = self._next_selectable_slash(self._slash_selected, +1)
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

    def _model_supports_thinking(self, cur=_UNSET) -> bool:
        """Whether the currently-selected model can emit hidden reasoning.

        Drives the `/thinking` selector and the status-bar label: diffusion
        models (and any future entry that declares no `thinking` capability)
        can't produce a toggleable reasoning channel, so the option is disabled
        rather than offered as if it works. Unknown / unresolved model → assume
        it can, so we never wrongly grey out a real reasoning model.

        `cur` may be passed in by callers (e.g. the 2 s status tick) that have
        already resolved the current ModelChoice, to avoid re-resolving it.
        """
        try:
            if cur is _UNSET:
                from ...models_catalog import current as current_choice
                cur = current_choice(self.tui.config)
        except Exception:
            return True
        if cur is None:
            return True
        try:
            return bool(cur.supports_thinking)
        except Exception:
            return True

    def _slash_cmd_disabled(self, cmd: str) -> bool:
        """True when a slash command must render greyed-out + non-selectable
        for the current model. Today only `/thinking` is gated (on models
        without a hidden-reasoning channel)."""
        if cmd == "/thinking":
            return not self._model_supports_thinking()
        return False

    def _next_selectable_slash(self, start: int, step: int) -> int:
        """Index of the next non-disabled slash match, walking `step` (±1).

        Skips greyed-out entries (e.g. /thinking on a non-reasoning model)
        so arrow/Tab navigation never lands the highlight on something that
        can't be chosen. Falls back to `start` if every match is disabled."""
        n = len(self._slash_matches)
        if n == 0:
            return 0
        idx = start
        for _ in range(n):
            idx = (idx + step) % n
            if not self._slash_cmd_disabled(self._slash_matches[idx][0]):
                return idx
        return start

    def _render_slash_menu(self) -> None:
        """Render the slash palette below the input.

        Each row stays on exactly ONE line. If the description doesn't
        fit, truncate with `…` instead of wrapping (wrapping made it
        look like there were extra options below). Width is computed
        from the actual screen width minus the fixed command column.

        Commands that are unavailable for the current model (e.g.
        `/thinking` on a diffusion model) render in a dimmed/struck style
        and can't be selected.
        """
        menu = self.query_one("#slash-menu", Static)
        try:
            avail = max(20, (self.size.width or 80) - 18)
        except Exception:
            avail = 60
        lines = []
        for i, (cmd, desc) in enumerate(self._slash_matches):
            disabled = self._slash_cmd_disabled(cmd)
            d = "unavailable for this model" if disabled else desc
            # Truncate BOTH the real and the "unavailable" text so neither wraps
            # to a second visual row on a narrow terminal (wrapping reads as a
            # phantom extra menu entry — the exact thing this render prevents).
            if len(d) > avail:
                d = d[: max(0, avail - 1)].rstrip() + "…"
            if disabled:
                # Greyed out + non-selectable: dim + strikethrough so it
                # reads as "offered but not usable here", never highlighted.
                lines.append(f"[dim strike]{cmd:<14}[/]  [dim italic]{d}[/]")
            elif i == self._slash_selected:
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
        # Only the single-line search box submits via Input now — the
        # chat input is a TextArea whose Enter is routed to
        # `_submit_message` (see _ChatTextArea._on_key).
        if event.input.id == "search-input":
            # Search input — Enter navigates to next result
            if self._search_results:
                self._search_idx = (self._search_idx + 1) % len(self._search_results)
                self._show_search_result()
            return

    def _submit_message(self, raw_text: str) -> None:
        """Submit the chat input's contents.

        Called by the TextArea's Enter binding (`_ChatTextArea._on_key`)
        with the widget's full multi-line `.text`. Mirrors the old
        single-line `on_input_submitted` chat path.
        """
        inp = self.query_one("#chat-input", _ChatTextArea)
        text = raw_text if raw_text is not None else inp.text
        # Expand any collapsed-paste chips back to their real text before
        # anything else looks at the message. Chips the user deleted from
        # the composer never match and are dropped; clear() forgets the
        # rest so they can't leak into the NEXT message.
        pb = getattr(inp, "_paste_buffer", None)
        if pb is not None and len(pb):
            text = pb.expand(text)
            pb.clear()
        text = (text or "").strip()
        if not text:
            return
        # Record into per-input history so ↑/↓ can recall it next time.
        # Skipped only for real slash commands (users don't want `/clear`
        # and `/quit` in navigable history) — a `/path` message IS recorded.
        if not _is_known_command(text):
            inp.history_push(text)
        inp.clear()
        # Belt + suspenders for the "submitted text reappears in the
        # input" bug. Three layers:
        #   1. Bump session counter — stale workers' apply call sees
        #      mismatched session and drops their update.
        #   2. Clear prefix + last_transcript so next voice session
        #      starts from a clean slate.
        #   3. Record submit_ts; _apply_partial_transcript refuses to
        #      write within 2 seconds of submit_ts (catches races
        #      I might have missed).
        #   4. Also force inp.value = "" again on a 100ms tick — if
        #      anything writes it back in that window, this overrides.
        import time as _t
        self._ptt_session = getattr(self, "_ptt_session", 0) + 1
        self._ptt_input_prefix = ""
        self._ptt_last_transcript = ""
        self._ptt_last_input_value = None
        self._ptt_last_submit_ts = _t.time()
        # Defensive second clear after a tick in case anything wrote the
        # value back between clear() and now. ONLY clear if the field
        # still holds the exact text we just submitted — the bug this
        # guards against re-inserts the SAME string, whereas a user who
        # starts typing a NEW message within 1.5 s produces different
        # text, which we must not wipe (that was eating fast follow-ups).
        _submitted = text
        def _double_clear(expected=_submitted) -> None:
            try:
                inp2 = self.query_one("#chat-input", _ChatTextArea)
                if inp2.text and inp2.text == expected:
                    inp2.text = ""
            except Exception:
                pass
        self.set_timer(0.1, _double_clear)
        self.set_timer(0.5, _double_clear)
        self.set_timer(1.5, _double_clear)

        # A leading "/" is a command ONLY when it matches a known one —
        # otherwise (e.g. a pasted path like /Users/you/repo) it's a normal
        # message for the model, not an "Unknown command".
        if _is_known_command(text):
            self._handle_command(text)
            return

        # Bash mode — `!command` runs through the user's shell and the
        # output lands in the chat log, no model involved. Mirrors the
        # `!` convention of the leading CLI agents.
        if text.startswith("!"):
            self._run_bash(text[1:].strip())
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

    def _run_bash(self, cmd: str) -> None:
        """Run a user-typed `!command` and append its output to the log.

        Executes in a thread worker so a slow command never blocks the
        UI; output is appended via call_from_thread when it finishes.
        The command is the USER'S OWN — typed by hand at their prompt —
        so no approval gate applies (same trust level as their normal
        terminal). Output is capped so `!cat bigfile` can't flood the
        log.
        """
        log = self.query_one("#chat-log", ChatLog)
        if not cmd:
            log.append_info(
                "[dim]bash mode — type ! followed by a command, e.g. !git status[/]"
            )
            return
        log.append_user(f"! {cmd}")
        log.scroll_end(animate=False)

        # Run at the repo root (the user's `!cmd` should behave like the
        # agent's bash tool, which cwds to the repo), falling back to the
        # process cwd before the backend is initialized.
        _cwd = None
        try:
            _eng = getattr(self.tui, "engine", None)
            _rr = getattr(_eng, "repo_root", None) if _eng is not None else None
            if _rr:
                _cwd = str(_rr)
        except Exception:
            _cwd = None

        def _work() -> None:
            import subprocess as _sp
            shell = os.environ.get("SHELL") or "/bin/sh"
            try:
                r = _sp.run(
                    [shell, "-c", cmd],
                    capture_output=True, text=True, errors="replace",
                    timeout=120, cwd=_cwd,
                )
                out = r.stdout or ""
                if r.stderr:
                    if out and not out.endswith("\n"):
                        out += "\n"
                    out += r.stderr
                out = out.rstrip("\n")
                code = r.returncode
            except _sp.TimeoutExpired:
                out, code = "(timed out after 120s)", 124
            except Exception as e:
                out, code = f"({e})", 1
            cap = 8000
            if len(out) > cap:
                out = out[:cap] + f"\n… (+{len(out) - cap:,} more chars)"

            def _render(out=out, code=code) -> None:
                lg = self.query_one("#chat-log", ChatLog)
                if out:
                    lg.append_tool_result(out, error=(code != 0))
                elif code != 0:
                    lg.append_tool_result(f"(exit {code}, no output)", error=True)
                else:
                    lg.append_tool_result("(no output)")
                lg.scroll_end(animate=False)

            self.app.call_from_thread(_render)

        self.run_worker(_work, thread=True, exclusive=False)

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

    def _interrupt_turn(self) -> None:
        """Esc-to-interrupt: cancel the in-flight agent turn.

        Same effect as typing 'stop'/'cancel' (see `_request_cancel` and
        `_is_stop_intent`), but triggered by the Escape key. Flips
        `app.cancel_requested` — the agent loop polls it at round and tool
        boundaries (agent/loop.py) — and also cancels the Textual worker as
        a backstop. Drops any queued messages and shows a short steering
        line (claude-code's InterruptedByUser pattern).
        """
        if not self._agent_busy:
            return
        log = self.query_one("#chat-log", ChatLog)
        # Primary mechanism: the loop polls this flag and unwinds cleanly.
        if self.tui.engine is not None:
            self.tui.engine.cancel_requested = True
        # Backstop: ask the worker to cancel too. It's a thread worker
        # running blocking work, so this mainly matters at await points —
        # cancel_requested is what actually unwinds the loop — but it's
        # cheap and harmless.
        w = getattr(self, "_turn_worker", None)
        if w is not None:
            try:
                w.cancel()
            except Exception:
                pass
        dropped = len(self._pending_messages)
        self._pending_messages.clear()
        try:
            self._update_queue()
        except Exception:
            pass
        note = "Interrupted · tell me what to do differently"
        if dropped:
            note += f" (dropped {dropped} queued message{'s' if dropped != 1 else ''})"
        log.append_info(note)

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
            self._last_round_prompt_tokens = 0
            self._total_tokens = 0
            self._update_status()
        elif text == "/undo" or text == "/undo all":
            engine = self.tui.engine
            changes = getattr(getattr(engine, "toolkit", None), "changes", None)
            if changes is None:
                log.append_info("Nothing to undo yet.")
            elif text == "/undo all":
                msgs = changes.undo_all()
                if msgs:
                    for m in msgs:
                        log.append_info(f"  └ {m}")
                    log.append_info(f"Reverted {len(msgs)} change(s).")
                else:
                    log.append_info("Nothing to undo.")
            else:
                ok, msg = changes.undo_last()
                (log.append_info if ok else log.append_error)(msg)
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
                    # Turning approvals back on must revoke broad grants cached
                    # while the session was prompt-free.
                    app._session_allow.clear()
                    app.perms._session_approved.clear()
                    apply_autonomy_to_permissions(app.perms, get_policy(app._autonomy))
                    log.append_info("Permissions ON — asks only for high-impact commands")
                else:
                    app._autonomy = AutonomyLevel.FULL_AUTO
                    apply_autonomy_to_permissions(app.perms, get_policy(app._autonomy))
                    log.append_info("Permissions OFF — no approval prompts")
                self._update_status()
        elif text == "/search":
            self.action_toggle_search()
        elif text == "/model" or text.startswith("/model "):
            self._handle_model_command(text)
        elif text == "/delete" or text.startswith("/delete "):
            self._handle_delete_command(text)
        elif text == "/hooks" or text.startswith("/hooks "):
            self._handle_hooks_command(text)
        elif text == "/paste" or text == "/image":
            log = self.query_one("#chat-log", ChatLog)
            if not self._attach_clipboard_image():
                log.append_info("[dim]No image on the clipboard — copy or screenshot one first.[/]")
        elif text == "/thinking" or text.startswith("/thinking "):
            self._handle_thinking_command(text)
        elif text == "/status":
            self._handle_status_command()
        elif text == "/restart":
            log = self.query_one("#chat-log", ChatLog)
            log.append_info("Restarting model server...")
            self._restart_for_vision_change(reason="Server restarted")
        elif text == "/mcp" or text.startswith("/mcp "):
            self._handle_mcp_command(text)
        elif text == "/skills":
            self._handle_skills_command(text)
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
        # Gate: models with no hidden-reasoning channel (diffusion) can't honor
        # a thinking policy. Refuse rather than silently pretend it took effect
        # — mirrors the greyed-out /thinking entry in the slash menu.
        if not self._model_supports_thinking():
            log.append_info(
                "The current model doesn't support hidden thinking, so "
                "/thinking has no effect here."
            )
            return
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
            self._update_status()  # reflect the change in the status bar now
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
        self._update_status()  # reflect the change in the status bar now

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
        _vision_on = bool(getattr(config.runtime, "vision_enabled", False))
        _mmproj_cached = bool(choice and choice.mmproj_path and choice.mmproj_path.is_file())
        mmproj_on = "yes" if (_vision_on and _mmproj_cached) else ("off (cached)" if _mmproj_cached else "no")

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
        progress message. Called on the UI thread only.

        Must ALSO toggle the `active` class: #active-step is
        `display: none` at rest, so a bare .update() painted text into
        an invisible widget — the reason download progress never showed
        up in the chat UI.
        """
        try:
            step = self.query_one("#active-step", Static)
            step.update(f"[dim]{msg}[/]")
            step.add_class("active")
        except Exception:
            pass

    def _clear_download_line(self) -> None:
        try:
            step = self.query_one("#active-step", Static)
            step.update("")
            # Only hide it when no agent turn owns the widget — mid-turn
            # the tool/thinking ticker is rendering into it.
            if not self._agent_busy:
                step.remove_class("active")
        except Exception:
            pass

    def _handle_skills_command(self, text: str) -> None:
        """List the skills the agent can use, grouped by origin, with token cost.

        Skills are auto-discovered markdown files (frontmatter + body). Add one
        by dropping a `.md` into ~/.localcode/skills/ (global) or
        ./.localcode/skills/ (project); user/project override bundled by name.
        """
        from pathlib import Path as _Path
        from ...skills import load_registry
        log = self.query_one("#chat-log", ChatLog)
        reg = load_registry(_Path.cwd())
        skills = sorted(reg.skills.values(), key=lambda s: (s.origin, s.name))
        if not skills:
            log.append_info(
                "No skills loaded. Drop a .md with `name:`/`description:` frontmatter "
                "into ~/.localcode/skills/ (global) or ./.localcode/skills/ (project)."
            )
            return
        log.append_info(f"{len(skills)} skill(s) loaded (auto-activated or via the skill tool):")
        for s in skills:
            tokens = max(1, (len(s.body) + len(s.description)) // 4)
            where = "(bundled)" if s.origin == "bundled" else str(s.source_path)
            log.append_info(f"  {s.name}  ·  {s.origin}  ·  ~{tokens} tok  ·  {where}")

    def _handle_mcp_command(self, text: str) -> None:
        """List + reload MCP servers configured in ~/.localcode/mcp.json.

        Subcommands:
          /mcp           — list connected servers + their tools
          /mcp reload    — disconnect all + re-spawn from config
        """
        # Resolve the log widget FIRST — it's referenced on every path below
        # (including the import-failure branch). Defining it after the try/except
        # made an import error raise NameError and take the whole TUI down.
        log = self.query_one("#chat-log", ChatLog)
        try:
            from ...mcp import (
                load_mcp_config, connect_all, list_connected, shutdown_all,
                MCP_CONFIG_PATH,
            )
        except ImportError as _e:
            log.append_error(
                f"MCP SDK not installed ({_e}). "
                "Install it with: uv add mcp>=1.28.0"
            )
            return

        parts = text.strip().split()
        sub = parts[1] if len(parts) >= 2 else None

        # Every MCP call below can spawn subprocesses / touch the async bridge;
        # a bad server config must surface as a message, never crash the TUI.
        try:
            if sub == "reload":
                shutdown_all()
                count, errors = connect_all()
                log.append_info(f"MCP reloaded: {count} server(s) connected.")
                for e in errors:
                    log.append_error(f"  {e}")
                return

            config = load_mcp_config()
            if not config:
                log.append_info(
                    f"No MCP servers configured. Create {MCP_CONFIG_PATH} with:\n"
                    '  {"mcpServers": {"myserver": {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/path"]}}}'
                )
                return
            connected = list_connected()
            if not connected:
                count, errors = connect_all()
                log.append_info(f"Connected {count} MCP server(s).")
                for e in errors:
                    log.append_error(f"  {e}")
                connected = list_connected()
            for name, tools in connected:
                tool_names = ", ".join(t.get("name", "?") for t in tools) or "(no tools)"
                log.append_info(f"  {name}: {tool_names}")
        except Exception as _e:  # noqa: BLE001 — never let /mcp crash the app
            log.append_error(f"MCP error: {type(_e).__name__}: {_e}")

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
                # Default engine is `say` (instant, no download). If the
                # user explicitly picked a Piper voice via `/audio voice
                # piper:<id>`, prefetch it now so the first spoken reply
                # doesn't stall waiting for the download.
                if state.tts_engine == "piper" and state.tts_voice:
                    def _prefetch():
                        try:
                            from ...voice import _ensure_piper_voice
                            _ensure_piper_voice(
                                state.tts_voice,
                                on_progress=lambda m: self.app.call_from_thread(
                                    log.append_info, m
                                ),
                            )
                        except Exception:
                            pass
                    import threading as _t
                    _t.Thread(target=_prefetch, daemon=True).start()
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
        # Persist the enabled flag now that the projector is on disk, so
        # the runtime relaunches WITH --mmproj and future toggles never
        # re-download (the file stays put; OFF just drops --mmproj).
        try:
            cfg = self.tui.config
            cfg.runtime.vision_enabled = True
            from ...config import save_config
            save_config(cfg)
        except Exception:
            pass
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

    def _attach_clipboard_image(self) -> bool:
        """Empty paste → maybe an image sits on the OS clipboard.

        Reads it as PNG, base64-encodes it into `_pending_images` (sent
        with the next message), shows a visible indicator, and auto-enables
        vision when the model supports it. Returns True if an image was
        attached (so the caller can `event.stop()` and avoid a terminal
        beep). macOS-only under the hood; a no-op elsewhere.
        """
        try:
            from ..clipboard_image import read_clipboard_png
        except Exception:
            return False
        try:
            png = read_clipboard_png()
        except Exception:
            png = None
        if not png:
            # Empty paste, no readable image. If the user was trying to paste an
            # image they'd otherwise get NO feedback — show a one-time grey hint
            # on how image pasting works, tailored to the current model. Guarded
            # so an accidental empty Cmd+V doesn't spam it.
            if not getattr(self, "_image_paste_hint_shown", False):
                try:
                    from ...models_catalog import current as current_choice
                    log = self.query_one("#chat-log", ChatLog)
                    choice = current_choice(self.tui.config)
                    vis_on = bool(getattr(self.tui.config.runtime, "vision_enabled", False))
                    if choice is not None and getattr(choice, "supports_vision", False):
                        if not vis_on:
                            log.append_info(
                                "[dim]To paste an image, run /vision to enable image "
                                "support, then paste again.[/]"
                            )
                            self._image_paste_hint_shown = True
                    else:
                        log.append_info(
                            "[dim]To paste images, switch to a vision model "
                            "(Gemma 4 / Qwen 3.6) with /model, then run /vision.[/]"
                        )
                        self._image_paste_hint_shown = True
                except Exception:
                    pass
            return False
        import base64
        b64 = base64.b64encode(png).decode("ascii")
        if getattr(self, "_pending_images", None) is None:
            self._pending_images = []
        self._pending_images.append(b64)
        n = len(self._pending_images)
        try:
            log = self.query_one("#chat-log", ChatLog)
            log.append_info(
                f"📎 image attached ({n}) — it will be sent with your next message"
            )
        except Exception:
            pass
        try:
            self._maybe_enable_vision_for_image()
        except Exception:
            pass
        return True

    def _maybe_enable_vision_for_image(self) -> None:
        """After attaching a pasted image, make sure the model can see it.

        If the current model supports vision but vision is OFF, enable it
        (which only downloads the projector when genuinely missing). If the
        model can't do vision at all, tell the user to switch models.
        """
        try:
            from ...models_catalog import current as current_choice
            config = self.tui.config
            choice = current_choice(config)
            log = self.query_one("#chat-log", ChatLog)
        except Exception:
            return
        if choice is None or not getattr(choice, "supports_vision", False):
            log.append_info(
                "Note: the current model can't see images — switch to a vision "
                "model with /model to use this attachment."
            )
            return
        if getattr(config.runtime, "vision_enabled", False):
            return  # already on
        log.append_info("Enabling vision so the model can see this image…")
        self._handle_vision_command("/vision")

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
        # `file_on_disk` = projector already downloaded; `enabled` = the
        # persistent config flag that decides whether the server loads it.
        # ON/OFF is driven by the FLAG, not file presence — turning OFF
        # keeps the file so re-enabling never re-downloads.
        file_on_disk = bool(mmproj and mmproj.is_file())
        enabled = bool(getattr(config.runtime, "vision_enabled", False))

        parts = text.strip().split()
        sub = parts[1] if len(parts) >= 2 else None

        def _set_flag(value: bool) -> None:
            config.runtime.vision_enabled = value
            try:
                from ...config import save_config
                save_config(config)
            except Exception:
                pass

        def _turn_on_with_existing_file() -> None:
            # Projector already on disk → just flip the flag + restart so
            # the runtime relaunches WITH --mmproj. No download.
            _set_flag(True)
            log.append_info("Vision ON — restarting server to load the projector…")
            self._restart_for_vision_change(reason="Vision ON")

        def _download_then_enable() -> None:
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

        # ── bare /vision → toggle ─────────────────────────────────
        if sub is None:
            if enabled:
                # Currently ON → flip flag off + restart so the server
                # relaunches WITHOUT --mmproj. Frees the encoder RAM but
                # KEEPS the projector file on disk (instant re-enable).
                _set_flag(False)
                log.append_info("Vision OFF — restarting server to release projector memory…")
                self._restart_for_vision_change(reason="Vision OFF")
                return
            # Currently OFF → enable. If the projector is already on disk,
            # just flip + restart; only download when genuinely missing.
            if file_on_disk:
                _turn_on_with_existing_file()
            else:
                _download_then_enable()
            return

        # ── power-user subcommands ───────────────────────────────
        if sub == "status":
            if enabled and file_on_disk:
                mb = mmproj.stat().st_size // (1024 * 1024)
                log.append_info(f"Vision: on · {mmproj.name} · {mb} MB")
            elif file_on_disk:
                log.append_info("Vision: off (projector cached on disk) · type /vision to enable instantly")
            else:
                log.append_info(
                    f"Vision: off · type /vision to enable "
                    f"(~{choice.mmproj_size_gb:.1f} GB)"
                )
            return

        if sub in ("on", "download", "setup"):
            if enabled:
                log.append_info("Vision already on.")
                return
            if file_on_disk:
                _turn_on_with_existing_file()
            else:
                _download_then_enable()
            return

        if sub == "off":
            if enabled:
                _set_flag(False)
                log.append_info("Vision OFF — restarting server to release projector memory…")
                self._restart_for_vision_change(reason="Vision OFF")
            else:
                log.append_info("Vision already off.")
            return

        log.append_info("Usage: /vision  (toggle)  ·  /vision status")

    def _handle_delete_command(self, text: str) -> None:
        """Handle /delete — free disk space by removing downloaded models.

        Three forms, all delegated to `model_delete.run_delete_command`
        (kept out of the TUI so the safety logic is unit-testable):
          /delete                    — list downloaded models + sizes
          /delete <number or name>   — show what would be removed (no delete)
          /delete <target> confirm   — actually delete

        The in-use model and in-flight downloads are refused outright;
        nothing is ever removed without the explicit `confirm` token.
        """
        from ...model_delete import run_delete_command
        log = self.query_one("#chat-log", ChatLog)
        arg = text[len("/delete"):].strip()
        try:
            lines = run_delete_command(arg, self.tui.config)
        except Exception as e:  # noqa: BLE001 — never let /delete crash the app
            log.append_error(f"/delete failed: {e}")
            return
        for kind, line in lines:
            (log.append_error if kind == "error" else log.append_info)(line)

    def _handle_hooks_command(self, text: str) -> None:
        """Handle /hooks — review and trust this repo's .localcode/hooks.toml.

        Project hooks run shell commands (session start, every prompt, before
        every tool), so an untrusted repo's hooks are NOT loaded until the user
        explicitly trusts them here — this is what stops clone-and-open RCE.

          /hooks          — show the file and whether it's trusted
          /hooks trust    — trust the current content (re-prompts if it changes)
        """
        from pathlib import Path
        from ...hooks import is_project_hooks_trusted, trust_project_hooks
        log = self.query_one("#chat-log", ChatLog)
        repo_root = "."
        try:
            repo_root = str(getattr(self.tui.engine, "repo_root", ".") or ".")
        except Exception:
            pass
        project_path = Path(repo_root) / ".localcode" / "hooks.toml"
        arg = text[len("/hooks"):].strip().lower()
        if not project_path.is_file():
            log.append_info("No .localcode/hooks.toml in this repo — nothing to trust.")
            return
        if arg == "trust":
            if trust_project_hooks(repo_root):
                log.append_info(
                    "Trusted this repo's hooks. They take effect next session "
                    "(restart LocalCode). Re-run /hooks trust if you edit the file."
                )
            else:
                log.append_error("Could not write the hooks trust store.")
            return
        # Bare /hooks — show status + content for review.
        trusted = is_project_hooks_trusted(repo_root)
        log.append_info(f"{project_path} — {'TRUSTED' if trusted else 'NOT TRUSTED (hooks disabled)'}")
        try:
            body = project_path.read_text()[:4000]
            for line in body.splitlines():
                log.append_info(f"  {line}")
        except Exception as e:  # noqa: BLE001
            log.append_error(f"Could not read hooks file: {e}")
            return
        if not trusted:
            log.append_info(
                "These hooks run shell commands. Review them above, then run "
                "`/hooks trust` to enable them for this repo."
            )

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
        from ... import bootstrap
        log = self.query_one("#chat-log", ChatLog)
        config = self.tui.config
        cur = current(config)
        if cur is not None and cur.key == choice.key:
            log.append_info(f"Already using {choice.name}.")
            return
        # The hot-swap below restarts llama-server against the new GGUF, so
        # the file must be fully on disk first. If it isn't complete, BLOCK
        # the switch: ensure a background download is running (respecting
        # bootstrap's slot cap) and tell the user how far along it is. The
        # periodic `_poll_active_downloads` toast will let them know when it
        # finishes so they can re-run /model. We do NOT persist config or
        # touch `_server_restarting` here — the current model keeps serving.
        if not bootstrap.is_download_complete(choice):
            bootstrap.start_background_download(choice)
            entry = bootstrap.download_status(bootstrap.model_key_for(choice))
            if entry is not None and entry.get("status") == "queued":
                ahead = max(
                    0, len(bootstrap.list_active_downloads()) - 1
                )
                log.append_info(
                    f"{choice.name} is queued ({ahead} ahead) — "
                    f"run /model again once it finishes."
                )
            else:
                pct = entry.get("progress_pct", 0) if entry is not None else 0
                log.append_info(
                    f"{choice.name} is still downloading ({pct} percent) — "
                    f"run /model again once it finishes."
                )
            return
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
        # The GGUF is fully on disk by this point (the not-complete case
        # returned early after kicking off / observing the background
        # download), so this is a pure server hot-swap.
        _from = f"from {cur.name} " if cur is not None else ""
        log.append_info(f"Switching {_from}to {choice.name} — restarting server...")
        # Block input submission until the new server is fully loaded.
        # Anything the user types during the restart window is queued
        # (see on_input_submitted) and drained by `_on_server_ready`
        # once /health returns 200. Without this, messages typed during
        # the 20-40 s model load hit the still-loading server and come
        # back as 503 "Loading model" → user-facing E3102.
        self._server_restarting = True
        self._update_status()

        def _worker() -> None:
            # Auto-init the backend if it hasn't been touched yet — a
            # /model swap should restart the server straight away, never
            # ask the user to type a message first. Mirrors the /vision
            # toggle path above; ensure_backend is idempotent + cheap
            # when the engine is already up.
            try:
                self.app.call_from_thread(self.tui.ensure_backend)
                import time as _t
                # Give the call_from_thread a tick to land before we
                # read engine. 50ms is enough; engine init is in-process.
                _t.sleep(0.05)
            except Exception:
                pass
            engine = self.tui.engine.engine if self.tui.engine is not None else None
            if engine is None:
                self.app.call_from_thread(
                    self._on_server_restart_failed,
                    "Backend couldn't initialize — can't restart server. "
                    "Check ~/.localcode/last_error.log.",
                )
                return
            # cohere2moe (North-Mini-Code) needs its dedicated PR-#24260
            # server built before the restart can launch it — the setup
            # screen does this on first launch, but an in-chat /model swap
            # bypasses setup, so build-on-demand here too (else the restart
            # falls through to the TurboQuant binary and times out: E1002).
            _arch = str(getattr(choice, "architecture", ""))
            if "cohere" in _arch:
                from ...bootstrap import ensure_cohere_server
                self.app.call_from_thread(
                    self._set_download_line, "Building cohere server (one-time)…")
                ok, res = ensure_cohere_server(
                    on_progress=lambda m: self.app.call_from_thread(self._set_download_line, m))
                self.app.call_from_thread(self._clear_download_line)
                if not ok:
                    self.app.call_from_thread(
                        self._on_server_restart_failed,
                        f"Couldn't build the cohere2moe server: {res}")
                    return
                engine.config.cohere_server_binary = res
            # muse_glimmer (Meta Muse Glimmer): same story — the TurboQuant
            # binary lacks the arch, so build the dedicated stock server on an
            # in-chat /model swap that bypasses the setup screen.
            if "muse" in _arch:
                from ...bootstrap import ensure_muse_server
                self.app.call_from_thread(
                    self._set_download_line, "Building Muse Glimmer server (one-time)…")
                ok, res = ensure_muse_server(
                    on_progress=lambda m: self.app.call_from_thread(self._set_download_line, m))
                self.app.call_from_thread(self._clear_download_line)
                if not ok:
                    self.app.call_from_thread(
                        self._on_server_restart_failed,
                        f"Couldn't build the Muse Glimmer server: {res}")
                    return
                engine.config.muse_server_binary = res
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
        self._drain_next_queued()

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
                inp = self.query_one("#chat-input", _ChatTextArea)
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
                inp = self.query_one("#chat-input", _ChatTextArea)
                existing = inp.text or ""
                # Ensure exactly one trailing space so the join is clean.
                self._ptt_input_prefix = existing.rstrip() + " " if existing.strip() else ""
                # Voice-recording indicator: red left-border accent via
                # the `recording` CSS class (the old inline colored-block
                # glyph from Input.render_line doesn't port to TextArea).
                inp.add_class("recording")
            except Exception:
                self._ptt_input_prefix = ""
            # Don't log "Recording — release Space" anymore — it spammed
            # the chat log if the watchdog mis-fired. The recording border
            # accent on the input is now the recording indicator.
            def _cursor_pulse() -> None:
                try:
                    self.query_one("#chat-input", _ChatTextArea).refresh()
                except Exception:
                    pass
            self._ptt_cursor_timer = self.set_interval(0.05, _cursor_pulse)

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
                # No repeat after 1.2 s → user tapped instead of holding.
                # We used to fall into a silence-detection mode that kept
                # the mic open until the room went quiet for 1.5 s. In
                # practice ANY ambient noise (typing, fan, breath) kept
                # recording alive indefinitely, which the user reported
                # as "audio recording randomly even though I'm not
                # holding space" — Whisper then transcribed the noise
                # as [Inaudible]/(muffled speaking) and leaked those
                # into the input box. Push-to-talk means HOLD; if you
                # don't hold, you tapped, and we stop now.
                if not got_repeat:
                    self._ptt_stop_and_finalize()
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
                        # Bail early if the recording is already over.
                        # Without this guard a worker that started just
                        # before release runs the full whisper pipeline
                        # against stale audio, races with the final
                        # worker, and can corrupt fd 2 / the Metal KV
                        # cache. Both manifest as second-press crashes.
                        if (getattr(self, "_ptt_session", -1) != session_at_start
                                or getattr(self, "_ptt_recorder", None) is None):
                            return
                        try:
                            size = snap.stat().st_size
                        except OSError:
                            size = 0
                        if size < 10_000:
                            return
                        from ...voice import transcribe as _trans
                        ok, text = _trans(state, snap)
                        if ok and text:
                            self.app.call_from_thread(
                                self._apply_partial_transcript, text, session_at_start
                            )
                    except Exception:
                        # Whisper can raise on malformed WAV / Metal
                        # buffer mismatch. Swallowing keeps the daemon
                        # thread from killing the TUI process.
                        pass
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
            inp = self.query_one("#chat-input", _ChatTextArea)
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
            inp.text = joined
            inp.move_cursor(inp.document.end)
            inp.focus()
            # Snapshot what we just wrote so the space-PTT gate can tell
            # whether the user has typed anything since. If the input
            # value still equals this snapshot when Space is pressed,
            # we treat the input as "voice-filled but untouched" and
            # start a NEW recording (appending) instead of typing a
            # space. Cleared in on_input_changed when user edits.
            self._ptt_last_input_value = joined
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
        # Drop the recording border accent + force one final repaint so
        # the indicator disappears immediately.
        try:
            inp = self.query_one("#chat-input", _ChatTextArea)
            inp.remove_class("recording")
            inp.refresh()
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
                # Defensive: skip whisper entirely for extremely short
                # audio. pywhispercpp's Metal backend has crashed on
                # < 0.3 s WAVs (16 kHz mono = < ~10 KB). Tap-release
                # mishaps would deliver a few hundred ms; better to
                # show no transcript than to segfault the TUI.
                from pathlib import Path as _P
                try:
                    size = _P(wav_path).stat().st_size
                except OSError:
                    size = 0
                # 16-bit mono PCM at 16 kHz: 32 KB/sec. 0.3 s ≈ 10 KB.
                # WAV header is 44 bytes, so the threshold below covers
                # both "no audio captured" and "milliseconds of audio".
                if size < 10_000:
                    return
                from ...voice import transcribe as _trans
                ok, text = _trans(state, wav_path)
                if ok and text:
                    self.app.call_from_thread(
                        self._apply_partial_transcript, text, session_final
                    )
            except Exception:
                # Whisper occasionally raises on malformed WAVs or
                # Metal-backend buffer mismatches. Swallow rather than
                # crash the daemon thread (which on macOS terminates
                # the whole process).
                pass
            finally:
                try:
                    from pathlib import Path as _P
                    _P(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass
        import threading as _t3
        _t3.Thread(target=_final_worker, daemon=True).start()

    def action_ptt_cancel(self) -> None:
        """Escape key. Priority order:

          1. An active voice recording → discard it (original behaviour).
          2. An in-flight agent turn → interrupt it (esc-to-interrupt, the
             affordance every peer CLI shows in its working badge).
          3. Idle → return without consuming, so other Esc handlers run.
        """
        if not getattr(self, "_ptt_recorder", None):
            # No recording. If an agent turn is running, Escape interrupts
            # it instead of doing nothing.
            if self._agent_busy:
                self._interrupt_turn()
            return  # otherwise let other Esc handlers run
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
        # Drop the recording border accent + force one final repaint so
        # the indicator disappears immediately.
        try:
            inp = self.query_one("#chat-input", _ChatTextArea)
            inp.remove_class("recording")
            inp.refresh()
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
            self.query_one("#chat-input", _ChatTextArea).focus()
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
            from rich.markup import escape as _mesc
            bar.update(f"[dim]No results for[/] [bold]\"{_mesc(query)}\"[/]")
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
        # Hand any images pasted from the clipboard to the engine so the
        # agent loop can attach them to THIS turn's first user message.
        # app.ask() composes text-only messages, so the images ride on the
        # engine object (which the loop already receives) rather than a
        # new param. Cleared after handoff so they send exactly once.
        _imgs = getattr(self, "_pending_images", None)
        if _imgs:
            try:
                self.tui.engine._pending_images = list(_imgs)
            except Exception:
                pass
            self._pending_images = []
        # Keep a handle on the turn worker so Escape (esc-to-interrupt) can
        # cancel it as a backstop to app.cancel_requested.
        self._turn_worker = self.run_agent_turn(text)

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
            # Safety net: a worker that died BEFORE calling its own
            # completion handler (e.g. the model-switch or vision-restart
            # worker throwing mid-flight) would otherwise leave
            # _server_restarting / _vision_download_in_flight stuck True
            # forever — input queues with no way to unblock, or /vision
            # is permanently dead. Clear them here whatever the worker.
            self._recover_stuck_state()
            # But ONLY the agent-turn worker may end the turn. Running
            # _on_turn_done for ANY errored worker (a `!cmd` helper, a
            # model-swap probe) mid-turn set _agent_busy=False and drained
            # the queue into a SECOND run_agent_turn while the first thread
            # was still running ask() — two agent loops mutating one session.
            if event.worker.name == "run_agent_turn":
                self._on_turn_done()
        elif event.state == WorkerState.CANCELLED:
            # Esc-to-interrupt cancels the worker (`_interrupt_turn` calls
            # `w.cancel()`), so a cancelled agent turn lands here — NOT in the
            # SUCCESS branch. Without this the turn never finalizes: the
            # `◆ thinking…` indicator keeps animating and `_agent_busy` stays
            # True (input wedged) after a cancel. Mirror the SUCCESS/ERROR cleanup.
            if event.worker.name == "run_agent_turn":
                self._on_turn_done()

    def _recover_stuck_state(self) -> None:
        """Clear blocking flags after a worker crash so the UI can't wedge.

        Idempotent and harmless when nothing is stuck. Restores input,
        refreshes the status bar + queue, and drains any messages the
        user typed while the (now-dead) worker held the lock.
        """
        was_blocked = self._server_restarting or getattr(
            self, "_vision_download_in_flight", False
        )
        self._server_restarting = False
        self._vision_download_in_flight = False
        try:
            self._clear_download_line()
        except Exception:
            pass
        if was_blocked:
            try:
                self._update_status()
            except Exception:
                pass
            # Kick the queue drain — input typed during the dead restart
            # is still in _pending_messages.
            try:
                self._drain_next_queued()
            except Exception:
                pass

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
        # Refresh the context counter. Prefer the backend's REAL prompt-token
        # count for the turn (`in_tokens`) — it includes the full agentic
        # working context (system prompt + tools + every tool result the model
        # actually processed this turn), which `_recompute_context_used` MISSES
        # because session.messages only holds user turns + final assistant text
        # (the tool history lives in the agent loop's own list). Summing only
        # session.messages was why the bar read ~99% free even mid-build. Fall
        # back to the session estimate when the backend reported no usage.
        self._context_used = max(
            self._last_round_prompt_tokens, self._recompute_context_used()
        )
        self._update_status()

        # Auto-submit queued messages
        self._drain_next_queued()

    def _drain_next_queued(self) -> None:
        """Pop + start the next queued message, crash-safe.

        If `_start_turn` throws, the popped message is put BACK at the
        front of the queue (instead of silently lost) and the error is
        surfaced — the user can retry rather than wondering where their
        message went.
        """
        if not self._pending_messages or self._agent_busy or self._server_restarting:
            return
        next_msg = self._pending_messages.pop(0)
        self._update_queue()
        try:
            log = self.query_one("#chat-log", ChatLog)
            log.append_user(next_msg)
            log.scroll_end(animate=False)
            self._start_turn(next_msg)
        except Exception as e:
            self._pending_messages.insert(0, next_msg)
            self._update_queue()
            try:
                self.query_one("#chat-log", ChatLog).append_error(
                    f"Couldn't start queued message (kept in queue): {e}"
                )
            except Exception:
                pass

    # ── Agent events (from bridge via OutputManager) ──

    def on_agent_event(self, event: AgentEvent) -> None:
        log = self.query_one("#chat-log", ChatLog)
        t = event.event_type
        p = event.payload

        if t == "content":
            # Hide animation once content starts flowing
            if self._active_mode:
                self._hide_active_step()
            # Close any live reasoning stream before the answer starts so the
            # dimmed thinking and the answer don't interleave.
            if getattr(self, "_thinking_streamed", False):
                log.end_thinking_stream()
                self._thinking_streamed = False
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
                _pt = int(p.get("prompt_tokens", 0) or 0)
                self._turn_prompt_tokens += _pt
                # The LAST round's prompt size = the real peak window occupancy
                # this turn (history grows each round, so the final round's
                # prompt is the fullest). The accumulated sum over-counts the
                # window; this is what the context bar should reflect.
                if _pt:
                    self._last_round_prompt_tokens = _pt
                    # Live update: let the context bar drop mid-turn as
                    # each round's prompt grows the window.  The turn-end
                    # handler overwrites this with the same max() logic so
                    # there is no regression on the final value.
                    self._context_used = max(
                        self._last_round_prompt_tokens, self._context_used
                    )
                    self._update_status()
                self._turn_completion_tokens += int(p.get("completion_tokens", 0) or 0)
                total = int(p.get("total_tokens", 0) or 0)
                if total > 0:
                    self._turn_total_tokens += total
                # Reconcile the live "↓ N tokens" counter with REAL usage.
                # Between rounds `_turn_tokens` only holds a char/4 estimate of
                # the LATEST round's streamed output (content + thinking + tool
                # args). Across a multi-round turn that estimate undercounts the
                # cumulative decode badly — the user watches "↓ 200 tokens" all
                # turn, then the final summary reports e.g. "out: 2.3k". Once a
                # round closes we have llama-server's real completion count for
                # every round so far (`_turn_completion_tokens`); snap the live
                # counter up to it so the badge tracks the same cumulative total
                # the summary will show. max() keeps the in-flight char estimate
                # for the round currently decoding (no real usage yet) so the
                # badge still advances live, and never regresses.
                self._turn_tokens = reconcile_live_tokens(
                    self._turn_tokens, self._turn_completion_tokens
                )
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
            # Close telemetry FIRST so we have the per-tool elapsed to render on
            # the done row (like codex `• {duration}` / claude-code `(49s)`).
            _dur_ms = 0
            if self._telemetry_turn is not None:
                _dur_ms = self._telemetry_turn.tool_finished(len(str(result)), is_error) or 0
            _dur = f"  · {_fmt_tool_duration(_dur_ms)}" if _dur_ms >= 1000 else ""
            # Always render ✓ green for success, ● + error for failure
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
                    _s = f"--- {file_path}" if file_path else ""
                    log.append_tool_done(name, args, f"{_s}{_dur}")
                    log.append_tool_result(result)
                else:
                    # Summarize by MAGNITUDE (N lines / N matches / N files), not
                    # the first line of output — matching claude-code/codex/opencode
                    # which show the result's shape. `_brief_result` is the same
                    # summarizer the CLI uses, so both surfaces agree, and it also
                    # cleans internal markers ([exit code], REJECTED, etc.).
                    from ...agent.helpers import _brief_result
                    summary = _brief_result(name, result)
                    log.append_tool_done(name, args, f"{summary}{_dur}")
            self._thinking_phase = ""
            # Show thinking indicator immediately after tool completion
            # to cover the gap while the model processes the result
            self._show_active_thinking("thinking")
        elif t == "thinking_start":
            self._thinking_phase = "thinking"
            self._thinking_text = ""
            self._thinking_streamed = False
            self._show_active_thinking("thinking")
        elif t == "thinking_chunk":
            chunk = p.get("chunk", "")
            self._thinking_text += chunk
            self._turn_tokens += max(1, len(chunk) // 4) if chunk else 0
            self._thinking_phase = "thinking"
            # Stream the reasoning to the log live (like Claude Code / Codex)
            # instead of hiding it. On the first chunk, drop the spinner and
            # start the dimmed reasoning stream; then feed each chunk.
            if chunk:
                if not self._thinking_streamed:
                    self._hide_active_step()
                    self._thinking_streamed = True
                log.stream_thinking(chunk)
            elif self._active_mode != "thinking" and not self._thinking_streamed:
                self._show_active_thinking("thinking")
        elif t == "thinking_peek":
            self._thinking_phase = "thinking"
            # thinking_peek carries model text in p["text"]; deliberately
            # ignore it — the live stream (thinking_chunk) shows the real text.
            if self._active_mode != "thinking" and not self._thinking_streamed:
                self._show_active_thinking("thinking")
        elif t == "thinking_done":
            text = p.get("text", "")
            self._thinking_text = text
            self._hide_active_step()
            # If we already streamed the reasoning live, it's on screen — just
            # close the stream. Only fall back to the collapsible block when
            # nothing streamed (e.g. a model that emits thinking only at the end).
            if self._thinking_streamed:
                log.end_thinking_stream()
            elif text.strip():
                log.append_thinking(text, expanded=self._thinking_expanded)
        elif t == "stream_start":
            self._thinking_phase = "generating"
            # Show localcode-themed animation while generating response
            self._show_active_thinking("generating")
            # Hide any active tool step animation
            if self._active_mode == "tool":
                self._hide_active_step()
        elif t == "notice":
            # User-facing notice (e.g. why a turn ended). Close any live
            # reasoning stream first so the notice isn't dimmed/indented under it.
            if getattr(self, "_thinking_streamed", False):
                log.end_thinking_stream()
                self._thinking_streamed = False
            text = p.get("text", "")
            if text:
                log.append_info(text)
        elif t == "error":
            # Close a dangling reasoning stream so the error isn't dimmed/indented.
            if getattr(self, "_thinking_streamed", False):
                log.end_thinking_stream()
                self._thinking_streamed = False
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
            inp = self.query_one("#chat-input", _ChatTextArea)
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
        inp = self.query_one("#chat-input", _ChatTextArea)
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
                self._slash_selected = self._next_selectable_slash(
                    self._slash_selected, +1
                )
                self._render_slash_menu()
                event.prevent_default()
                event.stop()
                return
            elif key == "up":
                self._slash_selected = self._next_selectable_slash(
                    self._slash_selected, -1
                )
                self._render_slash_menu()
                event.prevent_default()
                event.stop()
                return
            elif key == "enter":
                # Select the highlighted command and submit it. (With the
                # old single-line Input we just set the value and let
                # Input.Submitted fire; the TextArea has no equivalent
                # auto-submit, so clear the menu and submit explicitly.)
                cmd = self._slash_matches[self._slash_selected][0]
                # Disabled (greyed-out) commands are non-selectable — e.g.
                # /thinking on a model with no hidden-reasoning channel.
                if self._slash_cmd_disabled(cmd):
                    self.query_one("#chat-log", ChatLog).append_info(
                        f"{cmd} isn't available for the current model."
                    )
                    event.prevent_default()
                    event.stop()
                    return
                inp = self.query_one("#chat-input", _ChatTextArea)
                inp.text = cmd
                self._slash_matches = []
                self._slash_selected = 0
                self.query_one("#slash-menu", Static).remove_class("active")
                self.query_one("#status-bar", Static).remove_class("hidden")
                event.prevent_default()
                event.stop()
                self._submit_message(cmd)
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
        elif key == "4":
            # Stop asking entirely: approve this AND flip the session to
            # FULL_AUTO so nothing prompts for the rest of the session.
            # Same mechanism as `/permissions` toggling off; re-enable with
            # `/permissions`.
            verdict, label = "always", "permissions OFF — auto-approving everything this session (/permissions to re-enable)"
            try:
                _eng = self.tui.engine
                if _eng is not None:
                    _eng._autonomy = AutonomyLevel.FULL_AUTO
                    apply_autonomy_to_permissions(_eng.perms, get_policy(AutonomyLevel.FULL_AUTO))
            except Exception:
                pass

        if verdict is not None:
            # Clear the gate FIRST so any exception below can't leave the
            # input disabled forever; then deliver the verdict + re-enable.
            self._awaiting_approval = False
            try:
                log.append_info(f"  └ {label}")
                self.tui.bridge.set_approval(verdict)
            finally:
                inp = self.query_one("#chat-input", _ChatTextArea)
                inp.disabled = False
                inp.focus()
                # Reflect a permissions-off flip (key "4") in the status bar now.
                self._update_status()
        else:
            # Invalid key while a decision is required — silence reads as
            # "frozen / broken". Ring the bell and re-show the choices so
            # the user knows the prompt is live and what to press.
            try:
                self.app.bell()
            except Exception:
                pass
            log.append_info("  [dim]press 1/y allow · 2 always · 3/n/Esc deny · 4 stop asking[/]")

        # Block ALL other keys during approval
        event.prevent_default()
        event.stop()
