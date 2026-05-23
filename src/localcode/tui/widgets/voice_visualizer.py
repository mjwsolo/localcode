"""Live audio volume indicator — a single tall narrow bar.

Renders one short row containing a wide block character whose
*fill level* changes with the mic's instantaneous volume, plus a
small REC timer to the right. The bar's color cycles through a
rainbow palette over time so even silence has a quiet pulse.

Width is fixed (a few cells) — this isn't a waveform history strip,
it's a single VU-meter-style bar. Pulls `peak` straight off the
Recorder instance owned by the chat screen. Refreshes at ~30 FPS
when active; hidden the rest of the time.
"""
from __future__ import annotations

import time
from typing import Any

from textual.widgets import Static


# Vertical fill levels from empty to full. Each character occupies the
# bottom Nth of a single text cell, so reading left-to-right these
# look like a single bar growing from the bottom of the line up.
_FILL = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

# Curated rainbow that reads well on both dark and light terminals.
_PALETTE = [
    "#ff5470", "#ff8a5b", "#ffd166", "#9bff8a", "#5bffc1",
    "#5bd1ff", "#5b96ff", "#9b5bff", "#e75bff", "#ff5bd1",
]

# Bar is one cell wide — reads as a single thin pulsing bar at a glance.
_BAR_WIDTH = 1


class VoiceVisualizer(Static):
    """A small VU-meter-style bar that breathes with mic input."""

    DEFAULT_CSS = """
    VoiceVisualizer {
        height: 1;
        width: 1;                /* exactly one cell — no phantom column */
        background: transparent;
        padding: 0;
        margin: 0 0 0 1;         /* one space gap from input, nothing else */
        display: none;          /* hidden when not recording */
    }
    VoiceVisualizer.active {
        display: block;
    }
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("", *args, **kwargs)
        self.recorder: Any = None
        self._t0 = time.time()
        self._timer = None
        # Smoothed amplitude — EMA so the bar's height doesn't jitter
        # every frame. New samples weighted at 0.6, prior at 0.4.
        self._smoothed = 0.0

    # ── lifecycle ─────────────────────────────────────────────────

    def activate(self, recorder: Any) -> None:
        """Start the 30-FPS render loop pointed at `recorder`."""
        self.recorder = recorder
        self._smoothed = 0.0
        self._t0 = time.time()
        self.add_class("active")
        if self._timer is None:
            self._timer = self.set_interval(1 / 30, self._tick)

    def deactivate(self) -> None:
        """Stop the timer and hide the widget."""
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None
        self.recorder = None
        self.remove_class("active")
        self.update("")

    # ── render ────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self.recorder is None:
            return
        try:
            peak = float(getattr(self.recorder, "peak", 0.0) or 0.0)
        except Exception:
            peak = 0.0
        target = min(1.0, peak * 8.0)
        self._smoothed = 0.3 * self._smoothed + 0.7 * target

        # ALWAYS render the full-block char (█). The cell is fully
        # filled — no "bottom gray" because we don't use partial-height
        # fill chars anymore (they left the unfilled portion as the
        # terminal background). Amplitude is shown by COLOR INTENSITY:
        # quiet = dim, loud = bright. Color also cycles through the
        # rainbow palette over time.
        t = time.time() - self._t0
        # Cycle through palette; amplitude jumps phase forward.
        phase = int(t * 3) + int(self._smoothed * 4)
        base_color = _PALETTE[phase % len(_PALETTE)]

        # Brightness modulation — at silence, mix toward dim grey; at
        # peak, full saturation. Linear blend on RGB.
        def _hex_to_rgb(h):
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = _hex_to_rgb(base_color)
        # Floor at 0.35 so silence still shows a visible glow (not pitch
        # black, which would read as "off"). Loud peaks pull to 1.0.
        intensity = 0.35 + 0.65 * self._smoothed
        r = int(r * intensity)
        g = int(g * intensity)
        b = int(b * intensity)
        color = f"#{r:02x}{g:02x}{b:02x}"

        self.update(f"[{color}]█[/]")
