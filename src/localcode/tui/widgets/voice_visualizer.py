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
        width: auto;
        background: ansi_default;
        padding: 0 1;
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
        # Typical mic peaks for normal speech sit at 0.05-0.3 of full
        # scale. To make the bar actually fill the cell, we boost by 8×
        # and clamp. EMA smoothing for stability but biased heavier
        # toward the current sample (0.7) so the bar feels responsive.
        target = min(1.0, peak * 8.0)
        self._smoothed = 0.3 * self._smoothed + 0.7 * target

        # 9-step vertical fill (empty → full block). Floor at index 2
        # so silence still shows a small "I'm listening" baseline; ceil
        # at len-1 (= █ = full cell height).
        idx = max(2, int(self._smoothed * (len(_FILL) - 1)))
        char = _FILL[idx]

        # Color: rainbow cycle over time. When amplitude is high, jump
        # ahead in the palette to give an extra "energy spike" feel.
        t = time.time() - self._t0
        color = _PALETTE[(int(t * 3) + int(self._smoothed * 4)) % len(_PALETTE)]

        bar = "".join(f"[{color}]{char}[/]" for _ in range(_BAR_WIDTH))
        self.update(f"{bar}")
