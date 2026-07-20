"""Detect a degenerate reasoning loop in the model's thinking stream.

Small quantized local models sometimes collapse the reasoning channel into a
repetition loop: the same short phrase ("let me read the file", "OK let me call
read_file") is emitted over and over without ever transitioning to a tool call.
Token-level sampler penalties (presence/repeat) don't reliably break a tight
periodic oscillation, and the length cap (MAX_THINKING_CHARS) only fires after
tens of thousands of wasted characters — minutes later on a local model.

Detection is by STRUCTURE, not content: we look for exact *periodicity* in the
tail of the reasoning stream — a substring that repeats with a fixed period.
That is what an autoregressive degeneration loop actually is, and it catches the
phrase in any language on any task within a few repetitions.

Design notes (why this shape):
- Operate on a raw character tail, NOT on split lines. The reasoning arrives as
  arbitrary streaming chunks; the repeated unit may contain no newline and the
  chunk boundaries fall differently each time, so line-splitting is fragile.
- Anchor on the final n-gram and find its previous occurrence to derive the
  period in one `str.rfind` (C-fast), then verify true periodicity by scanning
  back. Cost is O(window), independent of total reasoning length, so the caller
  can pass a bounded tail buffer and stay linear overall.
- Require the periodic run to be both long enough (absolute chars) and to repeat
  enough times before firing, so genuine reasoning — which is not exactly
  periodic — is never cut.
"""
from __future__ import annotations

__all__ = ["reasoning_is_looping"]

# Need this many chars of uninterrupted reasoning before we judge at all — a
# floor against cutting short, legitimately-repetitive planning.
_MIN_TAIL = 300
# Rolling tail we actually score. The caller passes a bounded buffer; we also
# clamp here so a full-transcript caller stays cheap.
_WINDOW = 2400
# Length of the trailing n-gram we anchor on to recover the period. Long enough
# to be specific (won't coincidentally reappear in prose), short enough that a
# tight loop still contains two copies inside the window.
_ANCHOR = 32
# Only fire early on TIGHT loops; a very long period is better left to the
# length/time cap (and is far likelier to be genuine varied reasoning).
_MAX_PERIOD = 400
# The periodic unit must repeat at least this many times...
_MIN_REPEATS = 4
# ...spanning at least this many characters. Whichever is larger gates.
_MIN_SPAN = 240


def reasoning_is_looping(text: str) -> bool:
    """True when the tail of `text` has collapsed into an exact periodic loop.

    Content-agnostic and chunk-boundary robust. Conservative: returns False on
    short input and only fires on a genuinely periodic, sufficiently-repeated
    tail, so varied reasoning is never aborted.
    """
    if not text or len(text) < _MIN_TAIL:
        return False
    tail = text[-_WINDOW:]
    n = len(tail)
    if n < _ANCHOR * 2:
        return False
    anchor = tail[-_ANCHOR:]
    # Where did the final n-gram last occur before its trailing copy? The gap
    # between the two occurrences is the loop period.
    prev = tail.rfind(anchor, 0, n - _ANCHOR)
    if prev == -1:
        return False
    period = (n - _ANCHOR) - prev
    if period <= 0 or period > _MAX_PERIOD:
        return False
    # Verify the tail really is periodic with this period, measuring how far the
    # exact repetition extends back from the end.
    span = 0
    i = n - 1
    while i - period >= 0 and tail[i] == tail[i - period]:
        span += 1
        i -= 1
    return span >= max(_MIN_SPAN, _MIN_REPEATS * period)
