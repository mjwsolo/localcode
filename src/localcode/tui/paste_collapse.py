"""Large-paste → placeholder-chip collapsing for the chat composer.

Deliberately free of any Textual import so it can be unit-tested
headlessly (this repo's dev env has no `textual`) and so the chat screen
stays the only Textual-coupled module. The chat ``TextArea`` owns one
:class:`PasteBuffer`; a large paste is replaced in the *visible* composer
by a compact chip like ``[pasted #1 +42 lines]`` and the real text is
spliced back in at submit time.

Peer grounding (every leading CLI agent does this):
  * claude-code  ``[Pasted text #1 +400 lines]``
  * pi           ``[paste #1 +123 lines]``
  * opencode     ``[Pasted ~N lines]``
"""
from __future__ import annotations

# A paste is "large" (worth collapsing to a chip) when it spans several
# lines OR is a long single-line blob. Small pastes keep their current
# inline behaviour so a two-word paste isn't hidden behind a chip.
LARGE_PASTE_MIN_LINES = 4
LARGE_PASTE_MIN_CHARS = 300


def _line_count(text: str) -> int:
    """Number of lines in ``text`` (a lone trailing newline doesn't count)."""
    if not text:
        return 0
    stripped = text[:-1] if text.endswith("\n") else text
    return stripped.count("\n") + 1


def is_large_paste(
    text: str,
    min_lines: int = LARGE_PASTE_MIN_LINES,
    min_chars: int = LARGE_PASTE_MIN_CHARS,
) -> bool:
    """True when a paste is big enough to collapse to a chip."""
    if not text:
        return False
    return _line_count(text) >= min_lines or len(text) >= min_chars


class PasteBuffer:
    """Tracks large pastes collapsed to placeholder chips in the composer.

    Each large paste gets a unique chip token spliced into the visible
    text; the real text is stored keyed by that token. At submit,
    :meth:`expand` replaces every *surviving* token with its stored text —
    tokens the user deleted from the composer simply never match, so their
    text is dropped automatically.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._counter = 0

    def __len__(self) -> int:  # convenience for tests / truthiness
        return len(self._store)

    def add(self, text: str) -> str:
        """Register a large paste; return the chip token to insert inline."""
        self._counter += 1
        n = self._counter
        lines = _line_count(text)
        if lines <= 1:
            # A long single-line blob reads better counted in chars than
            # as "+1 lines".
            token = f"[pasted #{n} +{len(text)} chars]"
        else:
            token = f"[pasted #{n} +{lines} lines]"
        self._store[token] = text
        return token

    def expand(self, composed: str) -> str:
        """Replace surviving chip tokens in ``composed`` with real text.

        Tokens the user deleted never match and are dropped.
        """
        if not self._store or not composed:
            return composed
        for token, real in self._store.items():
            if token in composed:
                composed = composed.replace(token, real)
        return composed

    def prune(self, current_text: str) -> None:
        """Drop stored pastes whose chip no longer appears in ``current_text``."""
        for token in list(self._store):
            if token not in current_text:
                del self._store[token]

    def clear(self) -> None:
        """Forget every stored paste (called after a message is submitted)."""
        self._store.clear()
        self._counter = 0
