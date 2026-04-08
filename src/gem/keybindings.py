"""LocalCode keybindings — VIM mode + user-customizable shortcuts.

VIM mode: toggle with /vim command. Provides h/j/k/l, i/a/ESC.
Custom keybindings: ~/.localcode/keybindings.json overrides defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

from prompt_toolkit.enums import EditingMode

from .config import get_home_dir


KEYBINDINGS_FILE = get_home_dir() / "keybindings.json"

DEFAULT_KEYBINDINGS = {
    "submit": "enter",
    "cancel": "c-c",
    "clear": "c-l",
    "history_prev": "up",
    "history_next": "down",
}


def load_keybindings() -> dict[str, str]:
    """Load user keybindings, merged with defaults."""
    bindings = DEFAULT_KEYBINDINGS.copy()
    if KEYBINDINGS_FILE.exists():
        try:
            user = json.loads(KEYBINDINGS_FILE.read_text())
            bindings.update(user)
        except Exception:
            pass
    return bindings


def save_keybindings(bindings: dict[str, str]) -> Path:
    """Save keybindings to user config."""
    KEYBINDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEYBINDINGS_FILE.write_text(json.dumps(bindings, indent=2))
    return KEYBINDINGS_FILE


def get_editing_mode(vim_enabled: bool) -> EditingMode:
    """Get prompt_toolkit editing mode."""
    return EditingMode.VI if vim_enabled else EditingMode.EMACS
