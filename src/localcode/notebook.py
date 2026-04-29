"""Session-scoped notebook directory for the agent's working files.

Motivation — reduce in-context memory bloat
--------------------------------------------
Without a notebook, every intermediate draft, plan, half-finished script, or
piece of working data lived in the conversation history: written via
`write_file` into the user's project tree (polluting the repo) OR shoved
into a content message the model then has to re-read every round. Both
approaches leak context: project writes stick around forever and surprise
the user with random files; content-in-history grows the prompt linearly
and eats the window the model needs for the actual task.

The notebook is a per-session directory at
`~/.localcode/notebook/<session-id>/` that the model treats as its own
working memory. It uses regular `write_file`, `read_file`, and `bash`
against this path — no new tool — but the system prompt points the model
at the directory and tells it "use this for drafts, plans, intermediate
data; the user's project tree is for final output only."

Benefits:
  • Drafts and intermediate data don't pollute the user's repo.
  • The model can offload working memory to disk instead of re-emitting
    it into the chat context every round.
  • Trivial cleanup — wipe the session directory on exit.
  • Writes / edits inside the notebook are auto-approved at the
    permission layer — see `PermissionManager.check` in
    permissions_v2.py — regardless of autonomy mode (`suggest`,
    `auto_edit`, `full_auto`). `is_within_notebook` resolves
    symlinks and rejects path-traversal (`../..`), so the bypass
    cannot be abused to write into the user's project tree.

Design notes:
  • One directory per session — session_id is generated once at app start
    (UUID-based) and persists for the lifetime of that LocalCodeApp.
  • Parent directory `~/.localcode/notebook/` persists across sessions so
    stale session dirs from crashes can be garbage-collected later. For
    now, the most recent N are kept on disk for post-mortem debugging.
  • Does NOT persist across sessions by default — fresh session, fresh
    directory. If we later want long-term memory, that's a separate
    mechanism on top of this scratchpad.
  • Writes to the notebook never prompt for permission (see
    `is_within_notebook` check in the write/edit tool dispatchers).
"""
from __future__ import annotations

from pathlib import Path

from .paths import notebook_root as _notebook_root


# Per-project notebook root: `<project_root>/.localcode/notebook/`.
# Computed on first access (so `cd` to a different project picks up the
# new location). Resolved lazily via the property below; do not cache
# the value at module import time.
def _root() -> Path:
    return _notebook_root()


# Backwards-compat shim: code that imported `NOTEBOOK_ROOT` previously
# got a `Path`. Anyone reading the module-level constant now gets the
# CURRENT project's notebook root resolved at access time. If you need
# to pin a value (e.g., for the duration of a session), capture it
# into a local variable.
NOTEBOOK_ROOT = _root()
# Cap on the number of historical session directories we keep on disk.
# Each one holds a few small files at most, so 20 is cheap and gives a
# useful debugging tail without unbounded growth.
MAX_HISTORICAL_SESSIONS = 20


def notebook_dir_for(session_id: str) -> Path:
    """Return (and create) the notebook directory for `session_id`.

    Creating on access is idempotent — subsequent calls just return the
    existing path. Safe to call from any thread.
    """
    # Resolve the notebook root LIVE (not the cached module-level
    # NOTEBOOK_ROOT) so the lookup follows the current project even if
    # the user changed working directories since import time.
    d = _root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_within_notebook(path: Path) -> bool:
    """True if `path` resolves inside any session's notebook directory.

    Used by the tool-permission layer to skip the approval prompt for
    writes into the notebook. A path traversal (`../..`) from inside the
    notebook will resolve outside the notebook root and correctly return
    False, so the user's project is still protected from malicious or
    confused tool calls.
    """
    try:
        resolved = path.resolve()
        return resolved.is_relative_to(_root().resolve())
    except (OSError, ValueError):
        return False


def gc_old_sessions(keep: int = MAX_HISTORICAL_SESSIONS) -> None:
    """Delete all but the `keep` most-recently-modified session directories.

    Best-effort — silent on any filesystem error. Safe to call on every
    app start; idempotent when under the cap.
    """
    root = _root()
    if not root.is_dir():
        return
    try:
        subdirs = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return
    if len(subdirs) <= keep:
        return
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in subdirs[keep:]:
        try:
            # rmtree without importing shutil at module level (cold path).
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
