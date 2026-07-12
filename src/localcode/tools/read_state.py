"""Per-session read-state registry — the read-before-edit staleness guard.

Ports claude-code's `FileEditTool.readFileState` pattern (a map of
absolute-path -> {content-hash, mtime, full/partial}) and folds in the
compare-on-write idea from codex/opencode V2 (`writeIfUnchanged`).

Why a guard at all
------------------
A surgical edit (`edit_file`/`multi_edit`) only makes sense against the
CURRENT bytes of a file. If the model never read the file, or read it and
the file has since changed on disk (its own earlier write, a formatter, an
AV/cloud-sync touch, another process), then `old_string` is being matched
against text the model has NOT seen — the classic silent-clobber failure.
claude-code refuses such edits until the file is (re)read.

Two subtleties copied from claude-code
--------------------------------------
* A PARTIAL read (offset/limit or a truncated large-file read) does NOT
  satisfy the guard — the model hasn't seen the whole file, so its anchor
  may sit in an unseen region.
* CONTENT-EQUALITY fallback: if the mtime advanced but the bytes are
  byte-identical to what was read, we do NOT false-reject (a formatter or
  AV touch that rewrites the file with the same content bumps mtime).

Session scoping
---------------
State lives on the app object (`app._edit_read_state`), so it is naturally
scoped to one agent session and a fresh app (e.g. a unit test that calls
`edit_file` directly) starts empty. The guard only ENFORCES once the
session has recorded at least one read (`is_armed`): with no read on record
there is no freshness baseline, so enforcing would be pure friction. In a
real session `read_file` fires constantly, arming the guard immediately.

After a successful write/edit the tool calls `record_write`, so the model's
own writes refresh the baseline and never trip the guard on the next round.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_ATTR = "_edit_read_state"

# Files larger than this are not hashed (staleness still works off mtime;
# only the content-equality fallback is skipped for very large files).
_MAX_HASH_BYTES = 4 * 1024 * 1024


def _registry(app: Any) -> dict[str, dict]:
    reg = getattr(app, _ATTR, None)
    if not isinstance(reg, dict):
        reg = {}
        try:
            setattr(app, _ATTR, reg)
        except Exception:
            # Slotted/frozen app: fall back to a module-level per-id map so
            # the guard still functions instead of silently disabling.
            reg = _fallback_registry(app)
    return reg


_FALLBACK: dict[int, dict] = {}


def _fallback_registry(app: Any) -> dict[str, dict]:
    return _FALLBACK.setdefault(id(app), {})


def _key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_meta(path: Path) -> tuple[float, int, str | None]:
    st = path.stat()
    sha: str | None = None
    if st.st_size <= _MAX_HASH_BYTES:
        try:
            sha = _hash_bytes(path.read_bytes())
        except OSError:
            sha = None
    return st.st_mtime, st.st_size, sha


def record_read(app: Any, path: Path, *, full: bool) -> None:
    """Record that `path` was read this session.

    `full` must be False for a partial read (offset/limit or a truncated
    large-file view) — a partial read must NOT satisfy the guard.
    """
    try:
        mtime, size, sha = _file_meta(path)
    except OSError:
        return
    _registry(app)[_key(path)] = {
        "mtime": mtime,
        "size": size,
        "sha": sha,
        "full": bool(full),
    }


def record_write(app: Any, path: Path, content: str) -> None:
    """Refresh read-state after a successful write/edit (claude-code write-back)
    so the model's own writes never trip the guard next round."""
    try:
        mtime, size, _ = _file_meta(path)
    except OSError:
        mtime, size = 0.0, len(content.encode("utf-8", "replace"))
    _registry(app)[_key(path)] = {
        "mtime": mtime,
        "size": size,
        "sha": _hash_bytes(content.encode("utf-8", "replace")),
        "full": True,
    }


def is_armed(app: Any) -> bool:
    """True once the session has ≥1 read on record. The guard only enforces
    when armed (see module docstring)."""
    reg = getattr(app, _ATTR, None)
    if isinstance(reg, dict) and reg:
        return True
    return bool(_FALLBACK.get(id(app)))


def check(app: Any, path: Path) -> str | None:
    """Return None if editing `path` is allowed, else a reason code:

      * 'unread'  — never read this session
      * 'partial' — only a partial (offset/limit/truncated) read on record
      * 'stale'   — file changed on disk since it was read

    Only meaningful when `is_armed(app)` is True.
    """
    entry = _registry(app).get(_key(path))
    if entry is None:
        return "unread"
    if not entry.get("full"):
        return "partial"
    try:
        st = path.stat()
    except OSError:
        # File vanished — let the tool's own not-found handling deal with it.
        return None
    # Fresh: mtime has not advanced past the recorded read.
    if st.st_mtime <= entry["mtime"] + 1e-6:
        return None
    # mtime advanced — content-equality fallback (formatter / AV / cloud-sync
    # touch that changes mtime but not the bytes). Only applies to full reads,
    # which are the only ones with a stored hash.
    sha = entry.get("sha")
    if sha is not None:
        try:
            if _hash_bytes(path.read_bytes()) == sha:
                # Same bytes — refresh mtime so we don't re-hash every round.
                entry["mtime"] = st.st_mtime
                entry["size"] = st.st_size
                return None
        except OSError:
            return None
    return "stale"


def guard_edit(app: Any, path: Path, display_path: str) -> str | None:
    """Read-before-edit guard for surgical edits (edit_file / multi_edit).

    Returns an actionable error string to hand back to the model, or None to
    allow the edit. No-op unless the session is armed.
    """
    if not is_armed(app):
        return None
    reason = check(app, path)
    if reason == "unread":
        return (
            f"Error: read {display_path} before editing it. You are editing a "
            f"file you have not read this session, so `old_string` is being "
            f"matched against text you have not seen. Call read_file "
            f"path={display_path!r} first, then edit with an exact anchor."
        )
    if reason == "partial":
        return (
            f"Error: you only read PART of {display_path} (offset/limit or a "
            f"truncated view). Read the whole file — or at least the full "
            f"region around your edit — before editing, so `old_string` "
            f"matches the current text. Call read_file path={display_path!r}."
        )
    if reason == "stale":
        return (
            f"Error: {display_path} changed on disk since you read it (an edit "
            f"elsewhere, a formatter, or another process). Your `old_string` "
            f"may no longer match. Re-read it (read_file path={display_path!r}) "
            f"to get the current text, then edit again."
        )
    return None


def guard_overwrite(app: Any, path: Path, display_path: str, existed: bool) -> str | None:
    """Staleness guard for write_file overwrites.

    write_file is the create-or-full-rewrite tool, so — unlike edit_file — it
    does NOT require a prior read (a fresh scaffold/rewrite is legitimate). But
    if the file WAS read this session and has since changed on disk, a blind
    overwrite would clobber those changes; refuse it. Returns an error string
    or None.
    """
    if not existed or not is_armed(app):
        return None
    if check(app, path) == "stale":
        return (
            f"Error: {display_path} changed on disk since you last read it. "
            f"Overwriting now would silently clobber those changes. Re-read it "
            f"(read_file path={display_path!r}) to see the current contents, "
            f"then rewrite intentionally."
        )
    return None
