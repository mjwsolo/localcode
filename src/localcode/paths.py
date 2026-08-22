"""Single source of truth for where LocalCode reads/writes state on disk.

Two-tier layout:

  <project_root>/.localcode/        ← per-project state (this module's main job)
      lifecycle.log
      last-pressure-kill.txt
      last_error.log
      notebook/<session>/
      test-results/<timestamp>/
      sessions/      (future move)
      history.db     (future move)

  ~/.localcode/                     ← truly-global, machine-wide state
      llama-server.pid              (one server runs at a time, system-wide)
      stuck-server.txt              (D-state marker, system-wide)
      config.toml                   (user's preferences across projects)
      models cache                  (10 GB+ files, can't duplicate per project)

Why split this way
------------------
Per-project: anything specific to "what terminal coding tools did in THIS codebase" — sessions,
test results, lifecycle of THIS project's runs, the agent's notebook scratch.
You can `git ignore .localcode/` and the project tree stays clean. When you move
or rename a project, its conversation history comes with it.

Global: anything that's a singleton on the machine — the running llama-server
(only one on port 8081), the model files (10 GB each), and your global
preferences (autonomy level, default model). Duplicating these per project
would waste disk and create state confusion.

Project-root detection
----------------------
Walks up from cwd looking for, in priority order:
  1. existing `.localcode/` directory (explicit, the user already has one here)
  2. `.git/` (most common — git repo root)
  3. `LOCALCODE.md` (explicit localcode project marker)
  4. Standard project files (pyproject.toml, package.json, Cargo.toml, go.mod)
Falls back to the cwd itself if nothing is found, so launching from `/tmp` or
`~` still works (you just get `<cwd>/.localcode/` wherever you ran from).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


GLOBAL_STATE_DIR = Path.home() / ".localcode"

_PROJECT_MARKERS = (
    ".localcode",
    ".git",
    "LOCALCODE.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
)


def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk up from `start` (default: cwd) looking for a project marker.

    Returns the deepest ancestor that contains one of `_PROJECT_MARKERS`.
    If none found, returns `start` (or cwd) — guarantees a usable path.
    """
    p = (start or Path.cwd()).resolve()
    for ancestor in (p, *p.parents):
        for marker in _PROJECT_MARKERS:
            if (ancestor / marker).exists():
                return ancestor
    return p


def project_state_dir(cwd: Optional[Path] = None) -> Path:
    """Return (and create) `<project_root>/.localcode/`.

    Idempotent — creating an existing directory is a no-op. The
    directory is materialised on first access so callers can
    immediately write into it without checking existence.
    """
    d = find_project_root(cwd) / ".localcode"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_state_dir() -> Path:
    """Return (and create) `~/.localcode/`.

    Reserved for machine-wide singletons (llama-server PID, stuck
    server marker, user-global config, model cache). Anything that
    belongs to a specific project should NOT live here.
    """
    GLOBAL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_STATE_DIR


# ── Specific artifact paths ──────────────────────────────────────────
#
# Each helper returns the canonical path for one artifact. Consumers
# should import the helper, never hardcode `Path.home() / ".localcode"`.
# That way moving an artifact between project / global tiers requires
# changing exactly one line here, not hunting through the codebase.


def lifecycle_log_path(cwd: Optional[Path] = None) -> Path:
    """DEPRECATED — use `events_log_path()`. Kept as a back-compat shim
    while in-flight references are migrated; will be removed once the
    consolidation lands fully and no caller imports this name."""
    return project_state_dir(cwd) / "lifecycle.log"


def events_log_path(cwd: Optional[Path] = None) -> Path:
    """Single source of truth for ALL structured developer events:
    server lifecycle, tool calls, redactions, pressure kills, stall
    recovery, turn start/end. JSONL — one event per line, each line is
    valid JSON with at minimum `t` (ISO timestamp) and `type` fields.

    Per-project (lives at `<project_root>/.localcode/events.jsonl`)
    so each project gets its own audit trail. Append-only; never
    truncated. Local-only: nothing is uploaded anywhere. Designed to be
    tail-able with `jq` for live debugging. Any future remote sink would
    be a new opt-in feature that must strip user content first, not a
    flag flip."""
    return project_state_dir(cwd) / "events.jsonl"


def pressure_kill_marker_path(cwd: Optional[Path] = None) -> Path:
    """Marker the pressure monitor writes when it SIGTERMs the server.
    Per-project so an error message in project A doesn't reference a
    pressure-kill that happened while you were working on project B."""
    return project_state_dir(cwd) / "last-pressure-kill.txt"


def last_error_log_path(cwd: Optional[Path] = None) -> Path:
    """Verbose technical detail of the most recent user-facing error
    in THIS project — referenced by the error formatter's "(full
    technical detail saved to ...)" line."""
    return project_state_dir(cwd) / "last_error.log"


def notebook_root(cwd: Optional[Path] = None) -> Path:
    """Per-project notebook root. Each session creates a subdirectory
    `<notebook_root>/<session_id>/`. Per-project so the agent's working
    drafts for project A aren't visible / reachable from project B."""
    return project_state_dir(cwd) / "notebook"


def test_results_root(cwd: Optional[Path] = None) -> Path:
    """Where `tests/e2e/run.py` writes per-run reports. Per-project
    because test results pertain to the project being tested, and they
    should be `git ignore`-able (and visible in the project tree)."""
    return project_state_dir(cwd) / "test-results"


# ── Global-only paths ────────────────────────────────────────────────


def server_pid_file() -> Path:
    """llama-server PID file. STAYS GLOBAL because only one
    llama-server can run on the machine at a time (one port, one chunk
    of GPU memory). If you `cd` between projects mid-session the
    running server stays serving you — the per-project lifecycle log
    just records a fresh `server_started` entry on next launch from
    the new project."""
    return global_state_dir() / "llama-server.pid"


def stuck_server_marker_path() -> Path:
    """D-state marker for a server that won't die. Global because the
    stuck process is a system-level concern, not a project concern."""
    return global_state_dir() / "stuck-server.txt"


# ── Write containment ────────────────────────────────────────────────
#
# Every file-writing tool funnels its `path` argument through
# `contain_write_path()`. Before this existed, `ToolContext.resolve_path`
# computed `repo / raw` and returned it untouched — and because
# `Path.__truediv__` DISCARDS the left operand when the right one is
# absolute, `raw="/Users/victim/.zshrc"` produced exactly that path. A
# `../../..` prefix escaped just as easily, and a symlink committed inside
# the repo pointing outside was a third route (pathlib writes follow
# symlinks). Reads are deliberately NOT contained — the agent is expected to
# read anywhere on the machine — but writes now stay inside the project.


class PathContainmentError(ValueError):
    """A write was aimed outside the project root (or an allowed extra root)."""

    def __init__(self, raw: str, resolved: Path, root: Path):
        self.raw = raw
        self.resolved = resolved
        self.root = root
        super().__init__(
            f"path escapes the project root: {raw!r} resolves to {resolved} "
            f"which is outside {root}. Write only inside the project."
        )


def is_within(path: Path, root: Path) -> bool:
    """True if `path`, fully resolved, lives at or under `root`.

    Symlink-aware in both directions: `.resolve()` follows every component
    (including the final one), so a symlink inside the repo whose target
    escapes resolves to the escaping target and returns False. Resolving
    `root` too keeps `/tmp` → `/private/tmp` style prefixes from producing
    false negatives on macOS.
    """
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


def contain_write_path(
    candidate: Path | str,
    root: Path | str,
    extra_roots: "tuple[Path, ...] | list[Path] | None" = None,
) -> Path:
    """Return the resolved write target, or raise `PathContainmentError`.

    `candidate` is the already-healed absolute path (see
    `ToolContext.resolve_path`); `root` is the project root. `extra_roots`
    are additional sanctioned trees (the agent's notebook scratch dir) that
    may legitimately sit outside the project.
    """
    cand = Path(candidate)
    try:
        resolved = cand.resolve()
    except (OSError, ValueError):
        resolved = cand
    roots = [Path(root)]
    for extra in (extra_roots or ()):
        roots.append(Path(extra))
    for r in roots:
        if is_within(resolved, r):
            return resolved
    try:
        root_resolved = Path(root).resolve()
    except (OSError, ValueError):
        root_resolved = Path(root)
    raise PathContainmentError(str(candidate), resolved, root_resolved)
