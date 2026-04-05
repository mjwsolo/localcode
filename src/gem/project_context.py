"""Gem project context — reads GEM.md and .gem/context.md for per-project instructions."""
from __future__ import annotations

from pathlib import Path


# Files checked in priority order (first found wins, but all are included)
CONTEXT_FILES = [
    "GEM.md",
    ".gem/context.md",
    ".gem/instructions.md",
]

IGNORE_FILE = ".gem/ignore"


def load_project_context(repo_root: Path) -> str:
    """Load per-project context from GEM.md or .gem/context.md.

    Returns the combined content of all found context files,
    or empty string if none exist.
    """
    sections: list[str] = []
    for filename in CONTEXT_FILES:
        path = repo_root / filename
        if path.is_file():
            try:
                content = path.read_text(errors="replace").strip()
                if content:
                    sections.append(f"# Project instructions ({filename})\n{content}")
            except Exception:
                continue
    return "\n\n".join(sections)


def load_ignore_patterns(repo_root: Path) -> list[str]:
    """Load .gem/ignore patterns (one glob per line, like .gitignore)."""
    path = repo_root / IGNORE_FILE
    if not path.is_file():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    except Exception:
        return []


def has_project_context(repo_root: Path) -> bool:
    """Check if any project context files exist."""
    return any((repo_root / f).is_file() for f in CONTEXT_FILES)
