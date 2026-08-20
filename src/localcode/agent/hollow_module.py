"""Detect a module that was created, imported, and then left empty.

The failure this exists for: a build run that wrote ten real files and one
`src/lib/fsrs.ts` containing exactly `// placeholder`, imported it from the page
that needed it, and finished. The scheduling core — the entire point of the app
— was never written, and the README advertised it as a feature.

The signal is deliberately narrow: a file whose whole body is comments AND that
something else imports. Either half alone is noisy. An empty file nobody
references is usually intentional (a package marker, a barrel not wired up yet);
a file with real content is fine however sparse. The conjunction — "you created
this module, you import it, and there is nothing in it" — is a defect
essentially every time, which is what makes it safe to block a turn on.

Kept out of `loop.py` for the same reason `project_check_gate` is: the loop has
no test seam and this needs one.
"""
from __future__ import annotations

import os
import re

__all__ = ["HollowModuleGate", "hollow_imported_modules", "is_hollow_source"]

# Families whose "no executable content" question is well defined. Interpreted
# config/markup (JSON, YAML, Markdown, CSS) is excluded: an empty one is data,
# not a missing implementation.
_C_FAMILY = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
             ".java", ".swift", ".c", ".h", ".cpp", ".hpp", ".cs", ".kt"}
_HASH_FAMILY = {".py", ".rb", ".sh", ".bash"}
_CODE_EXTS = _C_FAMILY | _HASH_FAMILY

# Files that are legitimately empty by convention — never a missing module.
_EXEMPT_NAMES = {"__init__.py", "conftest.py", "setup.py", "py.typed"}
_EXEMPT_DIR_PARTS = {"tests", "test", "__tests__", "migrations", "fixtures",
                     "node_modules", ".git", "dist", "build", "vendor"}

_MAX_BYTES = 64_000   # a file this size is not a stub; don't read further


def _strip_c_comments(text: str) -> str:
    """Remove `//` and `/* */`, preserving string literals so a `"//"` inside a
    string still counts as content."""
    out: list[str] = []
    i, n = 0, len(text)
    quote = ""
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "\"'`":
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _strip_hash_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        quote = ""
        cut = len(line)
        for idx, ch in enumerate(line):
            if quote:
                if ch == quote and line[idx - 1: idx] != "\\":
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                cut = idx
                break
        out.append(line[:cut])
    return "\n".join(out)


def is_hollow_source(path: str) -> bool:
    """True when `path` is a code file with no executable content at all.

    Comments and whitespace are stripped; anything left — including a bare
    docstring, which is at least an intentional statement — counts as content.
    A file that cannot be read is NOT reported: this must never invent a defect.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _CODE_EXTS:
        return False
    if path.endswith(".d.ts"):
        return False
    if os.path.basename(path) in _EXEMPT_NAMES:
        return False
    parts = {p.lower() for p in path.replace("\\", "/").split("/")}
    if parts & _EXEMPT_DIR_PARTS:
        return False
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            return False
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        return False
    stripped = (_strip_hash_comments(text) if ext in _HASH_FAMILY
                else _strip_c_comments(text))
    return not stripped.strip()


def _import_patterns(stem: str) -> list[re.Pattern[str]]:
    """Ways another file could name this module. Matching on the STEM keeps this
    language-agnostic; the surrounding syntax keeps it from matching prose."""
    s = re.escape(stem)
    return [
        # from './lib/fsrs' | from "../fsrs.js" | require('./fsrs')
        re.compile(rf"""(?:from|require\s*\()\s*['"][^'"]*\b{s}(?:\.[a-z]+)?['"]"""),
        # import './fsrs'
        re.compile(rf"""import\s+['"][^'"]*\b{s}(?:\.[a-z]+)?['"]"""),
        # python: import fsrs | from fsrs import x | from .lib.fsrs import x
        re.compile(rf"^\s*(?:from\s+[.\w]*\b{s}\b|import\s+(?:[.\w]+\s*,\s*)*{s}\b)",
                   re.MULTILINE),
    ]


def hollow_imported_modules(repo_root: str, changed_files) -> list[str]:
    """Repo-relative paths of modules changed this turn that are empty AND
    imported from somewhere else in the project.

    `changed_files` is whatever the loop tracks for the turn — an iterable of
    paths, absolute or repo-relative. Anything unreadable is skipped silently.
    """
    root = os.path.abspath(repo_root)
    hollow: list[tuple[str, str]] = []   # (abs_path, stem)
    for raw in changed_files or []:
        p = str(raw)
        absolute = p if os.path.isabs(p) else os.path.join(root, p)
        if not os.path.isfile(absolute):
            continue
        rel = os.path.relpath(absolute, root)
        if rel.startswith(".."):
            continue
        if is_hollow_source(absolute):
            hollow.append((absolute, os.path.splitext(os.path.basename(absolute))[0]))
    if not hollow:
        return []

    # Scan the project once for importers, skipping the hollow files themselves
    # (a module importing its own name proves nothing).
    hollow_abs = {a for a, _ in hollow}
    patterns = {stem: _import_patterns(stem) for _, stem in hollow}
    imported: set[str] = set()
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _EXEMPT_DIR_PARTS and not d.startswith(".")]
        for name in files:
            if os.path.splitext(name)[1].lower() not in _CODE_EXTS:
                continue
            fpath = os.path.join(base, name)
            if fpath in hollow_abs:
                continue
            try:
                if os.path.getsize(fpath) > _MAX_BYTES * 8:
                    continue
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except Exception:
                continue
            for stem, pats in patterns.items():
                if stem in imported:
                    continue
                if any(pat.search(text) for pat in pats):
                    imported.add(stem)
        if len(imported) == len(patterns):
            break
    return sorted(
        os.path.relpath(a, root) for a, stem in hollow if stem in imported
    )


class HollowModuleGate:
    """Turn-level state: are there still imported-but-empty modules?

    Mirrors `ProjectCheckGate`. The nudge is bounded so a model that cannot
    implement the module doesn't spin, but running out of nudges does NOT make
    the app whole — `blocks_completion` keeps holding, and the reason rides the
    final result text so it reaches both the TUI and `--json`.
    """

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self.paths: list[str] = []
        self.retries = 0

    def mark(self, paths) -> None:
        self.paths = sorted(set(paths))

    def clear(self) -> None:
        """A later round found nothing hollow — the modules were implemented."""
        self.paths = []

    def consume_retry(self) -> bool:
        if self.retries >= self.max_retries:
            return False
        self.retries += 1
        return True

    def blocks_completion(self) -> bool:
        return bool(self.paths)

    def result_note(self) -> str:
        if not self.paths:
            return ""
        listed = "\n".join(f"  - {p}" for p in self.paths)
        return ("\n\nThese modules are imported but contain no code — the feature "
                f"they implement does not exist:\n{listed}")
