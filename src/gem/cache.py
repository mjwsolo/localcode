"""Tool result caching + speculative pre-execution + background indexing.

Three speed optimizations in one module:

1. Result Cache: memoize read_file, grep, glob results. Invalidate on file mtime change.
2. Speculative Pre-exec: predict likely tools from intent, pre-fetch in background.
3. Background Indexing: continuously index repo files so search is instant.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── Result Cache ─────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    result: str
    timestamp: float
    file_mtime: float = 0.0  # for file-based cache invalidation


class ToolResultCache:
    """Cache tool results within a session. Invalidate on file change."""

    CACHEABLE_TOOLS = {"read_file", "grep", "glob", "list_files", "git_status", "current_datetime"}
    TTL_SECONDS = {
        "read_file": 30,      # files change during coding
        "grep": 30,
        "glob": 60,
        "list_files": 60,
        "git_status": 10,     # changes frequently
        "current_datetime": 2,  # basically no cache
    }

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def _key(self, tool_name: str, args: dict) -> str:
        args_str = str(sorted(args.items()))
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()

    def get(self, tool_name: str, args: dict) -> str | None:
        if tool_name not in self.CACHEABLE_TOOLS:
            return None
        key = self._key(tool_name, args)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ttl = self.TTL_SECONDS.get(tool_name, 30)
            if time.time() - entry.timestamp > ttl:
                del self._cache[key]
                return None
            # Check file mtime for file-based tools
            if tool_name == "read_file" and "path" in args:
                try:
                    current_mtime = (self.repo_root / args["path"]).stat().st_mtime
                    if current_mtime != entry.file_mtime:
                        del self._cache[key]
                        return None
                except Exception:
                    pass
            return entry.result

    def put(self, tool_name: str, args: dict, result: str) -> None:
        if tool_name not in self.CACHEABLE_TOOLS:
            return
        key = self._key(tool_name, args)
        file_mtime = 0.0
        if tool_name == "read_file" and "path" in args:
            try:
                file_mtime = (self.repo_root / args["path"]).stat().st_mtime
            except Exception:
                pass
        with self._lock:
            self._cache[key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                file_mtime=file_mtime,
            )

    def invalidate_all(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Speculative Pre-execution ────────────────────────────────────────────

class SpeculativeExecutor:
    """Pre-fetch likely tool results while model is thinking.

    Usage:
        spec = SpeculativeExecutor(toolkit)
        spec.predict_and_prefetch("what time is it?", routing)
        # ... model thinks ...
        result = spec.get_if_ready("current_datetime", {})  # instant!
    """

    def __init__(self, execute_fn: Callable) -> None:
        self._execute = execute_fn
        self._results: dict[str, str] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def predict_and_prefetch(self, user_text: str, tool_names: set[str]) -> None:
        """Start pre-fetching predicted tool results in background."""
        predictions = self._predict(user_text, tool_names)
        for tool_name, args in predictions:
            t = threading.Thread(
                target=self._fetch,
                args=(tool_name, args),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def get_if_ready(self, tool_name: str, args: dict) -> str | None:
        """Get pre-fetched result if available."""
        key = f"{tool_name}:{sorted(args.items())}"
        with self._lock:
            return self._results.get(key)

    def _fetch(self, tool_name: str, args: dict) -> None:
        try:
            call = {"function": {"name": tool_name, "arguments": args}}
            results = self._execute([call])
            if results:
                key = f"{tool_name}:{sorted(args.items())}"
                with self._lock:
                    self._results[key] = results[0].get("content", "")
        except Exception:
            pass

    @staticmethod
    def _predict(user_text: str, tool_names: set[str]) -> list[tuple[str, dict]]:
        """Predict which tools will be called and with what args.

        Expanded from 3 predictions to 10+ based on routing intents.
        Each prediction that hits saves 1-5 seconds of tool execution time.
        """
        import re
        predictions: list[tuple[str, dict]] = []
        text_lower = user_text.lower()

        # Time queries
        if "current_datetime" in tool_names and any(w in text_lower for w in ("time", "date", "today", "now")):
            predictions.append(("current_datetime", {}))

        # Git operations — pre-fetch status AND diff (cheap, often needed together)
        if "git_status" in tool_names and any(w in text_lower for w in ("status", "changes", "diff", "modified", "commit", "git")):
            predictions.append(("git_status", {}))
        if "git_diff" in tool_names and any(w in text_lower for w in ("diff", "changes", "modified", "what changed")):
            predictions.append(("git_diff", {}))
        if "git_log" in tool_names and any(w in text_lower for w in ("log", "history", "recent", "last commit")):
            predictions.append(("git_log", {"count": 5}))

        # File reads — extract filenames from user text and pre-read them
        if "read_file" in tool_names:
            # Match common file patterns: foo.py, src/bar.ts, ./config.json
            file_matches = re.findall(r'(?:[\w./]+/)?(\w[\w.-]*\.(?:py|js|ts|json|md|txt|html|css|yaml|yml|toml|cfg|sh|go|rs|c|cpp|h))', user_text)
            for fname in file_matches[:3]:  # cap at 3 files to avoid over-fetching
                predictions.append(("read_file", {"path": fname}))

        # Web search
        if "web_search" in tool_names and any(w in text_lower for w in ("search", "look up", "latest", "news", "documentation")):
            predictions.append(("web_search", {"query": user_text}))

        # Code search — extract likely search terms
        if "grep" in tool_names and any(w in text_lower for w in ("find", "search", "where", "locate", "grep")):
            # Extract quoted strings or key terms
            quoted = re.findall(r'"([^"]+)"', user_text) or re.findall(r"'([^']+)'", user_text)
            if quoted:
                predictions.append(("grep", {"pattern": quoted[0]}))

        # List files for navigation queries
        if "glob" in tool_names and any(w in text_lower for w in ("files", "list", "what files", "directory", "structure")):
            predictions.append(("glob", {"pattern": "**/*.py"}))

        return predictions


# ── Background Indexer ───────────────────────────────────────────────────

class BackgroundIndexer:
    """Continuously indexes repo files so search is instant."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._running = False
        self._thread: threading.Thread | None = None
        self._file_mtimes: dict[str, float] = {}
        self._dirty = True

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while self._running:
            try:
                if self._check_for_changes():
                    self._rebuild_index()
            except Exception:
                pass
            # Check every 10 seconds
            for _ in range(100):
                if not self._running:
                    return
                time.sleep(0.1)

    def _check_for_changes(self) -> bool:
        """Check if any tracked files have changed."""
        from .context import IGNORE_DIRS
        changed = False
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORE_DIRS for part in path.relative_to(self.repo_root).parts):
                continue
            rel = str(path.relative_to(self.repo_root))
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            old_mtime = self._file_mtimes.get(rel, 0)
            if mtime != old_mtime:
                self._file_mtimes[rel] = mtime
                changed = True
        return changed

    def _rebuild_index(self) -> None:
        """Rebuild the code index."""
        from .indexer import build_index
        try:
            build_index(self.repo_root)
        except Exception:
            pass
