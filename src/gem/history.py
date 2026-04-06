"""Unified conversation history — SQLite-backed, queryable, auto-summarized.

Replaces three fragmented stores:
- prompt_history.txt (readline only, no outputs)
- sessions/{id}.json (full conversations but flat files)
- logs/session_{ts}.jsonl (structured but separate)

One database, all data:
- Conversations indexed by repo
- Every prompt + response + tool calls + file changes + timing
- Queryable: "what did I do last week", "show edits to auth.py"
- Auto-summary extraction for memory consolidation
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from .config import ensure_home_dirs


DB_NAME = "history.db"
DB_VERSION = 1


# ── Data types ──────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    """A single turn in conversation history."""
    id: str
    session_id: str
    repo_root: str
    timestamp: float
    role: str            # "user" | "assistant" | "tool"
    content: str
    # Metadata
    intent: str = ""     # classified intent (CREATE, EDIT, FIX, etc.)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0
    # Tool call details (if role == "tool")
    tool_name: str = ""
    tool_args: str = ""  # JSON
    tool_result: str = ""
    tool_error: bool = False
    # File changes (if any)
    files_changed: str = ""  # JSON list of paths
    diff_summary: str = ""   # "+5 -3 auth.py, +20 models/user.py"


@dataclass
class SessionSummary:
    """Summary of a conversation session."""
    session_id: str
    repo_root: str
    started_at: str
    last_active: str
    turn_count: int
    model: str
    total_tokens: int
    files_touched: list[str]
    summary: str = ""  # auto-generated summary


# ── Database ────────────────────────────────────────────────────────

class HistoryDB:
    """SQLite-backed conversation history."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(ensure_home_dirs() / DB_NAME)
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                repo_root TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                intent TEXT DEFAULT '',
                model TEXT DEFAULT '',
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0,
                tool_name TEXT DEFAULT '',
                tool_args TEXT DEFAULT '',
                tool_result TEXT DEFAULT '',
                tool_error INTEGER DEFAULT 0,
                files_changed TEXT DEFAULT '[]',
                diff_summary TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id);
            CREATE INDEX IF NOT EXISTS idx_history_repo ON history(repo_root);
            CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_history_role ON history(role);
            CREATE INDEX IF NOT EXISTS idx_history_tool ON history(tool_name);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                repo_root TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                model TEXT DEFAULT '',
                summary TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
        """)
        # Check schema version
        row = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] if row and row[0] else 0
        if current < DB_VERSION:
            self.conn.execute("INSERT OR REPLACE INTO schema_version VALUES (?)", (DB_VERSION,))
            self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Write ───────────────────────────────────────────────────────

    def record_turn(self, entry: HistoryEntry) -> None:
        """Record a single conversation turn."""
        self.conn.execute("""
            INSERT INTO history (
                id, session_id, repo_root, timestamp, role, content,
                intent, model, tokens_in, tokens_out, duration_ms,
                tool_name, tool_args, tool_result, tool_error,
                files_changed, diff_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id or uuid.uuid4().hex[:12],
            entry.session_id,
            entry.repo_root,
            entry.timestamp or time.time(),
            entry.role,
            entry.content,
            entry.intent,
            entry.model,
            entry.tokens_in,
            entry.tokens_out,
            entry.duration_ms,
            entry.tool_name,
            entry.tool_args,
            entry.tool_result,
            1 if entry.tool_error else 0,
            entry.files_changed,
            entry.diff_summary,
        ))
        self.conn.commit()

        # Update session record
        now = datetime.now(UTC).isoformat()
        self.conn.execute("""
            INSERT INTO sessions (session_id, repo_root, started_at, last_active, model)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active = excluded.last_active,
                model = CASE WHEN excluded.model != '' THEN excluded.model ELSE sessions.model END
        """, (entry.session_id, entry.repo_root, now, now, entry.model))
        self.conn.commit()

    def record_user_prompt(self, session_id: str, repo_root: str,
                           prompt: str, intent: str = "", model: str = "") -> str:
        """Record a user prompt. Returns the entry ID."""
        entry_id = uuid.uuid4().hex[:12]
        self.record_turn(HistoryEntry(
            id=entry_id,
            session_id=session_id,
            repo_root=repo_root,
            timestamp=time.time(),
            role="user",
            content=prompt,
            intent=intent,
            model=model,
        ))
        return entry_id

    def record_assistant_response(self, session_id: str, repo_root: str,
                                  response: str, tokens_in: int = 0,
                                  tokens_out: int = 0, duration_ms: float = 0,
                                  model: str = "", files_changed: list[str] | None = None) -> str:
        """Record an assistant response."""
        entry_id = uuid.uuid4().hex[:12]
        self.record_turn(HistoryEntry(
            id=entry_id,
            session_id=session_id,
            repo_root=repo_root,
            timestamp=time.time(),
            role="assistant",
            content=response,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            files_changed=json.dumps(files_changed or []),
        ))
        return entry_id

    def record_tool_call(self, session_id: str, repo_root: str,
                         tool_name: str, args: dict, result: str,
                         error: bool = False, duration_ms: float = 0) -> str:
        """Record a tool call."""
        entry_id = uuid.uuid4().hex[:12]
        self.record_turn(HistoryEntry(
            id=entry_id,
            session_id=session_id,
            repo_root=repo_root,
            timestamp=time.time(),
            role="tool",
            content=result[:500],
            tool_name=tool_name,
            tool_args=json.dumps(args, default=str)[:1000],
            tool_result=result[:2000],
            tool_error=error,
            duration_ms=duration_ms,
        ))
        return entry_id

    # ── Query ───────────────────────────────────────────────────────

    def get_session_history(self, session_id: str, limit: int = 100) -> list[dict]:
        """Get all turns for a session."""
        rows = self.conn.execute(
            "SELECT * FROM history WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_prompts(self, repo_root: str = "", limit: int = 20) -> list[dict]:
        """Get recent user prompts, optionally filtered by repo."""
        if repo_root:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE role = 'user' AND repo_root = ? ORDER BY timestamp DESC LIMIT ?",
                (repo_root, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE role = 'user' ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def search_history(self, query: str, repo_root: str = "",
                       limit: int = 20) -> list[dict]:
        """Full-text search across prompts and responses."""
        pattern = f"%{query}%"
        if repo_root:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE content LIKE ? AND repo_root = ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, repo_root, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_file_history(self, file_path: str, repo_root: str = "",
                         limit: int = 20) -> list[dict]:
        """Get all turns that touched a specific file."""
        pattern = f"%{file_path}%"
        if repo_root:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE files_changed LIKE ? AND repo_root = ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, repo_root, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE files_changed LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_usage(self, tool_name: str = "", limit: int = 50) -> list[dict]:
        """Get tool call history."""
        if tool_name:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE role = 'tool' AND tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE role = 'tool' ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self, repo_root: str = "", limit: int = 20) -> list[SessionSummary]:
        """List sessions, optionally filtered by repo."""
        if repo_root:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE repo_root = ? ORDER BY last_active DESC LIMIT ?",
                (repo_root, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY last_active DESC LIMIT ?",
                (limit,)
            ).fetchall()

        summaries = []
        for row in rows:
            # Count turns and tokens
            stats = self.conn.execute("""
                SELECT COUNT(*) as turns,
                       SUM(tokens_in + tokens_out) as total_tokens
                FROM history WHERE session_id = ?
            """, (row["session_id"],)).fetchone()

            # Get files touched
            files_rows = self.conn.execute("""
                SELECT DISTINCT files_changed FROM history
                WHERE session_id = ? AND files_changed != '[]'
            """, (row["session_id"],)).fetchall()

            all_files = set()
            for fr in files_rows:
                try:
                    all_files.update(json.loads(fr["files_changed"]))
                except (json.JSONDecodeError, TypeError):
                    pass

            summaries.append(SessionSummary(
                session_id=row["session_id"],
                repo_root=row["repo_root"],
                started_at=row["started_at"],
                last_active=row["last_active"],
                turn_count=stats["turns"] if stats else 0,
                model=row["model"],
                total_tokens=stats["total_tokens"] or 0 if stats else 0,
                files_touched=sorted(all_files),
                summary=row["summary"],
            ))

        return summaries

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self, repo_root: str = "", days: int = 30) -> dict:
        """Get usage statistics."""
        cutoff = time.time() - (days * 86400)

        where = "WHERE timestamp > ?"
        params: list[Any] = [cutoff]
        if repo_root:
            where += " AND repo_root = ?"
            params.append(repo_root)

        row = self.conn.execute(f"""
            SELECT
                COUNT(CASE WHEN role = 'user' THEN 1 END) as prompts,
                COUNT(CASE WHEN role = 'assistant' THEN 1 END) as responses,
                COUNT(CASE WHEN role = 'tool' THEN 1 END) as tool_calls,
                SUM(tokens_in) as total_tokens_in,
                SUM(tokens_out) as total_tokens_out,
                SUM(duration_ms) as total_duration_ms,
                COUNT(DISTINCT session_id) as sessions
            FROM history {where}
        """, params).fetchone()

        # Most used tools
        tools = self.conn.execute(f"""
            SELECT tool_name, COUNT(*) as count
            FROM history {where} AND role = 'tool' AND tool_name != ''
            GROUP BY tool_name ORDER BY count DESC LIMIT 10
        """, params).fetchall()

        # Most edited files
        file_rows = self.conn.execute(f"""
            SELECT files_changed FROM history {where} AND files_changed != '[]'
        """, params).fetchall()

        file_counts: dict[str, int] = {}
        for fr in file_rows:
            try:
                for f in json.loads(fr["files_changed"]):
                    file_counts[f] = file_counts.get(f, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:10]

        return {
            "period_days": days,
            "prompts": row["prompts"] or 0,
            "responses": row["responses"] or 0,
            "tool_calls": row["tool_calls"] or 0,
            "total_tokens_in": row["total_tokens_in"] or 0,
            "total_tokens_out": row["total_tokens_out"] or 0,
            "total_duration_s": round((row["total_duration_ms"] or 0) / 1000, 1),
            "sessions": row["sessions"] or 0,
            "top_tools": [(t["tool_name"], t["count"]) for t in tools],
            "top_files": top_files,
        }

    # ── Memory extraction ───────────────────────────────────────────

    def extract_recent_context(self, repo_root: str, max_turns: int = 50) -> str:
        """Extract a summary of recent work for memory/context injection.

        Returns a text block summarizing what was done recently in this repo.
        No LLM needed — rule-based extraction.
        """
        rows = self.conn.execute("""
            SELECT role, content, intent, tool_name, files_changed, timestamp
            FROM history WHERE repo_root = ? AND role IN ('user', 'assistant')
            ORDER BY timestamp DESC LIMIT ?
        """, (repo_root, max_turns)).fetchall()

        if not rows:
            return ""

        lines = []
        seen_prompts = set()

        for row in reversed(list(rows)):
            if row["role"] == "user":
                prompt = row["content"][:100]
                if prompt not in seen_prompts:
                    seen_prompts.add(prompt)
                    intent = f" [{row['intent']}]" if row["intent"] else ""
                    lines.append(f"- User{intent}: {prompt}")
            elif row["role"] == "assistant":
                files = row["files_changed"]
                if files and files != "[]":
                    try:
                        flist = json.loads(files)
                        if flist:
                            lines.append(f"  → Modified: {', '.join(flist[:5])}")
                    except (json.JSONDecodeError, TypeError):
                        pass

        if not lines:
            return ""

        return "Recent work in this project:\n" + "\n".join(lines[-20:])

    # ── Cleanup ─────────────────────────────────────────────────────

    def cleanup(self, max_age_days: int = 90) -> int:
        """Delete history older than max_age_days. Returns rows deleted."""
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self.conn.execute(
            "DELETE FROM history WHERE timestamp < ?", (cutoff,)
        )
        self.conn.commit()
        return cursor.rowcount
