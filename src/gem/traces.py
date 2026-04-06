from __future__ import annotations

import gzip
import json
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .session import SessionStore


class EventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    USER_INPUT = "user_input"
    ERROR = "error"
    PLAN = "plan"


class SessionLogger:
    """Real-time session logger that writes events to JSONL files in ~/.gem/logs/."""

    LOG_DIR = Path.home() / ".gem" / "logs"
    COMPRESS_AFTER_DAYS = 7
    DELETE_AFTER_DAYS = 30

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.started_at = time.monotonic()
        self.start_ts = datetime.now(timezone.utc).isoformat()
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._log_path = self.LOG_DIR / f"session_{timestamp}.jsonl"
        self._file = open(self._log_path, "a")  # noqa: SIM115

        self._counts: dict[str, int] = {e.value: 0 for e in EventType}
        self._total_tokens = 0
        self._total_duration_ms = 0.0
        self._parse_ok = 0
        self._parse_fail = 0

        self._rotate()

    # ── public api ──────────────────────────────────────────────

    def log(
        self,
        event_type: EventType | str,
        *,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        tokens: int | None = None,
        parse_success: bool | None = None,
    ) -> None:
        """Append a single event to the session log file."""
        etype = EventType(event_type) if isinstance(event_type, str) else event_type

        record: dict[str, Any] = {
            "session_id": self.session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": etype.value,
        }
        if data is not None:
            record["data"] = data
        if duration_ms is not None:
            record["duration_ms"] = duration_ms
            self._total_duration_ms += duration_ms
        if tokens is not None:
            record["tokens"] = tokens
            self._total_tokens += tokens
        if parse_success is not None:
            record["parse_success"] = parse_success
            if parse_success:
                self._parse_ok += 1
            else:
                self._parse_fail += 1

        self._counts[etype.value] = self._counts.get(etype.value, 0) + 1
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_llm_call(
        self,
        *,
        model: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_ms: float = 0.0,
        data: dict[str, Any] | None = None,
    ) -> None:
        merged = {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        if data:
            merged.update(data)
        self.log(
            EventType.LLM_CALL,
            data=merged,
            duration_ms=duration_ms,
            tokens=prompt_tokens + completion_tokens,
        )

    def log_tool_call(
        self,
        tool_name: str,
        *,
        duration_ms: float = 0.0,
        parse_success: bool | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        merged = {"tool": tool_name}
        if data:
            merged.update(data)
        self.log(EventType.TOOL_CALL, data=merged, duration_ms=duration_ms, parse_success=parse_success)

    def log_user_input(self, content: str, *, data: dict[str, Any] | None = None) -> None:
        merged = {"content": content}
        if data:
            merged.update(data)
        self.log(EventType.USER_INPUT, data=merged)

    def log_error(self, error: str | Exception, *, data: dict[str, Any] | None = None) -> None:
        merged = {"error": str(error), "type": type(error).__name__ if isinstance(error, Exception) else "str"}
        if data:
            merged.update(data)
        self.log(EventType.ERROR, data=merged)

    def log_plan(self, plan: str, *, data: dict[str, Any] | None = None) -> None:
        merged = {"plan": plan}
        if data:
            merged.update(data)
        self.log(EventType.PLAN, data=merged)

    def get_session_stats(self) -> dict[str, Any]:
        """Return summary statistics for the current session."""
        elapsed = time.monotonic() - self.started_at
        return {
            "session_id": self.session_id,
            "started_at": self.start_ts,
            "elapsed_s": round(elapsed, 2),
            "event_counts": dict(self._counts),
            "total_events": sum(self._counts.values()),
            "total_tokens": self._total_tokens,
            "total_duration_ms": round(self._total_duration_ms, 2),
            "parse_success": self._parse_ok,
            "parse_failure": self._parse_fail,
            "log_file": str(self._log_path),
        }

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    # ── rotation / compression ──────────────────────────────────

    def _rotate(self) -> None:
        """Compress logs older than 7 days, delete logs older than 30 days."""
        now = datetime.now(timezone.utc)
        compress_cutoff = now - timedelta(days=self.COMPRESS_AFTER_DAYS)
        delete_cutoff = now - timedelta(days=self.DELETE_AFTER_DAYS)

        for path in self.LOG_DIR.iterdir():
            if path == self._log_path:
                continue

            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue

            # Delete old logs (both .jsonl and .jsonl.gz)
            if mtime < delete_cutoff:
                path.unlink(missing_ok=True)
                continue

            # Compress uncompressed logs older than 7 days
            if path.suffix == ".jsonl" and mtime < compress_cutoff:
                gz_path = path.with_suffix(".jsonl.gz")
                with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                path.unlink()

    def __enter__(self) -> SessionLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ── existing export function ────────────────────────────────────


def export_training_traces(output_path: Path) -> tuple[int, Path]:
    store = SessionStore()
    rows = []
    for session_id, _created_at, _repo_root in store.list_sessions():
        session = store.load(session_id)
        for idx in range(1, len(session.messages)):
            current = session.messages[idx]
            previous = session.messages[idx - 1]
            if current.get("role") != "assistant":
                continue
            if previous.get("role") != "user":
                continue
            rows.append(
                {
                    "session_id": session.session_id,
                    "profile": session.profile,
                    "model": session.model,
                    "prompt": previous.get("content", ""),
                    "response": current.get("content", ""),
                    "events": session.events[-20:],
                }
            )
    output_path.write_text("\n".join(json.dumps(row) for row in rows))
    return len(rows), output_path
