"""Single event sink for LocalCode — Phase 1 (local-only).

Why one sink
------------
Before this module, telemetry was scattered across:
  • lifecycle.log         (server lifecycle, tool calls, redactions)
  • telemetry/turns.jsonl (per-turn detailed traces)
  • last-pressure-kill.txt (single-value marker file)
  • SessionStore.append_event (SQLite session events)

Same data emitted to multiple places, no consistent schema, no single
file to `tail` or `jq` against. This module replaces all of that with
ONE function, ONE file, ONE schema:

    emit(event_type, **fields) → appends one JSON line to
                                 `.localcode/events.jsonl`

Schema
------
Every line is a single JSON object with at minimum:

    {
      "t":         ISO-8601 UTC timestamp (millisecond precision)
      "type":      string event name (e.g. "tool_call", "server_started")
      "session":   process-lifetime UUID (groups events from same run)
      "pid":       writer PID (helps when multiple processes append)
      ...event-specific fields...
    }

The `t`/`type`/`session`/`pid` envelope is added automatically by
`emit()`. Callers only pass the event-specific payload as kwargs.

Why JSONL
---------
  • One event per line — atomic append on POSIX, no locking needed
    for single-writer cases (each process is its own writer).
  • Each line is independently parseable — `jq` / `awk` / Python
    `json.loads` all work without a custom parser.
  • Easy to tail: `tail -f .localcode/events.jsonl | jq .`
  • Append-only file plays well with `git ignore` and never needs
    truncation — if it grows past ~50 MB on disk, rotate manually.

This log is LOCAL-ONLY. Nothing here is transmitted off the machine: no
uploader, no endpoint, no `LOCALCODE_TELEMETRY_URL`. Any future remote/cloud
sink would be a new, opt-in feature that must first strip user content (prompts,
outputs, file paths) — not a flip of a flag documented here.

Performance
-----------
  • Single open(append) + write per event — roughly 50-100µs per
    emit on warm disk. Fine at our event rate (~10-50 events/sec
    peak during heavy tool usage).
  • Defensive on every I/O — telemetry must never break the agent
    loop. All exceptions silently swallowed.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .paths import events_log_path


# Process-lifetime session identifier. Groups all events from a single
# `localcode` run, regardless of how many turns / model switches /
# server restarts happen within that run. Useful for "show me everything
# that happened in the user's session that crashed at 3 PM."
SESSION_ID = uuid.uuid4().hex[:12]
PID = os.getpid()


def _remove_legacy_user_id() -> None:
    """Delete the persistent per-install id earlier versions minted.

    `~/.localcode/user_id` was a stable 16-hex id stamped on every event.
    The log is local-only, so the only thing a persistent install id could
    ever serve is cross-machine correlation by a remote sink that does not
    exist — SESSION_ID already groups all events from one run. Remove the
    code that minted it AND the file it left behind. Best-effort: failure
    to unlink must never break startup.
    """
    try:
        from .paths import global_state_dir
        p = global_state_dir() / "user_id"
        if p.is_file():
            p.unlink()
    except Exception:
        pass


_remove_legacy_user_id()


# ── MECE event taxonomy ──────────────────────────────────────────────
# Every event_type lives in EXACTLY ONE bucket. `emit()` auto-injects
# the bucket based on this mapping so downstream analysis can filter
# with a single `jq` expression (`select(.bucket == "tool")`) without
# needing to remember every event name.
#
# Buckets:
#   session  — once per process (machine, install, config snapshot)
#   turn     — once per user turn (input/output boundaries)
#   round    — once per agent-loop round (inside a turn)
#   tool     — per tool call (dispatch + result)
#   server   — llama-server lifecycle, health, memory
#   model    — model behavior outcomes (abort, loop, stall)
#   recovery — auto-nudge, compaction, server-reconnect
#   error    — unhandled errors that bubble to the user
EVENT_BUCKETS: dict[str, str] = {
    # session
    "session_info":           "session",
    # turn
    "turn_start":             "turn",
    "turn_end":               "turn",
    "ui_turn_end":            "turn",
    "skill_injection":        "turn",
    "skill_selection":        "turn",
    "launcher_result":        "turn",
    "launcher_error":         "turn",
    "edit_quality":           "turn",
    "tool_facts":             "tool",
    # round
    "round_start":            "round",
    "round_end":              "round",
    "redaction":              "round",
    # tool
    "tool_call":              "tool",
    "tool_result":            "tool",
    # server
    "server_started":         "server",
    "server_stopped":         "server",
    "server_health_result":   "server",
    "memory_wait_start":      "server",
    "memory_released":        "server",
    "memory_release_timeout": "server",
    "pressure_kill":          "server",
    # model behavior
    "thinking_abort":         "model",
    "model_loop_detected":    "model",
    "generation_started":     "model",
    "generation_aborted":     "model",
    # recovery
    "auto_nudge":             "recovery",
    "compaction":             "recovery",
    "server_reconnect":       "recovery",
    "recovery_scheduled":     "recovery",
    "recovery_completed":     "recovery",
    "recovery_exhausted":     "recovery",
    # error
    "error":                  "error",
}


def _bucket_for(event_type: str) -> str:
    """Return the MECE bucket for an event_type, or 'other' if unmapped.

    Keep the mapping above in sync when adding new event types. Unknown
    types landing in 'other' is a soft signal to the dev reviewing the
    log that a new event type was introduced without being categorised."""
    return EVENT_BUCKETS.get(event_type, "other")


def _iso_now() -> str:
    """Millisecond-precision ISO-8601 UTC timestamp.

    Resolution matters because tool calls fire on millisecond cadence
    in tight loops; full-second precision would collapse multiple
    distinct events to the same `t` and confuse downstream analysis."""
    now = datetime.now(timezone.utc)
    # `isoformat(timespec='milliseconds')` gives YYYY-MM-DDTHH:MM:SS.fff+00:00
    return now.isoformat(timespec="milliseconds")


# Event capture is OPT-IN. The LOCALCODE_EVENTS env var wins when set
# (test suites set it explicitly in either direction); otherwise
# `[telemetry] enabled` in config.toml decides, and its default is FALSE.
def _enabled() -> bool:
    # The per-project events.jsonl is a LOCAL debug log: append-only, never
    # uploaded, redacted of secrets at emit, and (since the user-id removal) it
    # carries no cross-session identifier. So it defaults ON - it is the log the
    # user tails to see what a turn did. `LOCALCODE_EVENTS=0` turns it off. This
    # is deliberately separate from the UI turn-trace telemetry: the privacy fix
    # was deleting the persistent install id, not silencing local debugging.
    env = os.environ.get("LOCALCODE_EVENTS")
    if env is not None:
        return env not in ("0", "false", "no", "off")
    return True


# Cache the resolved path per-process. find_project_root() walks the
# filesystem; doing it on every emit() would add unnecessary I/O at
# the rate we emit events. The cache invalidates when cwd changes
# (rare in a single process — typically only at startup).
_cached_path: Optional[Path] = None
_cached_cwd: Optional[str] = None
_path_lock = threading.Lock()


def _resolve_path() -> Path:
    """Return the events.jsonl path for the current cwd, cached."""
    global _cached_path, _cached_cwd
    cwd = os.getcwd()
    with _path_lock:
        if _cached_path is None or _cached_cwd != cwd:
            _cached_path = events_log_path()
            _cached_cwd = cwd
        return _cached_path


def emit(event_type: str, **fields: Any) -> None:
    """Append one structured event to `.localcode/events.jsonl`.

    Always returns None. Never raises — telemetry failures are
    silently swallowed because the agent loop must not break on a
    full disk or a permission glitch.

    Args:
        event_type: short snake_case name (e.g. "tool_call",
                    "server_started", "redaction"). Convention is
                    noun_verbed when describing past events.
        **fields:   event-specific payload. Keep keys short and
                    snake_case. Values must be JSON-serializable.
                    Strings longer than ~500 chars should be
                    summarised at the call site (this module is not
                    in the business of truncating user/model text).
    """
    if not _enabled():
        return
    record = {
        "t": _iso_now(),
        "bucket": _bucket_for(event_type),
        "type": event_type,
        "session": SESSION_ID,
        "pid": PID,
    }
    record.update(fields)
    # Secret redaction chokepoint. events.jsonl is a durable, cleartext
    # record of tool args and result summaries — a `cat .env` or a key
    # pasted into a prompt would otherwise be written to disk verbatim
    # and stay there. Scrub VALUES (not just secret-looking key names)
    # before serialization. Cheap: scrub() short-circuits on strings
    # with no credential-prefix substring, which is nearly all of them.
    try:
        from .redaction import scrub_obj
        record = scrub_obj(record)
    except Exception:
        pass
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception:
        # Last-ditch: serialize the record with everything stringified.
        # Better to write something than to silently drop it.
        try:
            try:
                from .redaction import scrub as _scrub
            except Exception:
                def _scrub(x):  # type: ignore[misc]
                    return x
            safe = {k: _scrub(str(v)) for k, v in record.items()}
            line = json.dumps(safe, ensure_ascii=False)
        except Exception:
            return
    try:
        path = _resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _collect_machine_specs() -> dict:
    """Snapshot of machine + install context at session start.

    Captured once per process so analysis can correlate behaviour with
    hardware (e.g. "memory pressure happens 3× more often on 16 GB
    Macs vs 32 GB"). Best-effort — any field that can't be read
    defaults to empty string, never raises."""
    import platform as _platform
    specs: dict[str, Any] = {
        "os": _platform.system(),
        "os_release": _platform.release(),
        "os_version": _platform.version(),
        "arch": _platform.machine(),
        "python": _platform.python_version(),
    }
    # macOS: physical RAM via sysctl, GPU via system_profiler (cheap).
    try:
        import subprocess as _sp
        if _platform.system() == "Darwin":
            r = _sp.run(["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True, timeout=1)
            if r.returncode == 0:
                specs["ram_gb"] = int(r.stdout.strip()) // (1024 ** 3)
            r = _sp.run(["sysctl", "-n", "hw.model"],
                        capture_output=True, text=True, timeout=1)
            if r.returncode == 0:
                specs["mac_model"] = r.stdout.strip()
            r = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True, timeout=1)
            if r.returncode == 0:
                specs["cpu"] = r.stdout.strip()
            r = _sp.run(["sysctl", "-n", "hw.ncpu"],
                        capture_output=True, text=True, timeout=1)
            if r.returncode == 0:
                specs["cpu_cores"] = int(r.stdout.strip())
    except Exception:
        pass
    # App version + git commit.
    try:
        from importlib.metadata import version as _pkgver
        specs["app_version"] = _pkgver("localcode")
    except Exception:
        specs["app_version"] = ""
    try:
        import subprocess as _sp
        r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=1,
                    cwd=os.getcwd())
        specs["git_commit"] = r.stdout.strip() if r.returncode == 0 else ""
        r = _sp.run(["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=1,
                    cwd=os.getcwd())
        specs["git_dirty"] = bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        specs["git_commit"] = ""
        specs["git_dirty"] = False
    return specs


_session_info_emitted = False
_session_info_lock = threading.Lock()


def emit_session_info_once(**extra: Any) -> None:
    """Emit the one-per-process `session_info` event with machine specs.

    Safe to call multiple times — the second+ call is a no-op. The
    caller (usually `LocalCodeApp.__init__` or cli.main) passes
    runtime-specific extras via **extra: the active model name,
    current mode, autonomy level, context window size, etc."""
    global _session_info_emitted
    with _session_info_lock:
        if _session_info_emitted:
            return
        _session_info_emitted = True
    try:
        specs = _collect_machine_specs()
        specs.update(extra)
        emit("session_info", **specs)
    except Exception:
        pass


def read_events(since_seconds: Optional[float] = None) -> list[dict]:
    """Read events from the log, optionally only those newer than
    `since_seconds` ago. Used by the error formatter to check for
    recent pressure kills, by the e2e harness for assertions, and by
    future analysis tools.

    Returns a list (not a generator) because callers usually want to
    do multiple passes (filter by type, then group, etc.).
    """
    out: list[dict] = []
    try:
        path = _resolve_path()
        if not path.is_file():
            return out
        cutoff = time.time() - since_seconds if since_seconds is not None else None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff is not None:
                    # Parse the ISO timestamp lazily — only for events
                    # we might care about.
                    ts_str = rec.get("t", "")
                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                    except Exception:
                        continue
                    if ts < cutoff:
                        continue
                out.append(rec)
    except Exception:
        pass
    return out


def find_recent(event_type: str, within_seconds: float = 60.0) -> Optional[dict]:
    """Return the most recent event of `event_type` within the window,
    or None. Helper for "did pressure_kill fire in the last minute?"
    """
    candidates = [r for r in read_events(since_seconds=within_seconds)
                  if r.get("type") == event_type]
    if not candidates:
        return None
    return candidates[-1]
