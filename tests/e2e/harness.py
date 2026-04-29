"""Headless test harness — drives the real LocalCode agent loop without the TUI.

What this lets us do
--------------------
The TUI's `LocalCodeTUI` registers two callbacks on the OutputManager:
  * `set_event_callback(on_event)`   — receives every agent event
  * `set_approval_callback(...)`     — answers risky-tool prompts

The harness substitutes:
  * a recorder callback that captures every event into a list
  * an auto-approve callback that always says "yes" (so the test never blocks)

Then it calls `agent.run_agent_loop(...)` directly. From the agent loop's
perspective nothing's different — it's still streaming, still calling
tools, still hitting the real llama-server. We just see every event
and can write declarative assertions against the captured trace.

Why this works
--------------
The agent loop has no TUI dependencies — it only talks to OutputManager.
The TUI is a consumer of events; replacing the consumer with a recorder
gives us full programmatic control without touching agent.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Allow `from localcode...` imports when running this file directly.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── Event recorder ─────────────────────────────────────────────────


@dataclass
class RecordedEvent:
    """One agent event with monotonic timestamp.

    `event_type` matches the strings the OutputManager emits (`thinking`,
    `content`, `tool_start`, `tool_result`, `response_done`, `error`,
    `stage`, etc.). `payload` is whatever the producer attached.
    """
    t: float                 # seconds since recorder started
    event_type: str
    payload: dict


@dataclass
class TurnTrace:
    """Everything captured during one user-turn through the agent loop.

    `turn_text` is the user's prompt that started the turn. `events` is
    the full ordered event stream. `final_response` is what the agent
    returned (the value of `run_agent_loop`'s return). `error` holds any
    exception text if the agent loop raised. `wall_clock_s` is total
    runtime including tool execution.
    """
    turn_text: str
    events: list[RecordedEvent] = field(default_factory=list)
    final_response: str = ""
    error: str | None = None
    wall_clock_s: float = 0.0

    # Convenience accessors — used by scenario assertions.
    def events_of(self, event_type: str) -> list[RecordedEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def tool_calls_made(self) -> list[str]:
        """List of tool names called (in order). Read from `tool_start` events."""
        return [e.payload.get("name", "") for e in self.events_of("tool_start")]

    def tool_call_args(self) -> list[tuple[str, dict]]:
        """List of (tool_name, args_dict) for every tool call this turn."""
        out: list[tuple[str, dict]] = []
        for e in self.events_of("tool_start"):
            name = e.payload.get("name", "")
            raw_args = e.payload.get("args", "")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            out.append((name, args))
        return out

    def info_messages(self) -> list[str]:
        """All `out.print_info(...)` strings — used to detect auto-nudges,
        recovery messages, abort warnings, etc."""
        return [str(e.payload.get("message", "")) for e in self.events_of("info")]

    def errors(self) -> list[str]:
        return [str(e.payload.get("message", "")) for e in self.events_of("error")]

    def thinking_text(self) -> str:
        """Concatenated thinking content for this turn."""
        return "".join(str(e.payload.get("chunk", e.payload.get("text", "")))
                       for e in self.events
                       if e.event_type in ("thinking_chunk", "thinking", "thinking_done"))

    def content_text(self) -> str:
        """Concatenated user-facing content for this turn."""
        return "".join(str(e.payload.get("chunk", "")) for e in self.events_of("content"))


class EventRecorder:
    """Captures OutputManager events into a flat list, partitioned by turn.

    `start_turn(text)` opens a new TurnTrace; subsequent events accrete to
    it. `end_turn(response)` closes the trace and appends to `turns`.
    Auto-approve callback returns True for every prompt — tests never
    block on a permission dialog.
    """

    def __init__(self) -> None:
        self.turns: list[TurnTrace] = []
        self._current: TurnTrace | None = None
        self._t0 = time.monotonic()

    # OutputManager hooks
    def on_event(self, event_type: str, payload: dict) -> None:
        if self._current is None:
            return
        self._current.events.append(RecordedEvent(
            t=time.monotonic() - self._t0,
            event_type=event_type,
            payload=dict(payload),  # defensive copy — payload may be mutated
        ))

    def on_approval(self, tool_name: str, command: str) -> str:
        # "once" matches the verdict the TUI's approval popup returns when
        # the user picks "allow once". Never block, never deny — the
        # harness is testing agent behaviour, not security gating.
        return "once"

    def start_turn(self, text: str) -> None:
        self._current = TurnTrace(turn_text=text)
        self._t0_turn = time.monotonic()

    def end_turn(self, response: str = "", error: str | None = None) -> TurnTrace:
        assert self._current is not None
        self._current.final_response = response
        self._current.error = error
        self._current.wall_clock_s = time.monotonic() - self._t0_turn
        out = self._current
        self.turns.append(out)
        self._current = None
        return out


# ── Headless app + agent driver ─────────────────────────────────────


def build_headless_app():
    """Construct a real `LocalCodeApp` without the TUI.

    Uses the user's normal config and current working directory. Server
    state is managed via `ServerManager` exactly as in production —
    the only difference is that no TUI/SetupScreen runs, so we have to
    ensure the server is up ourselves (see `ensure_server_ready`).
    """
    from localcode.config import load_config
    from localcode.app import LocalCodeApp
    config = load_config()
    return LocalCodeApp(config)


def ensure_server_ready(app, *, timeout_s: float = 180.0) -> bool:
    """Make sure llama-server is up and serving. Restart if not.

    Tries the existing health endpoint first (cheap). If it fails,
    drives a `_restart_server()` on the runtime gateway, which goes
    through the same path the `/model` switch uses. Blocks until the
    server reports healthy or the timeout fires.
    """
    from localcode.server_manager import _probe_health
    if _probe_health(8081, timeout=2.0):
        return True
    # Walk the fallback range too — port may have moved.
    for p in range(8081, 8100):
        if _probe_health(p, timeout=0.5):
            return True
    print(f"  server not responding on 8081-8099 — calling _restart_server() …")
    return app.engine._restart_server()


def run_one_turn(
    app,
    recorder: EventRecorder,
    user_text: str,
    *,
    system_prompt: str | None = None,
) -> TurnTrace:
    """Drive the agent loop for ONE user turn and return its TurnTrace.

    Mirrors what `chat.py:_start_turn` does in the TUI: hooks the
    recorder onto the OutputManager, builds the message list with the
    new user message appended, calls `run_agent_loop`, captures the
    return value (or any exception), and closes the trace.

    `system_prompt`: optional override threaded to `run_agent_loop`.
    Eval uses this to swap prompts between variants without touching
    module globals. None → loop uses its default SYSTEM_PROMPT.
    """
    from localcode.agent import run_agent_loop

    # Wire recorder ↔ OutputManager. Idempotent: re-binding is fine.
    app.out.set_event_callback(recorder.on_event)
    app.out.set_approval_callback(recorder.on_approval)

    # The headless harness keeps its own minimal message list. We don't
    # use the full session/history machinery — each scenario decides
    # what context to feed in, so behavior is reproducible.
    if not hasattr(app, "_e2e_messages"):
        app._e2e_messages = []
    messages = app._e2e_messages
    messages.append({"role": "user", "content": user_text})

    recorder.start_turn(user_text)
    try:
        response = run_agent_loop(
            app, user_text, messages, app.out,
            system_prompt=system_prompt,
        )
        return recorder.end_turn(response=response or "", error=None)
    except Exception as exc:
        import traceback
        return recorder.end_turn(
            response="",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


# ── Scenario base + assertion helpers ───────────────────────────────


@dataclass
class ScenarioResult:
    """Outcome of one scenario. `passed` is the headline; `assertions`
    is the per-check breakdown so the report can show what specifically
    failed. `evidence` carries any structured data (token counts, event
    counts) that helps diagnose without re-running."""
    name: str
    passed: bool
    assertions: list[tuple[str, bool, str]]   # (name, passed, detail)
    evidence: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnTrace] = field(default_factory=list)
    wall_clock_s: float = 0.0


def asserts(*checks: tuple[str, bool, str]) -> list[tuple[str, bool, str]]:
    """Tiny helper to collect assertion tuples without per-check
    boilerplate. Pass any number of `(name, ok, detail)` triples.
    """
    return list(checks)


def lifecycle_log_tail(n: int = 50) -> list[str]:
    """Return the last N events from the centralised event log as
    one string-per-line. Used by scenarios that want to assert on
    server/tool/redaction events emitted DURING the run.

    Each line is a single JSON object — callers using `'foo' in line`
    string matching still work, but JSON-aware callers can `json.loads`
    each line for structured assertions.
    """
    from localcode.paths import events_log_path
    p = events_log_path()
    if not p.is_file():
        return []
    try:
        lines = p.read_text().splitlines()
        return lines[-n:]
    except Exception:
        return []


def lifecycle_events_during(start_t: float, end_t: float) -> list[str]:
    """Filter the central event log to events whose timestamps fall in
    [start_t, end_t]. `start_t` and `end_t` are unix seconds.
    """
    from localcode.paths import events_log_path
    p = events_log_path()
    if not p.is_file():
        return []
    out = []
    try:
        import json as _json
        from datetime import datetime
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # New format: each line is JSON; old format was
            # `<iso_ts> <event> <kv pairs>`. Support both during the
            # transition so a half-migrated log doesn't break tests.
            try:
                rec = _json.loads(line)
                ts_str = rec.get("t", "")
            except _json.JSONDecodeError:
                ts_str = line.split(" ", 1)[0]
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
            except Exception:
                continue
            if start_t <= ts <= end_t:
                out.append(line)
    except Exception:
        pass
    return out
