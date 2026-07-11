"""Turn-level telemetry: append-only JSONL trace for DEV/MANAGER review.

This log is for the team building LocalCode (me + the user who owns the
project), not for end-users. Purpose is offline quality analysis — read
the file, cluster bad turns, fix prompts/tools/skills. Nothing in here
is exposed through the TUI; there's no user-facing privacy surface to
defend. Capture everything useful.

One JSON object per completed user turn is written to:

    ~/.localcode/telemetry/turns.jsonl

The goal is to capture enough signal that, after a day of dogfooding,
we can go back and answer questions like:

  * Which tools fail most often, and with what args?
  * What tool SEQUENCES precede a user correction or repeated prompt?
  * How much real wall-clock time is the model spending in each tool?
  * When the model skipped a skill that "should have" matched, what was
    the user's follow-up?
  * Which skills actually get invoked vs. which sit unused?

No model is in the loop. This is raw event data — the feedback loop is
human-driven: read the JSONL, cluster bad turns, fix prompts/tools/skills,
rinse. (Schema loosely mirrors agent's SerializedMessage — we dropped
things we don't have like cache tokens / cost, added things we care about
like `tool.duration_ms` and `skill_used`.)

### Schema (per line in turns.jsonl)

    {
      "turn_id": "uuid4",
      "session_id": "uuid4",             # stable for the chat session
      "timestamp": "2026-04-23T01:45:12Z",
      "duration_ms": 4812,
      "cwd": "/Users/.../gemma",
      "model": "qwen36",
      "mode": "fast",
      "user_message": "...",
      "assistant_message": "...",        # final text shown to user
      "thinking_text": "",               # extended-thinking block, if any
      "tools": [                         # ordered by invocation
        {"name": "bash", "args": "ls", "duration_ms": 21,
         "result_chars": 412, "error": false},
         ...
      ],
      "skills_used": ["plan-task"],      # subset of tools where name=="skill"
      "tokens_out": 244,                 # approx (len/4 heuristic)
      "interrupted": false,              # user hit Ctrl+C
      "error": ""                        # turn-level error, empty on success
    }

### Rotation

None at file level (agent doesn't rotate either; the file is append-only
JSONL and analysis tools handle size). If it grows past ~50 MB, split
manually or add a rotator later.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TELEMETRY_DIR = Path.home() / ".localcode" / "telemetry"
_TURNS_LOG = _TELEMETRY_DIR / "turns.jsonl"


def telemetry_enabled() -> bool:
    """Env-var kill switch for dev convenience (e.g. running the test
    suite without polluting the log). Default is ON — this is an
    internal dev/manager tool; we want the data."""
    return os.environ.get("LOCALCODE_TELEMETRY", "1") not in ("0", "false", "no", "off")


@dataclass
class ToolEvent:
    name: str
    args: str                       # one-line, truncated at 500 chars on write
    start_ts: float                 # monotonic, for duration math
    duration_ms: int = 0
    result_chars: int = 0
    error: bool = False


@dataclass
class TurnTrace:
    """Per-turn accumulator. Construct at turn start, mutate while the
    agent loop runs, call `finalize()` at turn end to produce a dict ready
    for JSONL serialization."""
    session_id: str
    user_message: str
    cwd: str
    model: str = ""
    mode: str = ""
    git_branch: str = ""          # captured at turn start via `git rev-parse`
    app_version: str = ""         # importlib.metadata.version("localcode")
    context_window: int = 0       # configured max_ctx for this session
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    started_mono: float = field(default_factory=time.monotonic)
    tools: list[ToolEvent] = field(default_factory=list)
    _active_tool: ToolEvent | None = None
    assistant_message: str = ""
    thinking_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    interrupted: bool = False
    error: str = ""

    # ── Tool events ──

    def tool_started(self, name: str, args: str) -> None:
        """Record the start of a tool invocation. Pair each call with
        exactly one `tool_finished` — we assume serial execution, which is
        true for our agent loop (one tool at a time)."""
        self._active_tool = ToolEvent(name=name, args=args or "",
                                      start_ts=time.monotonic())

    def tool_finished(self, result_chars: int, error: bool) -> int:
        """Close out the most recent tool call and return its duration in ms
        (0 if no tool is active — e.g. a lost tool_start event). If no tool is
        active we silently drop; the loss is more useful than a noisy assertion.
        The returned duration lets the UI render a per-tool time on the done row."""
        if self._active_tool is None:
            return 0
        t = self._active_tool
        t.duration_ms = int((time.monotonic() - t.start_ts) * 1000)
        t.result_chars = result_chars
        t.error = bool(error)
        self.tools.append(t)
        self._active_tool = None
        return t.duration_ms

    # ── Finalization ──

    def finalize(self) -> dict[str, Any]:
        """Produce the JSON-serializable record. Doesn't write — that's the
        log_turn() function's job, so tests can inspect the dict directly."""
        duration_ms = int((time.monotonic() - self.started_mono) * 1000)
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp": _iso_utc(self.started_at),
            "duration_ms": duration_ms,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "app_version": self.app_version,
            "model": self.model,
            "mode": self.mode,
            "context_window": self.context_window,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "thinking_text": self.thinking_text,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total or (self.tokens_in + self.tokens_out),
            "tools": [
                {
                    "name": t.name,
                    "args": _truncate(t.args, 500),
                    "duration_ms": t.duration_ms,
                    "result_chars": t.result_chars,
                    "error": t.error,
                }
                for t in self.tools
            ],
            "skills_used": _extract_skill_names(self.tools),
            "tools_count": len(self.tools),
            "tool_errors_count": sum(1 for t in self.tools if t.error),
            "interrupted": self.interrupted,
            "error": self.error,
        }


def log_turn(trace: TurnTrace) -> None:
    """Emit a UI turn trace into the centralised event log.

    Previously wrote a separate JSONL at `~/.localcode/telemetry/turns.jsonl`,
    which duplicated data already captured via `tool_call` events in
    lifecycle.log. Now consolidated: this turn-trace becomes ONE
    `ui_turn_end` event in `.localcode/events.jsonl`. The agent loop owns
    the authoritative `turn_end`; keeping the UI trace separate avoids two
    incompatible summaries for the same turn.

    No-ops on error — telemetry failures must never disrupt the
    user's session."""
    if not telemetry_enabled():
        return
    try:
        from .events import emit
        emit("ui_turn_end", **trace.finalize())
    except Exception:
        # Deliberately swallow.
        pass


# ── internals ──

def _iso_utc(epoch_seconds: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(epoch_seconds, tz=_dt.timezone.utc).isoformat()


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def current_git_branch(cwd: str | Path) -> str:
    """Best-effort: returns branch name or '' if not in a git repo or git
    isn't installed. Called once per turn — cheap enough not to cache."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=1,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def current_app_version() -> str:
    """Returns the installed LocalCode package version, or '' if not
    installed (running from a source checkout without pip install -e)."""
    try:
        from importlib.metadata import version
        return version("localcode")
    except Exception:
        return ""


def _extract_skill_names(tools: list[ToolEvent]) -> list[str]:
    """The `skill` tool's args are JSON-ish like `{"name": "plan-task"}`.
    Pull the skill name out for a dedicated field — most common analysis
    question is 'which skills fired today'."""
    import re
    out: list[str] = []
    for t in tools:
        if t.name != "skill":
            continue
        m = re.search(r'"name"\s*:\s*"([^"]+)"', t.args or "")
        if m:
            out.append(m.group(1))
    return out
