"""JSONL event emission for `localcode run --json` (headless API).

`localcode run --goal "…"` streams human-readable, Rich-formatted output to
the terminal. `--json` instead emits the agent's event stream as JSON Lines
(one JSON object per line) on stdout, so editors / CI / other programs can
drive LocalCode programmatically (Codex-exec style).

Design
------
The agent already routes every UI-relevant signal through
`OutputManager._emit_event(event_type, **payload)` — the exact same callback
the TUI subscribes to via `OutputManager.set_event_callback`. We reuse that
single hook: install a callback that serialises each `(event_type, payload)`
to one `json.dumps` line on the *real* stdout.

Clean stdout matters for machine consumers, so while `--json` is active we:
  • swap the Rich `Console` for a silent one (model-resolution chatter, etc.),
  • redirect `OutputManager`'s own raw ANSI writes (it pokes `sys.stdout`
    directly in `log_tool` / `tool_result` / `done`) to /dev/null, and
  • write JSONL to a private duplicate of the original stdout fd.

The terminal event (`type: "result"`) carries the final status, exit reason,
accumulated token counts, and the assistant's final text so a consumer can
read a single trailing line instead of reassembling content deltas.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO


class JsonlEmitter:
    """Serialise agent events to JSONL on a fixed stream.

    One instance per headless `--json` run. `emit()` is wired in as the
    `OutputManager` event callback; `result()` writes the terminal event.
    Token counts are accumulated from the `turn_tokens` events the agent
    loop fires once per round.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def _write(self, obj: dict[str, Any]) -> None:
        # Single line, no pretty-printing — each line must be one valid
        # JSON object. default=str keeps a stray non-serialisable value
        # from ever breaking the stream.
        try:
            line = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:
            pass

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """OutputManager event callback: `(event_type, payload_dict)`."""
        if event_type == "turn_tokens":
            # Accumulate across rounds for the final summary. Values arrive
            # as strings (OutputManager.update_turn_tokens stringifies them).
            self.prompt_tokens += _as_int(payload.get("prompt_tokens"))
            self.completion_tokens += _as_int(payload.get("completion_tokens"))
            self.total_tokens += _as_int(payload.get("total_tokens"))
        record: dict[str, Any] = {"type": event_type}
        record.update(payload)
        self._write(record)

    def result(self, *, status: str, exit_code: int, reason: str,
               final_text: str = "") -> None:
        """Terminal event — always the last line of the stream."""
        self._write({
            "type": "result",
            "status": status,
            "exit_code": exit_code,
            "reason": reason,
            "final_text": final_text,
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens or (self.prompt_tokens + self.completion_tokens),
            },
        })


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def open_clean_stdout() -> TextIO:
    """Return a writable text stream bound to the *original* stdout fd.

    We duplicate fd 1 before redirecting `sys.stdout` to /dev/null, so the
    emitter keeps a clean channel for JSONL even after the agent's raw
    ANSI writes are silenced. Falls back to the live `sys.stdout` if the
    fd can't be duplicated (e.g. it's already a non-fd object in tests).
    """
    import os
    try:
        fd = os.dup(sys.stdout.fileno())
        return os.fdopen(fd, "w", encoding="utf-8", closefd=True)
    except Exception:
        return sys.stdout
