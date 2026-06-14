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


def run_headless_json(config, args) -> int:
    """Headless run that emits the agent event stream as JSONL on stdout.

    Same backend as entrypoint._run_headless (LocalCodeApp + `.ask`, full-auto
    approvals) but with all human/Rich output suppressed: model-resolution
    chatter goes to a silent Console, the agent's own raw ANSI writes
    (OutputManager pokes sys.stdout directly) are redirected to /dev/null, and
    structured events are written as JSON Lines to a private duplicate of the
    original stdout via the OutputManager event callback.

    Exit codes match _run_headless: 0 ok · 1 error · 124 timeout · 130
    interrupted. The final line is always a `result` event carrying the
    status, exit reason, and accumulated token counts.
    """
    import os
    import sys as _sys
    from pathlib import Path as _Path

    from rich.console import Console as _Console

    out_stream = open_clean_stdout()
    emitter = JsonlEmitter(out_stream)

    def _emit_result(status: str, code: int, reason: str, text: str = "") -> int:
        emitter.result(status=status, exit_code=code, reason=reason, final_text=text)
        return code

    _Console(file=open(os.devnull, "w"), quiet=True)  # silence resolve/start chatter

    from .app import LocalCodeApp
    from .server_manager import _probe_health
    from .bootstrap import get_model_path
    from .models_catalog import CHOICES

    resolved: _Path | None = None
    for candidate in (args.model, config.runtime.model):
        name = _Path(candidate).name if candidate else None
        if name and name.endswith(".gguf"):
            resolved = get_model_path(name)
            if resolved:
                break
    if resolved is None:
        downloaded = [c for c in CHOICES if c.local_path.exists()]
        if downloaded:
            resolved = min(downloaded, key=lambda c: c.size_gb).local_path
    if resolved is None:
        return _emit_result("error", 1, "no model found on disk")
    config.runtime.model = str(resolved)
    if args.binary:
        config.runtime.llama_cpp_binary = args.binary

    app = LocalCodeApp(config, profile_name=args.profile)
    app.out.set_event_callback(emitter.emit)

    if not any(_probe_health(p, timeout=1.0) for p in range(8081, 8100)):
        if not app.engine._restart_server():
            return _emit_result("error", 1, "could not start the model server")

    if args.timeout and args.timeout > 0:
        import signal

        def _on_timeout(_sig, _frame):
            raise TimeoutError(f"run exceeded {args.timeout}s")

        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(args.timeout)

    devnull = open(os.devnull, "w")
    saved_stdout = _sys.stdout
    saved_fd = None
    try:
        saved_fd = os.dup(1)
        os.dup2(devnull.fileno(), 1)
    except Exception:
        saved_fd = None
    _sys.stdout = devnull

    try:
        result_text = app.ask(args.goal, stream=True)
    except TimeoutError:
        return _emit_result("timeout", 124, f"run exceeded {args.timeout}s")
    except KeyboardInterrupt:
        return _emit_result("interrupted", 130, "keyboard interrupt")
    except Exception as e:  # noqa: BLE001 — headless: surface any failure as exit 1
        return _emit_result("error", 1, f"{type(e).__name__}: {e}")
    finally:
        if args.timeout and args.timeout > 0:
            import signal
            signal.alarm(0)
        _sys.stdout = saved_stdout
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
            except Exception:
                pass
        try:
            devnull.close()
        except Exception:
            pass

    return _emit_result("ok", 0, "completed", result_text or "")


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
