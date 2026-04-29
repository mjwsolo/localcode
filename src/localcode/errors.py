"""Error code registry — every user-visible error gets a stable ID + remediation.

Audience for this module: the LocalCode team (developers + managers reviewing
turn telemetry). NOT end-users. End-users see the rendered output —
`[E2103] Unknown tool 'list_files ' — fix: …` — but the registry behind
it is for triage:

    1. Bug reports become self-routing. "I got E3105" is one grep away
       from the file that raises it, instead of needing to grep the
       free-text message across 30k LOC.
    2. The remediation field doubles as inline docs — no need to
       cross-reference a wiki to know what a user should do.
    3. `docs/ERRORS.md` is generated from this registry (run
       `python -m localcode.errors --emit-docs`), so the docs can never
       drift from the code.

### Code ranges

  E1xxx — Setup / startup        (server didn't start, model not found)
  E2xxx — Tool dispatch          (unknown tool, malformed args, denied)
  E3xxx — Runtime / model        (HTTP disconnect, EOS-too-early, OOM)
  E4xxx — Filesystem / git       (path not found, IsADirectory, blocked)
  E5xxx — User cancellation      (stop intent, loop-breaker tripped)
  E9xxx — Wrapped unknown        (catch-all — captures original exc class)

### Adding a new code

Append a new `ErrorCode` instance to `_REGISTRY` below. Keep:
- `code`: stable; never re-use a number after retiring one.
- `summary`: ≤60 chars; what failed in plain English.
- `remediation`: ≤120 chars; one concrete action the user can take.

If the failure is one we DETECT but is a model-side issue (e.g. EOS
too early), set `cause="model"` — it changes the rendering accent so
the user knows it's not their fault.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Cause = Literal["user", "model", "system", "internal"]


@dataclass(frozen=True)
class ErrorCode:
    code: str                    # e.g. "E2103"
    summary: str                 # short title shown to user
    remediation: str             # one-line fix hint
    cause: Cause = "internal"    # who's "at fault" — affects display accent


@dataclass(frozen=True)
class LocalCodeError(Exception):
    """Throwable wrapper for an error code + dynamic context."""
    code: ErrorCode
    detail: str = ""             # optional extra context (path, name, etc.)

    def __str__(self) -> str:
        if self.detail:
            return f"[{self.code.code}] {self.code.summary}: {self.detail}"
        return f"[{self.code.code}] {self.code.summary}"


# ── Registry ─────────────────────────────────────────────────────────

_REGISTRY: list[ErrorCode] = [
    # E1xxx — Setup / startup
    ErrorCode("E1001", "Server failed to start",
              "Run `localcode setup` or check ~/.local/share/localcode/server.log",
              cause="system"),
    ErrorCode("E1002", "Server didn't come up in time",
              "Try again. If it persists, re-run `localcode setup`.", cause="system"),
    ErrorCode("E1003", "Model file not found",
              "Run `localcode setup` to download the model, or pick another via /model.",
              cause="user"),
    ErrorCode("E1004", "Backend not initialized",
              "Type a message to trigger backend startup.", cause="user"),
    ErrorCode("E1010", "Insufficient memory to launch",
              "Quit other heavy apps (browsers, IDEs, Docker) and retry.", cause="user"),
    ErrorCode("E1011", "Stuck llama-server from prior session",
              "Restart your Mac to clear the GPU-wait — auto-recovery couldn't unstick it.",
              cause="system"),

    # E2xxx — Tool dispatch
    ErrorCode("E2101", "Unknown tool",
              "The model emitted a tool name that isn't registered. Likely a quantization artifact.",
              cause="model"),
    ErrorCode("E2102", "Malformed tool arguments",
              "The model emitted invalid JSON for the tool's arguments.", cause="model"),
    ErrorCode("E2103", "Tool name had stray whitespace",
              "Auto-stripped at dispatch — the call still ran. Logged for telemetry.",
              cause="model"),
    ErrorCode("E2104", "Missing required argument",
              "The model omitted a required parameter for the tool. Often retried in the next round.",
              cause="model"),
    ErrorCode("E2110", "Tool denied by permission policy",
              "The user (or session policy) declined this tool call.", cause="user"),
    ErrorCode("E2111", "Tool blocked by hook",
              "A configured hook in ~/.localcode/hooks.toml refused the tool call.",
              cause="user"),

    # E3xxx — Runtime / model
    ErrorCode("E3101", "The model stopped responding too early",
              "Try again. If it keeps happening, switch model with /model.",
              cause="model"),
    ErrorCode("E3102", "Lost connection to the model server",
              "Try again. If it persists, quit LocalCode and relaunch.", cause="system"),
    ErrorCode("E3103", "Conversation is too long for this model",
              "Type /clear to start a fresh conversation.", cause="user"),
    ErrorCode("E3104", "The GPU couldn't run the model",
              "Quit big apps to free memory and try again. If it keeps failing, restart your Mac.",
              cause="system"),
    ErrorCode("E3105", "The model is still loading",
              "Wait a few seconds and try again — the model takes a moment to warm up.",
              cause="system"),
    ErrorCode("E3106",
              "macOS paused your model server to protect the system from running out of memory",
              "We auto-restart on your next message — usually no action needed. To prevent it: "
              "close memory-heavy apps (browsers with many tabs, IDEs with large projects, video "
              "editors), or `/model` to a smaller quant. Your Mac was at critical memory pressure "
              "during your last request, so the safety monitor freed memory by killing the server "
              "before macOS itself would have force-killed it.",
              cause="system"),

    # E4xxx — Filesystem / git
    ErrorCode("E4101", "Path is a directory, not a file",
              "Use `bash mkdir -p <path>` to create directories; write_file is for files.",
              cause="model"),
    ErrorCode("E4102", "Path not found",
              "Check the path. The model may have hallucinated it.", cause="model"),
    ErrorCode("E4103", "Permission denied (filesystem)",
              "macOS may be sandboxing this directory. Try a path under your home dir.",
              cause="system"),
    ErrorCode("E4110", "Git command failed",
              "Check `git status` for repo state; the failed command's output is logged above.",
              cause="user"),

    # E5xxx — User cancellation / loop-breakers
    ErrorCode("E5101", "Cancelled by user",
              "You typed 'stop' / 'cancel' / 'abort' during a running turn.", cause="user"),
    ErrorCode("E5102", "Loop-breaker: same call 3× in a row",
              "Model was stuck. Tell it what you actually want, more concretely.",
              cause="model"),
    ErrorCode("E5103", "Loop-breaker: too many tool calls in one turn",
              "Model was thrashing. Break the task into smaller steps.", cause="model"),
    ErrorCode("E5104", "Loop-breaker: file edited too many times",
              "Model was oscillating. Read the file and tell it what's actually wrong.",
              cause="model"),

    # E9xxx — Wrapped unknown
    ErrorCode("E9001", "Unhandled exception in agent loop",
              "Internal bug. Paste the exception type + message into a github issue.",
              cause="internal"),
    ErrorCode("E9002", "Unhandled exception in TUI",
              "Internal bug. Restart localcode; if it recurs, file a github issue.",
              cause="internal"),
]

# Quick lookup by code string.
BY_CODE: dict[str, ErrorCode] = {c.code: c for c in _REGISTRY}


def by_code(code: str) -> ErrorCode | None:
    """Lookup helper. Returns None for unknown codes (do NOT raise — error
    paths must never raise themselves)."""
    return BY_CODE.get(code)


def _detect_known_pattern(err_str: str) -> str | None:
    """Match known error strings (from llama-server, http stack, etc.)
    to a more specific code than the catch-all fallback. This lets the
    user see "the GPU couldn't run the model" instead of an HTTP 500
    JSON dump. Returns a code string or None."""
    s = err_str.lower()
    if "compute error" in s or "ggml" in s and "metal" in s:
        return "E3104"            # GPU compute failure
    if "context" in s and ("exceed" in s or "overflow" in s):
        return "E3103"            # context window
    if "connecterror" in s or "connection refused" in s:
        # Connection-refused has TWO common causes on a local llama-server:
        # (a) the pressure monitor SIGTERM'd it because macOS reported
        #     CRITICAL memory pressure, OR
        # (b) the server crashed / hasn't been started yet.
        # Distinguish by checking for a recent pressure-kill marker so the
        # user sees "memory pressure killed your model server" — actionable
        # — instead of a generic "lost connection / try again" that doesn't
        # explain what happened or how to prevent it.
        if _recent_pressure_kill():
            return "E3106"        # pressure-kill (specific to memory pressure)
        return "E3102"            # generic connection lost
    if "timeout" in s or "timed out" in s:
        return "E3105"            # timeout
    return None


def _recent_pressure_kill(within_seconds: int = 120) -> bool:
    """True if a `pressure_kill` event was emitted in the last
    `within_seconds` seconds. Reads from the centralised event log
    (events.jsonl) so we have ONE source of truth — no separate
    marker file to keep in sync. Defensive on every I/O — never
    raises out of the error path.

    Falls back to the legacy `last-pressure-kill.txt` marker file if
    the event log doesn't have a recent kill but the marker exists,
    so users mid-upgrade don't lose pressure detection.
    """
    try:
        from .events import find_recent
        if find_recent("pressure_kill", within_seconds=within_seconds) is not None:
            return True
    except Exception:
        pass
    # Legacy fallback (kept until the marker file is fully phased out).
    try:
        import time
        from .paths import pressure_kill_marker_path
        marker = pressure_kill_marker_path()
        if not marker.is_file():
            return False
        body = marker.read_text()
        for line in body.splitlines():
            if line.startswith("time="):
                try:
                    ts = int(line.split("=", 1)[1])
                except ValueError:
                    return False
                return (time.time() - ts) <= within_seconds
        return False
    except Exception:
        return False


def _persist_full_detail(prefix: str, full_text: str) -> bool:
    """Write the verbose technical detail to
    `<project_root>/.localcode/last_error.log` so it's not lost — but
    doesn't pollute the user's screen. Per-project so an error in
    project A doesn't get mixed with logs from project B.
    Returns True if the write succeeded."""
    try:
        from .paths import last_error_log_path
        p = last_error_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{prefix}\n\n{full_text}")
        return True
    except Exception:
        return False


def format_for_user(err: BaseException | LocalCodeError | str,
                    fallback_code: str = "E9001") -> str:
    """Render any error into the canonical `[Eccc] summary — fix: <action>` form.

    Display rule: when we matched a SPECIFIC code (anything other than
    the catch-all E9001), the summary alone tells the user what happened
    — adding the raw HTTP/JSON/exception detail just adds noise. So we
    HIDE the detail in those cases and only persist it to the log file.
    For the unknown-error catch-all (E9001), we DO show a short
    one-liner because that's the only signal the user has about what
    went wrong.
    """
    pattern_matched = False
    if isinstance(err, LocalCodeError):
        code = err.code
        detail = err.detail
        pattern_matched = True   # explicit codes are always specific
    elif isinstance(err, BaseException):
        text = str(err)
        better = _detect_known_pattern(text)
        if better:
            pattern_matched = True
            code = BY_CODE.get(better) or _REGISTRY[-1]
            detail = text
        else:
            code = BY_CODE.get(fallback_code) or _REGISTRY[-1]
            detail = f"{type(err).__name__}: {text}" if text else type(err).__name__
    else:
        code = BY_CODE.get(fallback_code) or _REGISTRY[-1]
        detail = str(err)

    body = f"[{code.code}] {code.summary}"

    # Show a short detail ONLY when we couldn't match a specific code.
    # When code is specific (E3104 = "GPU couldn't run model"), the
    # summary IS the user-visible explanation — no raw HTTP / JSON.
    if not pattern_matched and detail:
        short_detail = detail.split("\n", 1)[0]
        if len(short_detail) > 80:
            short_detail = short_detail[:77].rstrip() + "…"
        body += f" — {short_detail}"

    if code.remediation:
        body += f"\nfix: {code.remediation}"

    # Persist the full detail for post-hoc diagnosis no matter what,
    # but never spam it on the user's screen.
    if detail:
        if _persist_full_detail(f"[{code.code}] {code.summary}", detail):
            body += "\n(full technical detail saved to ~/.localcode/last_error.log)"

    return body


# ── Doc generation ───────────────────────────────────────────────────

def emit_docs() -> str:
    """Render the registry as a markdown reference. Used to (re)generate
    docs/ERRORS.md so the docs are never out of sync with the code."""
    out = ["# LocalCode Error Codes", ""]
    out.append("Every user-facing error in LocalCode has a stable `Eccc` code.")
    out.append("This file is **generated from `src/localcode/errors.py`**;")
    out.append("don't hand-edit it. To add or change a code, edit the registry")
    out.append("then run `python -m localcode.errors --emit-docs > docs/ERRORS.md`.")
    out.append("")
    last_prefix = ""
    for c in _REGISTRY:
        prefix = c.code[:2]                       # E1, E2, ...
        if prefix != last_prefix:
            section = {
                "E1": "## E1xxx — Setup / startup",
                "E2": "## E2xxx — Tool dispatch",
                "E3": "## E3xxx — Runtime / model",
                "E4": "## E4xxx — Filesystem / git",
                "E5": "## E5xxx — User cancellation / loop-breakers",
                "E9": "## E9xxx — Wrapped unknown",
            }.get(prefix, f"## {prefix}xxx")
            out.append(section)
            out.append("")
            last_prefix = prefix
        out.append(f"### `{c.code}` — {c.summary}")
        out.append(f"- **Cause:** `{c.cause}`")
        out.append(f"- **Remediation:** {c.remediation}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--emit-docs":
        print(emit_docs())
    else:
        # Pretty-print the registry for quick inspection.
        for c in _REGISTRY:
            print(f"{c.code}  {c.summary}")
            print(f"        fix: {c.remediation}  ({c.cause})")
