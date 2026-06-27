"""Display-formatting utilities — truncation, wrapping, summary lines.

Used everywhere a tool result, file read, or shell stdout might be too
long to dump into the chat log unfiltered. The whole-buffer-into-the-
screen failure mode (50 KB of grep output, 800-line file dumps, etc.)
is what made our chat unscrollable in earlier sessions; this module
caps that.

Pattern stolen from agent's UserPromptMessage.tsx — head + tail with
an ellipsis row in the middle, plus a pointer to the full payload on
disk for users who actually need it.
"""
from __future__ import annotations

from pathlib import Path


def truncate_with_tail(
    text: str,
    head: int = 15,
    tail: int = 5,
    persist_label: str = "",
) -> str:
    """Truncate `text` to `head` lines + `tail` lines with an ellipsis
    row showing the count of dropped lines.

    If the text fits in `head + tail + 1` lines, returns it unchanged.

    If `persist_label` is provided AND we truncated, also writes the
    full text to `~/.localcode/last_<label>.log` and includes a
    pointer in the ellipsis row so the user can grab the original.

    Example:
        truncate_with_tail("a\\nb\\nc\\n…\\n", head=2, tail=1)
        # → "a\\nb\\n… +N more lines …\\n<last>\\n"
    """
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text

    dropped = len(lines) - head - tail
    head_block = lines[:head]
    tail_block = lines[-tail:] if tail > 0 else []

    if persist_label:
        try:
            p = Path.home() / ".localcode" / f"last_{persist_label}.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            ellipsis = f"… +{dropped} more lines (full output: ~/.localcode/last_{persist_label}.log) …"
        except Exception:
            ellipsis = f"… +{dropped} more lines …"
    else:
        ellipsis = f"… +{dropped} more lines …"

    return "\n".join(head_block + [ellipsis] + tail_block)


def split_stdout_stderr(text: str) -> tuple[str, str]:
    """Heuristic split of bash output into (stdout, stderr).

    Most tools mix the two streams. We detect lines that LOOK like
    stderr (start with "Warning:", "Error:", "Traceback", etc.) and
    bucket them. Not perfect, but distinguishes the common case of
    a successful command with diagnostic warnings from a failed one
    with errors.

    Returns ("stdout text", "stderr text"). Either may be empty.
    """
    stderr_markers = (
        "warning:", "error:", "traceback", "exception:",
        "fatal:", "fail:", "fatal error",
    )
    stdout_lines, stderr_lines = [], []
    in_traceback = False
    for line in (text or "").splitlines():
        s = line.lstrip().lower()
        # Python tracebacks span multiple lines — keep them together.
        if in_traceback:
            stderr_lines.append(line)
            if not (line.startswith("  ") or line.startswith("\t") or line == ""):
                in_traceback = False
            continue
        if s.startswith("traceback"):
            in_traceback = True
            stderr_lines.append(line)
        elif any(s.startswith(m) for m in stderr_markers):
            stderr_lines.append(line)
        else:
            stdout_lines.append(line)
    return "\n".join(stdout_lines), "\n".join(stderr_lines)
