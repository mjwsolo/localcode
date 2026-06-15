"""Context-management pipeline — redaction, compaction, aging.

Extracted from agent/__init__.py during T0.1-c. No behaviour change —
every function here is verbatim from the pre-split module. The loop
imports them from `.context` instead of having them in the same file.

Why these nine functions together:
  They all shape the `messages` list that goes to the model. Some
  truncate tool results inline (`_truncate_result`). Some age older
  tool-call payloads so the context window doesn't explode
  (`_redact_old_write_args`, `_redact_duplicate_reads`,
  `_compact_old_tool_results`). Some bound / estimate the total
  (`_msg_bytes`, `_estimate_tokens`). The top-level `_prepare_model_messages`
  composes the aging pipeline in order. `_compact_messages` is the
  LLM-summary fallback when the deterministic pipeline still leaves
  the context too big. `_summarize_args` formats tool-call args for
  both compaction-summary text and live tool logging.

Nothing here is loop logic or model-specific. Pure functions over the
message-list shape we pass to the runtime.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .constants import (
    COMPACT_KEEP_RECENT_TOOL_RESULTS,
    COMPACT_MIN_CONTENT_CHARS,
    REDACT_KEEP_RECENT_WRITES,
    REDACT_MIN_CONTENT_CHARS,
    READ_UNCHANGED_STUB_PREFIX,
    RESULT_LIMITS,
)

if TYPE_CHECKING:
    from ..output import OutputManager


# Every helper here is intentionally private (underscore-prefixed).
# The aging pipeline internals may be reshaped freely as long as
# `_prepare_model_messages` and `_compact_messages` keep the
# behaviour the loop expects. `__all__` is empty deliberately —
# callers import by name rather than through a public surface.
__all__: list[str] = []


def _truncate_result(result: str, tool_name: str) -> str:
    """Truncate a tool result to its per-tool size limit with a
    strategy tuned to what each tool's output actually looks like.

    Strategies:
      • `read_file` — keep HEAD only. File content is ordered; the
        model can paginate with `offset=` to see more. Middle-dropping
        gives the model the illusion of complete context when it only
        saw half. Replace dropped tail with an explicit offset hint.
      • `grep` — keep HEAD only. Individual matches are small; losing
        trailing matches to a count is fine (the hint tells the
        model to narrow the pattern if it needs more).
      • `bash` — prioritise the TAIL (most recent output, which
        typically contains the error / result) plus a small HEAD
        (which often contains the command echo + initial status).
      • default — original middle-drop, a reasonable fallback for
        tools whose output structure we haven't characterised.

    Returns the result unchanged when it's within the limit.
    """
    limit = RESULT_LIMITS.get(tool_name, RESULT_LIMITS["default"])
    if len(result) <= limit:
        return result

    dropped = len(result) - limit

    if tool_name == "read_file":
        # Keep the HEAD; if we can find line numbers in the content
        # (read_file numbers lines "N\t..."), compute the offset the
        # model should call with next.
        head = result[:limit]
        # Estimate next-offset by counting newlines in head + 1.
        # Line-numbered output makes this tighter than any heuristic.
        last_numbered = None
        for line in head.rsplit("\n", 200)[-200:]:
            # Lines look like "1234\tsome content". Pull leading int.
            if "\t" in line and line.split("\t", 1)[0].isdigit():
                last_numbered = int(line.split("\t", 1)[0])
        hint = (
            f"\n\n[truncated — {dropped} more chars not shown. "
            f"{'Call read_file with offset=' + str(last_numbered) + ' to continue reading from there.' if last_numbered else 'Call read_file with a larger offset to continue.'}]"
        )
        return head + hint

    if tool_name == "grep":
        # Keep HEAD only. Grep output is one-match-per-line; losing
        # trailing matches to a counter is OK — if the model needed
        # more hits, it should narrow the pattern.
        head = result[:limit]
        # Count how many lines we've dropped so the model knows
        # whether to broaden or narrow.
        dropped_lines = result[limit:].count("\n") + 1
        hint = (
            f"\n\n[truncated — {dropped_lines} more match-lines not shown. "
            "Narrow with a more specific pattern or `include=*.py` to see them.]"
        )
        return head + hint

    if tool_name == "bash":
        # Bash output is the noisiest path in long app tasks. Prefer a
        # structured summary over raw head/tail when the output looks
        # like a directory tree or a huge install/build log; otherwise
        # keep the most recent tail with a short head for context.
        lines = result.splitlines()
        non_empty = [ln for ln in lines if ln.strip()]
        if len(lines) >= 40:
            first_sig = "\n".join(non_empty[:8])
            last_sig = "\n".join(non_empty[-12:]) if len(non_empty) > 12 else ""
            marker = (
                f"\n\n[... {dropped} chars of bash output compressed ...]\n\n"
            )
            if any(token in result for token in ("backend", "frontend", "src/", "node_modules", "package.json", "pyproject.toml")):
                tree_hint = (
                    f"[bash output summarized: {len(lines)} lines, "
                    f"{len(non_empty)} non-empty lines]\n"
                )
                return tree_hint + first_sig + marker + last_sig
            if any(token in result.lower() for token in ("vite", "webpack", "npm run", "uvicorn", "fastapi", "build", "install")):
                log_hint = (
                    f"[bash output summarized: {len(lines)} lines, build/install log]\n"
                )
                return log_hint + first_sig + marker + last_sig
        # Prioritise TAIL (recent output = errors, final result)
        # with a small HEAD for command echo / initial status.
        head_size = min(limit // 4, 1000)
        tail_size = limit - head_size - 64  # 64 chars for the marker line
        head = result[:head_size]
        tail = result[-tail_size:]
        marker = f"\n\n[... {dropped} chars of middle output dropped ...]\n\n"
        return head + marker + tail

    # Default — middle-drop.
    half = limit // 2
    return (
        result[:half]
        + f"\n\n[...{dropped} chars truncated...]\n\n"
        + result[-half:]
    )


# COMPACT_KEEP_RECENT_TOOL_RESULTS / COMPACT_MIN_CONTENT_CHARS moved
# to constants.py during the T0.1 split; re-exported at the top of
# this module for back-compat.

def _compact_old_tool_results(
    messages: list[dict], keep_recent: int = COMPACT_KEEP_RECENT_TOOL_RESULTS
) -> list[dict]:
    """Return a copy of `messages` with older tool-role results summarized.

    Gated on `Feature.TOOL_RESULT_AGING` — when disabled the caller gets
    the input list back unchanged, which is what eval uses to A/B
    "how much does aging actually save per turn?"

    We keep the last `keep_recent` tool results verbatim (the model usually
    only needs the recent ones to decide the next step) and replace earlier
    ones with a compact "[summarized ...]" placeholder. `keep_recent` scales
    with the context window (more on a big machine — see
    `_prepare_model_messages`), so a 128K-window session preserves far more
    raw tool output than a 16K one instead of crushing both to the same 4.
    User/assistant/system messages pass through unchanged, and `tool_call_id`
    is preserved so the chat protocol still reconciles ids correctly.
    """
    from ..features import Feature, is_enabled
    if not is_enabled(Feature.TOOL_RESULT_AGING):
        return messages
    keep_recent = max(1, keep_recent)
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_idxs) <= keep_recent:
        return messages

    cutoff_idx = tool_idxs[-keep_recent]
    out: list[dict] = []
    for i, m in enumerate(messages):
        if i >= cutoff_idx or m.get("role") != "tool":
            out.append(m)
            continue
        content = m.get("content") or ""
        if not isinstance(content, str) or len(content) < COMPACT_MIN_CONTENT_CHARS:
            out.append(m)
            continue
        summary = _semantic_tool_summary(content)
        out.append({**m, "content": summary})
    return out


def _semantic_tool_summary(content: str) -> str:
    """Compact old tool output by preserving operational facts first."""
    lines = content.splitlines()
    first_line = next((ln.strip()[:120] for ln in lines if ln.strip()), "")
    facts = ""
    if "[tool facts:" in content:
        facts = content[content.find("[tool facts:"):].splitlines()[0][:300]
    is_error = (
        content.startswith("Error:")
        or content.startswith("REJECTED:")
        or content.startswith("[exit code ")
        or "old_string not found" in content
        or "Traceback " in content
    )
    if is_error:
        tail = "\n".join([ln for ln in lines[-12:] if ln.strip()])[:1200]
        return (
            f"[older tool error preserved: {len(content)} chars, "
            f"{len(lines)} lines]\n{first_line}\n{tail}"
        )
    if facts:
        return (
            f"[older successful tool result summarized: {len(content)} chars, "
            f"{len(lines)} lines]\n{facts}"
        )
    if any(token in content for token in ("http://localhost:", "127.0.0.1:", "PID:", "URL:")):
        important = [
            ln.strip()
            for ln in lines
            if any(token in ln for token in ("http://localhost:", "127.0.0.1:", "PID:", "URL:", "Log:"))
        ][:8]
        return (
            f"[older runtime result summarized: {len(content)} chars, "
            f"{len(lines)} lines]\n" + "\n".join(important)
        )
    return (
        f"[older tool result summarized: {first_line} … "
        f"({len(content)} chars, {len(lines)} lines)]"
    )


# REDACT_KEEP_RECENT_WRITES / REDACT_MIN_CONTENT_CHARS moved to
# constants.py during the T0.1 split; re-exported at the top of
# this module for back-compat.

def _redact_old_write_args(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with OLD write-tool arguments redacted.

    The single biggest source of context bloat on coding-heavy sessions
    is this: every write_file call carries its full `content` (often
    hundreds of lines of code) in the assistant's tool_calls arguments.
    That payload then sits in history forever and is re-shipped to the
    model on every subsequent round — even though the file is already
    on disk and a cheap read_file can reload it when needed.

    This mirrors the agent `microcompact` pattern (see
    `microCompactCore.ts:446`): keep the last N verbatim, replace older
    ones with a short placeholder that tells the model "the payload is
    gone but the file is still on disk — use read_file to see it."
    Applied to write_file, edit_file, multi_edit since those are the
    tools whose argument payloads dominate history size. Other tools
    (bash, grep, read_file itself) have bounded or short arg payloads
    so they don't need redaction — their BULK is in the tool_result
    instead, which `_compact_old_tool_results` already handles.

    Leaves the most-recent `REDACT_KEEP_RECENT_WRITES` calls untouched
    so the model can reference the code it JUST wrote without a
    read_file round-trip. Preserves tool_call id + name + `path` arg
    (those are small and needed to reconcile the history chain). Only
    strips the bulky payload field.

    Returns a new list; never mutates the input.

    Gated on `Feature.WRITE_ARG_REDACTION` — disabling returns the
    input unchanged, which is what eval uses to A/B "how much bloat
    is this preventing per session?"
    """
    from ..features import Feature, is_enabled
    if not is_enabled(Feature.WRITE_ARG_REDACTION):
        return messages

    REDACT_TOOLS = {"write_file", "append_file", "edit_file", "multi_edit"}

    # Collect (outer_idx, tool_call_idx, tool_name) for every targeted call.
    targets: list[tuple[int, int, str]] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        for j, tc in enumerate(m.get("tool_calls") or []):
            name = ((tc.get("function") or {}).get("name") or "").strip()
            if name in REDACT_TOOLS:
                targets.append((i, j, name))

    if len(targets) <= REDACT_KEEP_RECENT_WRITES:
        return messages

    # Everything except the last N is a redaction candidate.
    to_redact = set(targets[:-REDACT_KEEP_RECENT_WRITES])
    if not to_redact:
        return messages

    out: list[dict] = []
    for i, m in enumerate(messages):
        tcs = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not tcs:
            out.append(m)
            continue
        new_tcs = []
        touched = False
        for j, tc in enumerate(tcs):
            name = ((tc.get("function") or {}).get("name") or "").strip()
            if (i, j, name) not in to_redact:
                new_tcs.append(tc)
                continue
            # Redact: parse args, strip the bulky field, re-serialise.
            try:
                raw_args = (tc.get("function") or {}).get("arguments") or "{}"
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                new_tcs.append(tc)
                continue
            path = args.get("path") or "?"
            size_hint = ""
            if name in {"write_file", "append_file"}:
                content = args.get("content") or ""
                if isinstance(content, str) and len(content) >= REDACT_MIN_CONTENT_CHARS:
                    # DROP the `content` key entirely instead of replacing
                    # it with a stub string. Earlier we put a "[REDACTED
                    # from history — …]" sentinel into `args["content"]`
                    # to inform the model the file existed but the bytes
                    # were elided. IQ2_M Qwen, on the next round, would
                    # SEE that string in its own past tool_call and copy
                    # it verbatim as the `content` arg of a NEW write_file
                    # call — overwriting the real file on disk with the
                    # stub itself. Dropping the
                    # key means there's nothing to copy. The synthetic
                    # `_redaction_note` sibling tells the model what
                    # happened without exposing a copyable `content`.
                    size_hint = f" ({len(content)} chars, {content.count(chr(10)) + 1} lines)"
                    args.pop("content", None)
                    args["_redaction_note"] = (
                        f"content elided from history{size_hint}. "
                        f"file is on disk at {path}. "
                        f"call read_file path={path!r} to reload."
                    )
                    touched = True
            elif name == "edit_file":
                old_s = args.get("old_string") or ""
                new_s = args.get("new_string") or ""
                if isinstance(new_s, str) and len(new_s) + len(old_s) >= REDACT_MIN_CONTENT_CHARS:
                    size_hint = f" ({len(new_s)} chars new, {len(old_s)} chars old)"
                    args["old_string"] = f"[REDACTED{size_hint}]"
                    args["new_string"] = (
                        f"[REDACTED — edit applied to {path}. "
                        f"Call read_file with path={path!r} to see current content.]"
                    )
                    touched = True
            elif name == "multi_edit":
                edits = args.get("edits") or []
                if isinstance(edits, list) and edits:
                    total = sum(
                        len((e or {}).get("new_string") or "")
                        + len((e or {}).get("old_string") or "")
                        for e in edits
                    )
                    if total >= REDACT_MIN_CONTENT_CHARS:
                        args["edits"] = [
                            {"old_string": f"[REDACTED edit {k+1}/{len(edits)}]",
                             "new_string": f"[REDACTED — {len(edits)} edits applied to {path}. "
                                           f"Call read_file with path={path!r} to see current content.]"}
                            for k in range(len(edits))
                        ]
                        touched = True
            if touched:
                # Rebuild the tool_call with the redacted arguments.
                new_fn = {**(tc.get("function") or {}), "arguments": json.dumps(args)}
                new_tcs.append({**tc, "function": new_fn})
            else:
                new_tcs.append(tc)
        out.append({**m, "tool_calls": new_tcs})
    return out


# READ_UNCHANGED_STUB_PREFIX moved to constants.py during the T0.1
# split; re-exported at the top of this module for back-compat.

def _redact_duplicate_reads(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with older duplicate read_file results stubbed.

    Walks the message list once. For each read_file tool_use -> tool_result
    pair, tracks which paths have been read. The LAST read for each path
    keeps its full result; earlier ones get replaced with
    `READ_UNCHANGED_STUB_PREFIX`. The tool_call itself (in the assistant
    message) is untouched — only the result content shrinks.

    This runs alongside `_redact_old_write_args` and
    `_compact_old_tool_results`. The three are complementary:
      • write-arg redaction shrinks assistant → tool_calls payloads
      • duplicate-read stubbing shrinks tool role results for read_file
      • tool-result aging shrinks tool role results for everything else

    Gated on `Feature.DUPLICATE_READ_STUB`. Disabled returns the input
    unchanged.
    """
    from ..features import Feature, is_enabled
    if not is_enabled(Feature.DUPLICATE_READ_STUB):
        return messages
    # First pass: build map from tool_call_id → (path, outer_idx) for
    # every read_file call. We need the id to match it against the
    # corresponding tool-role result message later.
    read_calls: dict[str, str] = {}  # id → path
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = ((tc.get("function") or {}).get("name") or "").strip()
            if name != "read_file":
                continue
            tc_id = tc.get("id")
            if not tc_id:
                continue
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except Exception:
                continue
            path = args.get("path")
            if isinstance(path, str) and path:
                read_calls[tc_id] = path

    if not read_calls:
        return messages

    # Second pass: find the tool-role result index for each read_file call
    # and group by path. Keep the highest-index (most recent) per path;
    # stub everything else for the same path.
    path_to_result_idxs: dict[str, list[int]] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        tc_id = m.get("tool_call_id")
        if tc_id in read_calls:
            path_to_result_idxs.setdefault(read_calls[tc_id], []).append(i)

    # Any path with 2+ reads is a redaction candidate.
    to_stub: set[int] = set()
    for path, idxs in path_to_result_idxs.items():
        if len(idxs) >= 2:
            # Keep the last; stub all earlier ones.
            to_stub.update(idxs[:-1])

    if not to_stub:
        return messages

    out: list[dict] = []
    for i, m in enumerate(messages):
        if i not in to_stub:
            out.append(m)
            continue
        content = m.get("content") or ""
        if not isinstance(content, str) or len(content) < REDACT_MIN_CONTENT_CHARS:
            out.append(m)
            continue
        out.append({**m, "content": READ_UNCHANGED_STUB_PREFIX})
    return out

def _msg_bytes(messages: list[dict]) -> int:
    """Approximate serialized size of `messages` for telemetry.

    Uses `json.dumps` so we count tool_call argument payloads and
    tool_result content the same way the API client will. Cheap enough
    to call once per round; not exact-token-count, but proportional
    to wire size which is what we actually care about.
    """
    try:
        return len(json.dumps(messages))
    except Exception:
        return 0

def _window_aware_compaction(ctx_window_chars: int | None) -> tuple[int, int]:
    """(history_budget_bytes, keep_recent_tool_results) scaled to the context
    window. The whole point: compaction must be DYNAMIC per machine. A 16 GB
    Mac with a ~16K window has no room, so aggressive lossy compaction (the
    old fixed 36 KB / keep-4) is correct. A 128 GB Mac with a 128K window has
    ~8x the room — crushing its history to 36 KB threw away the data model the
    model needs and made it lose track / re-read / drift. So budget ~55% of
    the window to history and keep proportionally more recent tool output.

    `ctx_window_chars` ≈ num_ctx tokens × ~3.5 chars/token. None/unknown →
    the legacy fixed behaviour (back-compat, safe on tiny windows).
    """
    if not ctx_window_chars or ctx_window_chars <= 0:
        return 36_000, COMPACT_KEEP_RECENT_TOOL_RESULTS
    budget = max(36_000, int(ctx_window_chars * 0.55))
    keep = max(COMPACT_KEEP_RECENT_TOOL_RESULTS, min(24, ctx_window_chars // 20_000))
    return budget, keep


def _prepare_model_messages(
    messages: list[dict], ctx_window_chars: int | None = None
) -> list[dict]:
    """One-stop context-shrink pass before sending to the model.

    `ctx_window_chars` (the model's context window in chars, RAM-derived) makes
    compaction window-aware: big machines keep far more history, small ones
    stay aggressive. None preserves the legacy fixed budget.

    Composes the three reduction passes in a fixed order:
      1. `_redact_old_write_args` — strip bulky content from old
         write_file / edit_file / multi_edit tool_call arguments.
      2. `_redact_duplicate_reads` — collapse older read_file results
         for the same path to a stub pointing at the newer read.
      3. `_compact_old_tool_results` — summarize old tool-role results
         for every other tool (bash, grep, etc.).
    All three return copies, so the original `messages` list is
    untouched. Order matters: (1) and (2) each shrink different parts
    of the history, and (3) catches whatever's left from (1) and (2)
    plus unrelated tools.

    Telemetry: when any pass shrinks the message list, we append a
    `redaction` line to ~/.localcode/lifecycle.log with per-pass byte
    deltas. Lets us answer "how much is each layer actually saving in
    practice?" without instrumenting the agent loop. Silent on no-op
    so the log doesn't fill up with empty events.
    """
    budget_bytes, keep_recent = _window_aware_compaction(ctx_window_chars)
    before = _msg_bytes(messages)
    after_writes = _redact_old_write_args(messages)
    bytes_writes = before - _msg_bytes(after_writes)
    # RAM-tier the duplicate-read stubbing: on a big window (≥128K chars ≈
    # 36K tokens, i.e. 48 GB+) there's room to KEEP prior reads so the model
    # retains what it saw (less "re-read because it scrolled away"). Only stub
    # duplicate reads on smaller windows where the space is actually needed.
    # The progress ledger covers the small-window case where reads must be
    # dropped. Codex similarly re-hydrates rather than aggressively stubbing.
    if ctx_window_chars and ctx_window_chars >= 128_000:
        after_reads = after_writes
    else:
        after_reads = _redact_duplicate_reads(after_writes)
    bytes_reads = _msg_bytes(after_writes) - _msg_bytes(after_reads)
    after_tools = _compact_old_tool_results(after_reads, keep_recent=keep_recent)
    after_budget = _microcompact_for_prompt_budget(after_tools, target_bytes=budget_bytes)
    bytes_tools = _msg_bytes(after_reads) - _msg_bytes(after_tools)
    bytes_budget = _msg_bytes(after_tools) - _msg_bytes(after_budget)
    total_saved = bytes_writes + bytes_reads + bytes_tools + bytes_budget
    if total_saved > 0:
        try:
            from ..server_manager import _lifecycle_log
            _lifecycle_log(
                "redaction",
                msgs=len(messages),
                bytes_before=before,
                bytes_after=_msg_bytes(after_budget),
                saved_writes=bytes_writes,
                saved_reads=bytes_reads,
                saved_tools=bytes_tools,
                saved_budget=bytes_budget,
                saved_total=total_saved,
                pct_saved=int(total_saved * 100 / max(1, before)),
            )
        except Exception:
            pass
    return after_budget


def _microcompact_for_prompt_budget(messages: list[dict], *, target_bytes: int = 36_000) -> list[dict]:
    """Deterministically summarize older protocol-safe history.

    The per-tool aging passes shrink payloads, but long turns can still
    accumulate dozens of small assistant/tool pairs plus recent write args.
    Keep a compact task ledger and recent messages instead of every
    intermediate tool transcript. This pass keeps the system message and a
    protocol-safe recent suffix, replacing the older middle with a factual
    summary of user requests, files touched, commands, and errors. It
    never mutates the source session history.
    """
    if _msg_bytes(messages) <= target_bytes or len(messages) <= 14:
        return messages

    system_count = 1 if messages and messages[0].get("role") == "system" else 0
    boundary = max(system_count, len(messages) - 12)
    while boundary > system_count:
        if messages[boundary].get("role") == "tool":
            boundary -= 1
            continue
        prev = messages[boundary - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            boundary -= 1
            continue
        break
    if boundary <= system_count:
        return messages

    old = messages[system_count:boundary]
    recent = messages[boundary:]
    summary = _compact_history_summary(old)
    candidate = [
        *messages[:system_count],
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "Continuing with the summarized prior work."},
        *recent,
    ]
    if _msg_bytes(candidate) >= _msg_bytes(messages):
        return messages
    return candidate


def _compact_history_summary(messages: list[dict]) -> str:
    files: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    users: list[str] = []
    actions: list[str] = []

    def _add_unique(items: list[str], value: str, limit: int) -> None:
        value = value.strip()
        if value and value not in items:
            items.append(value[:180])
            del items[:-limit]

    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content") or "")
        if role == "user" and content:
            _add_unique(users, content.replace("\n", " "), 4)
        if role == "tool" and content:
            if (
                content.startswith("Error:")
                or content.startswith("REJECTED:")
                or content.startswith("[exit code ")
                or "Traceback " in content
                or "old_string not found" in content
            ):
                _add_unique(errors, _semantic_tool_summary(content).replace("\n", " "), 6)
        if role != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "").strip()
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            path = args.get("path") or args.get("file_path")
            if isinstance(path, str) and path:
                _add_unique(files, path, 12)
            if name == "bash":
                cmd = str(args.get("command") or "")
                _add_unique(commands, cmd.replace("\n", " "), 8)
            if name:
                actions.append(f"{name}({_summarize_args(args)})")
                del actions[:-10]

    lines = ["Earlier context summarized to control prompt size:"]
    if users:
        lines.append("Recent user intent: " + " | ".join(users[-3:]))
    if files:
        lines.append("Files touched/read: " + ", ".join(files[-10:]))
    if commands:
        lines.append("Commands run: " + "; ".join(commands[-5:]))
    if errors:
        lines.append("Errors/fixes to preserve: " + " | ".join(errors[-4:]))
    if actions:
        lines.append("Recent prior tool actions: " + "; ".join(actions[-8:]))
    lines.append("If exact file content is needed, call read_file on the relevant path.")
    return "\n".join(lines)

def build_progress_ledger(
    changed_files: list[str],
    bash_history: list[tuple[str, str]],
    files_read: list[str],
    budget_chars: int,
) -> str:
    """A compact, always-current 'what I've already done this task' ledger.

    Ported from Codex's design (handoff summary + tool-state awareness + an
    explicit 'build on this, don't duplicate work' instruction). Built from the
    loop's DURABLE state (changed_files / bash_history / files_read) — NOT the
    message history — so it survives compaction: on a small-RAM machine the raw
    tool results get compacted to placeholders, but this ledger persists, which
    is exactly what stops the model re-reading files and "starting from scratch."

    `budget_chars` is window-scaled (see model_config.progress_ledger_budget_chars):
    compact on a 16 GB Mac, richer on a 128 GB Mac. Returns "" when nothing has
    happened yet (keeps the prompt prefix stable on the first round).
    """
    if not (changed_files or bash_history or files_read):
        return ""

    def _uniq(seq: list[str]) -> list[str]:
        return list(dict.fromkeys(s for s in seq if s))

    lines = [
        "## Work already done this task — build on it. Do NOT re-read a file or "
        "re-run a command listed below unless you have CHANGED it since; you "
        "already have the result.",
    ]
    rd = _uniq(files_read)
    if rd:
        lines.append("Files read: " + ", ".join(rd[-12:]))
    cf = _uniq(changed_files)
    if cf:
        lines.append("Files created/edited: " + ", ".join(cf[-12:]))
    if bash_history:
        cmds: list[str] = []
        for cmd, res in bash_history[-8:]:
            r = str(res)
            bad = (
                r.startswith("[exit code ") or r.startswith("Error:")
                or r.startswith("REJECTED:") or "Traceback " in r
            )
            cmds.append(f"{'x' if bad else 'ok'} {str(cmd).strip().splitlines()[0][:70]}"
                        if str(cmd).strip() else "")
        cmds = [c for c in cmds if c]
        if cmds:
            lines.append("Commands run: " + " | ".join(cmds))
    text = "\n".join(lines)
    if budget_chars and len(text) > budget_chars:
        text = text[: max(0, budget_chars - 2)].rstrip() + " …"
    return text


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: chars / 4."""
    total = 0
    for m in messages:
        total += len(str(m.get("content", "")))
        for tc in m.get("tool_calls", []):
            total += len(str(tc.get("function", {}).get("arguments", "")))
    return total // 4

def _compact_messages(messages: list[dict], out: "OutputManager") -> list[dict]:
    """Summarize old messages, keep recent context.

    The split must never orphan a `tool` message (server returns 400 if a tool
    response has no preceding assistant/tool_calls in the message list). Walk
    the boundary back until both halves are self-contained.
    """
    if len(messages) <= 12:
        return messages

    system = [m for m in messages[:2] if m.get("role") == "system"]
    sys_len = len(system)

    boundary = max(sys_len, len(messages) - 8)
    # Walk back until: messages[boundary] is not "tool" AND
    # messages[boundary-1] is not an assistant with tool_calls.
    while boundary > sys_len:
        if messages[boundary].get("role") == "tool":
            boundary -= 1
            continue
        prev = messages[boundary - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            boundary -= 1
            continue
        break

    if boundary <= sys_len:
        # Whole tail is one tool-using sequence — nothing safe to compact
        return messages

    recent = messages[boundary:]
    old = messages[sys_len:boundary]
    if not old:
        return messages

    # Build summary from old messages
    parts = []
    files_modified = set()
    commands_run = []

    for m in old:
        role = m.get("role", "")
        content = str(m.get("content", ""))

        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                if name in ("write_file", "append_file", "edit_file"):
                    files_modified.add(args.get("path", "?"))
                elif name == "bash":
                    commands_run.append(args.get("command", "")[:60])
                parts.append(f"Called {name}({_summarize_args(args)})")

        elif role == "user" and not content.startswith("Previous"):
            parts.append(f"User: {content[:100]}")

    summary_lines = ["Previous conversation summary:"]
    if files_modified:
        summary_lines.append(f"Files modified: {', '.join(files_modified)}")
    if commands_run:
        summary_lines.append(f"Commands run: {'; '.join(commands_run[:5])}")
    summary_lines.extend(parts[-8:])

    out.print_info("Context compacted — older messages summarized.")

    return [
        *system,
        {"role": "user", "content": "\n".join(summary_lines)},
        {"role": "assistant", "content": "Got it. Continuing from where we left off."},
        *recent,
    ]

def _summarize_args(args: dict) -> str:
    """Short summary of tool args for display."""
    if "path" in args:
        return args["path"]
    if "pattern" in args:
        return args["pattern"]
    if "command" in args:
        return args["command"][:60]
    if "query" in args:
        return args["query"][:40]
    return str(args)[:60]
