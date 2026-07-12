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


def _spill_tool_output(result: str, tool_name: str) -> str | None:
    """Write an oversized tool result to a file and return its path.

    Truncation alone makes a small model re-run the command "to see the rest".
    Instead we keep the full output on disk and point the model at it (grep/
    read the exact part). Best-effort: None on error → caller falls back to
    plain truncation. Content-hashed filenames; old spills reaped after 7 days.
    """
    try:
        import hashlib
        import time as _time
        from ..paths import global_state_dir
        spill_dir = global_state_dir() / "tool_output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        # Reap spills older than 7 days so this never grows unbounded.
        try:
            cutoff = _time.time() - 7 * 86400
            for old in spill_dir.glob("*.txt"):
                if old.stat().st_mtime < cutoff:
                    old.unlink(missing_ok=True)
        except Exception:
            pass
        digest = hashlib.sha1(result.encode("utf-8", "replace")).hexdigest()[:12]
        path = spill_dir / f"{tool_name}_{digest}.txt"
        if not path.exists():
            path.write_text(result, encoding="utf-8", errors="replace")
        return str(path)
    except Exception:
        return None


# How much of the context window (in chars) a single tool result may occupy,
# per tool. The static RESULT_LIMITS values act as the FLOOR, so small machines
# (64K window ≈ 224K chars → these fractions fall below the floor) are
# byte-identical to before, while a big machine (256K ≈ 900K chars) lets the
# model keep far more of the output it has room for instead of a fixed 20–50K
# slice. grep matches are dense/repetitive so get the smallest share; read_file
# is the primary code-ingestion path so gets the most.
_RESULT_LIMIT_WINDOW_FRACTION: dict[str, float] = {
    "grep": 0.03,
    "bash": 0.05,
    "read_file": 0.08,
    "default": 0.08,
}


def _dynamic_result_limit(tool_name: str, ctx_tokens: int) -> int:
    """Per-tool truncation budget (chars), scaled to the real context window.

    Returns the static RESULT_LIMITS value as a floor; on a large window the
    limit grows to a fraction of the window so the model isn't starved of output
    it has room for. web_search and any tool without a fraction stay at the
    fixed floor (web results don't benefit from window-scaling).
    """
    floor = RESULT_LIMITS.get(tool_name, RESULT_LIMITS["default"])
    frac = _RESULT_LIMIT_WINDOW_FRACTION.get(tool_name)
    if not ctx_tokens or frac is None:
        return floor
    # CHARS_PER_TOKEN=4 is the internal estimate; use 3.5 to match loop.py's
    # tokens→chars conversion and stay conservative.
    return max(floor, int(ctx_tokens * 3.5 * frac))


def _truncate_result(result: str, tool_name: str, ctx_tokens: int = 0) -> str:
    """Truncate a tool result to its per-tool size limit with a
    strategy tuned to what each tool's output actually looks like.

    `ctx_tokens` (the model's real context window) scales the limit up on big
    machines; 0 falls back to the static floor (small machines unchanged).

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
    limit = _dynamic_result_limit(tool_name, ctx_tokens)
    if len(result) <= limit:
        return result

    dropped = len(result) - limit
    # Spill the FULL output to disk and point the model at it, so it fetches
    # the exact part it needs instead of re-running the command to "see the
    # rest". Appended to every strategy's hint below.
    _spill = _spill_tool_output(result, tool_name)
    spill_hint = (
        f" Full output saved to {_spill} — read_file (with offset) or grep it "
        f"to see the rest; do NOT re-run the command."
        if _spill else ""
    )

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
            f"{'Call read_file with offset=' + str(last_numbered) + ' to continue reading from there.' if last_numbered else 'Call read_file with a larger offset to continue.'}{spill_hint}]"
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
            f"Narrow with a more specific pattern or `include=*.py` to see them.{spill_hint}]"
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
                f"\n\n[... {dropped} chars of bash output compressed.{spill_hint} ...]\n\n"
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
        marker = f"\n\n[... {dropped} chars of middle output dropped.{spill_hint} ...]\n\n"
        return head + marker + tail

    # Default — middle-drop.
    half = limit // 2
    return (
        result[:half]
        + f"\n\n[...{dropped} chars truncated.{spill_hint} ...]\n\n"
        + result[-half:]
    )


# COMPACT_KEEP_RECENT_TOOL_RESULTS / COMPACT_MIN_CONTENT_CHARS moved
# to constants.py during the T0.1 split; re-exported at the top of
# this module for back-compat.

# Only these tools' RESULTS are safe to age. They are REPLAYABLE
# OBSERVATIONS: their output is a snapshot of external state (the filesystem,
# a search, a shell command) that the model can regenerate on demand by
# re-running the exact same call. Aging them is lossless-in-effect — the model
# can always get them back. This mirrors claude-code's `COMPACTABLE_TOOLS` set
# (microCompact.ts) and opencode's prune (which protects non-replayable tool
# output). Deliberately EXCLUDED: write_file / edit_file / multi_edit /
# append_file. Their result is the DIFF of what the model just wrote — the only
# record in history of the change it made. Aging it away is what made the model
# "forget" what it wrote after a build failure and rewrite whole files from
# scratch (churn). Anything we can't attribute to a replayable tool is also left
# intact (conservative default).
_REPLAYABLE_TOOLS: frozenset[str] = frozenset({
    "read_file", "grep", "glob", "list_files", "bash",
    "web_search", "web_fetch",
})


def _tool_names_by_call_id(messages: list[dict]) -> dict[str, str]:
    """Map tool_call_id -> tool name by scanning assistant tool_calls, so a
    tool-role result (which only carries `tool_call_id`) can be attributed to
    the tool that produced it."""
    names: dict[str, str] = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tc_id = tc.get("id")
            name = ((tc.get("function") or {}).get("name") or "").strip()
            if tc_id and name:
                names[tc_id] = name
    return names


def _result_tool_name(m: dict, id_to_name: dict[str, str]) -> str:
    """Best-effort tool name for a tool-role result: explicit `name` field if
    present (some providers set it), else via the tool_call_id map."""
    explicit = (m.get("name") or "").strip()
    if explicit:
        return explicit
    return id_to_name.get(m.get("tool_call_id"), "")


def _compact_old_tool_results(
    messages: list[dict], keep_recent: int = COMPACT_KEEP_RECENT_TOOL_RESULTS
) -> list[dict]:
    """Return a copy of `messages` with older REPLAYABLE-OBSERVATION tool
    results summarized.

    Gated on `Feature.TOOL_RESULT_AGING` — when disabled the caller gets
    the input list back unchanged, which is what eval uses to A/B
    "how much does aging actually save per turn?"

    We keep the last `keep_recent` tool results verbatim (the model usually
    only needs the recent ones to decide the next step) and replace earlier
    ones with a compact "[summarized ...]" placeholder — but ONLY for tools in
    `_REPLAYABLE_TOOLS`. write_file/edit_file/multi_edit results (the diffs of
    what the model wrote) are NEVER aged, so the model doesn't lose sight of
    its own changes. `keep_recent` scales with the context window (more on a big
    machine — see `_prepare_model_messages`), floored at 1 so aging never clears
    the entire working set. User/assistant/system messages pass through
    unchanged, and `tool_call_id` is preserved so the chat protocol still
    reconciles ids correctly.
    """
    from ..features import Feature, is_enabled
    if not is_enabled(Feature.TOOL_RESULT_AGING):
        return messages
    # Floor at 1 so we never age away the entire working set of tool output.
    keep_recent = max(1, keep_recent)
    id_to_name = _tool_names_by_call_id(messages)
    # Only REPLAYABLE-observation results are age-eligible; keep the last
    # `keep_recent` of THOSE verbatim (counting non-replayable results toward
    # the recency window would let a burst of writes push a still-needed read
    # out of the kept set).
    ageable_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
        and _result_tool_name(m, id_to_name) in _REPLAYABLE_TOOLS
    ]
    if len(ageable_idxs) <= keep_recent:
        return messages

    cutoff_idx = ageable_idxs[-keep_recent]
    out: list[dict] = []
    for i, m in enumerate(messages):
        if i >= cutoff_idx or m.get("role") != "tool":
            out.append(m)
            continue
        if _result_tool_name(m, id_to_name) not in _REPLAYABLE_TOOLS:
            out.append(m)  # non-replayable (write/edit diff, etc.) — preserve
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
        # Self-conditioning (arXiv:2509.09677): echoing an OLD error's text
        # pushes the model to repeat the mistake. Recent errors stay intact
        # (kept above by keep_recent) so it can fix the immediate problem;
        # older ones drop to a neutral note (the ledger still records the fact).
        return "[an earlier tool call errored and was handled — details dropped]"
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
    """Pass-through: the model's OWN write/edit bodies are NEVER stripped.

    History note (this used to strip them, and that was a bug)
    ----------------------------------------------------------
    This pass previously replaced older write_file `content` (and edit_file /
    multi_edit anchors) in the assistant tool_calls with a "[REDACTED — file is
    on disk, call read_file to reload]" stub, keeping only the last
    `REDACT_KEEP_RECENT_WRITES` verbatim. The intent was to cut context bloat.

    The real cost (observed in logs): after a build failure the model could no
    longer SEE what it had written a few rounds earlier — the body was a stub —
    so instead of a targeted fix it rewrote the whole file from scratch, over
    and over (churn). What the model wrote is exactly what it needs to reason
    about its own change; that record must stay in history.

    The reference agents agree: claude-code's microCompact clears old tool
    RESULTS but never the write BODIES in assistant messages; codex/opencode
    likewise summarize whole turns wholesale only when over budget rather than
    surgically deleting the code the model wrote. So this function no longer
    strips anything. Context is bounded instead by:
      * `_compact_old_tool_results` — ages only REPLAYABLE observation results
        (read/grep/glob/list_files/bash), never write/edit diffs;
      * `_redact_duplicate_reads` — collapses re-reads of the same path;
      * `_microcompact_for_prompt_budget` / `_compact_messages` — summarize old
        turns wholesale when the prompt genuinely approaches the window budget.

    Kept as a named no-op (rather than deleted) so callers, telemetry, and the
    A/B feature flag keep working. Still gated on `Feature.WRITE_ARG_REDACTION`.
    """
    from ..features import Feature, is_enabled
    if not is_enabled(Feature.WRITE_ARG_REDACTION):
        return messages
    # Never strip write/edit bodies — see docstring. Older-write redaction is
    # intentionally disabled; the empty target set makes the rest of this
    # function a structural no-op while preserving its shape for back-compat.
    REDACT_TOOLS: set[str] = set()

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
                _raw = (tc.get("function") or {}).get("arguments") or "{}"
                args = _raw if isinstance(_raw, dict) else json.loads(_raw)
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
    window. Capacity and replay latency are separate budgets: large-RAM Macs
    can retain richer recent evidence, but hybrid/recurrent local models may
    still re-process the entire prompt each round. Durable ledgers preserve
    completed work while this hot transcript remains bounded.

    `ctx_window_chars` ≈ num_ctx tokens × ~3.5 chars/token. None/unknown →
    the legacy fixed behaviour (back-compat, safe on tiny windows).
    """
    if not ctx_window_chars or ctx_window_chars <= 0:
        return 36_000, COMPACT_KEEP_RECENT_TOOL_RESULTS
    # A large KV window is capacity, not a latency target. Live 128 GB / Qwen
    # traces showed TTFT rising from ~2 s to 44 s as one build replayed ~28K
    # prompt tokens. Cap the hot transcript while the durable progress ledger
    # and filesystem-state block preserve what was completed.
    budget = max(36_000, min(48_000, int(ctx_window_chars * 0.20)))
    keep = max(COMPACT_KEEP_RECENT_TOOL_RESULTS, min(8, ctx_window_chars // 40_000))
    return budget, keep


def latency_budgeted_hot_replay(
    ctx_window_chars: int | None,
    observed_ttft_ms: int | None,
    *,
    target_ttft_ms: int = 8_000,
) -> tuple[int, int]:
    """Adapt hot replay to measured local prefill latency, not KV capacity."""
    budget, keep = _window_aware_compaction(ctx_window_chars)
    if not observed_ttft_ms or observed_ttft_ms <= target_ttft_ms:
        return budget, keep
    ratio = max(0.35, target_ttft_ms / observed_ttft_ms)
    return max(18_000, int(budget * ratio)), max(2, int(keep * ratio))


def _prepare_model_messages(
    messages: list[dict], ctx_window_chars: int | None = None,
    observed_ttft_ms: int | None = None,
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
    budget_bytes, keep_recent = latency_budgeted_hot_replay(ctx_window_chars, observed_ttft_ms)
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
    best = messages
    # Tighten the protocol-safe suffix until the serialized prompt fits. A
    # fixed 12-message tail can itself contain several complete source files.
    for tail_size in (12, 10, 8, 6, 4):
        boundary = max(system_count, len(messages) - tail_size)
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
            continue
        old = messages[system_count:boundary]
        recent = messages[boundary:]
        summary = _compact_history_summary(old)
        candidate = [
            *messages[:system_count],
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "Continuing with the summarized prior work."},
            *recent,
        ]
        if _msg_bytes(candidate) < _msg_bytes(best):
            best = candidate
        if _msg_bytes(candidate) <= target_bytes:
            return candidate
    return best


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
                _raw = fn.get("arguments") or "{}"
                args = _raw if isinstance(_raw, dict) else json.loads(_raw)
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
        "## A log of YOUR OWN tool calls so far this turn — NOT the user's work "
        "and NOT pre-existing files on disk (leftover files you did not create are "
        "not your progress or your task). Don't re-read/re-run anything below "
        "unless you changed it; keep making NEW progress toward the user's goal.",
    ]
    # Files created/edited is the most important line (it's "where I am" in a
    # multi-file build) — list it FIRST and show ALL of them, not just the last
    # 12. Capping at 12 dropped the earliest files once a build exceeded 12,
    # so the model re-checked what existed and felt like it restarted. Paths are
    # short; the whole list is cheap and is the anchor that prevents forgetting.
    cf = _uniq(changed_files)
    # The created/edited list is PROTECTED — never truncated by the budget.
    # It's the anchor that stops the model forgetting what it built; losing any
    # of it is what caused the "restarting" feeling. Everything ELSE (files
    # read, commands) shares the remaining budget and can be trimmed.
    protected = []
    if cf:
        protected.append(f"Files YOU already created/edited this task ({len(cf)}) — "
                         "do NOT recreate or re-check these, build on them: " + ", ".join(cf))
    optional = []
    rd = _uniq(files_read)
    if rd:
        optional.append("Files read: " + ", ".join(rd[-12:]))
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
            optional.append("Commands run: " + " | ".join(cmds))

    head = "\n".join(lines + protected)
    if not optional:
        return head
    extra = "\n".join(optional)
    if budget_chars and len(head) + 1 + len(extra) > budget_chars:
        room = max(0, budget_chars - len(head) - 3)
        extra = extra[:room].rstrip() + " …" if room > 0 else ""
    return head + ("\n" + extra if extra else "")


_PROJECT_MARKERS = ("package.json", "pyproject.toml", ".git", "go.mod", "Cargo.toml", "tsconfig.json")


def _project_root_for(common: str, dirs: list[str]) -> str | None:
    """The project directory to scan: the nearest marker at-or-below `common`,
    else the deepest single changed-file dir. Bounds the walk to one project."""
    import os
    # Walk DOWN from each changed dir toward `common` looking for a marker,
    # picking the shallowest marker that is still within `common`.
    candidates = []
    for d in dirs:
        cur = d
        while cur and cur.startswith(common):
            if any(os.path.exists(os.path.join(cur, m)) for m in _PROJECT_MARKERS):
                candidates.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    if candidates:
        return min(candidates, key=len)  # shallowest marker dir
    # No marker: only trust it if all changed files share ONE dir (a real
    # single-project build); otherwise decline (common ancestor may be broad).
    return dirs[0] if len(set(dirs)) == 1 else None


def build_filesystem_state(changed_files: list[str], max_files: int = 80) -> str:
    """Ground-truth 'what's actually on disk' for the project being built.

    The #1 fix for the model re-checking which files exist and re-creating
    them (pi/codex/opencode all converge on this): don't rely on the model's
    recollection — reconcile against the FILESYSTEM in code, each round. We
    walk the tree rooted at the project the model is building (the common
    parent of the files it has changed) and list the real source files. Because
    the truth comes from the OS, it survives compaction/restart and the model
    can't hallucinate it away. Returns "" when nothing has been built yet.
    """
    import os
    roots = [os.path.dirname(f) for f in changed_files if f]
    if not roots:
        return ""
    try:
        common = os.path.commonpath([os.path.abspath(os.path.dirname(f))
                                     for f in changed_files if f])
    except Exception:
        common = os.path.abspath(roots[0])
    # SCOPE GUARD: only ever walk a real PROJECT dir, never a broad ancestor
    # like ~ or a top-level workspace. Anchor to the nearest project marker
    # (package.json / pyproject.toml / .git / go.mod / Cargo.toml) at-or-below
    # the common path; if none, use the deepest single changed-file dir. And
    # NEVER walk the home dir or anything shallower — bail out entirely.
    root = _project_root_for(common, [os.path.abspath(os.path.dirname(f))
                                      for f in changed_files if f])
    home = os.path.expanduser("~")
    if not root or not os.path.isdir(root):
        return ""
    if os.path.abspath(root) in (home, os.path.dirname(home), os.path.sep):
        return ""
    if len(os.path.abspath(root).rstrip("/").split("/")) < 4:  # too shallow → unsafe to walk
        return ""
    _SKIP = {".git", "node_modules", "dist", "build", ".venv", "__pycache__", ".next"}
    _SRC = (".ts", ".tsx", ".js", ".jsx", ".py", ".css", ".scss", ".html", ".json",
            ".md", ".go", ".rs", ".java", ".rb", ".php", ".vue", ".svelte", ".toml", ".yaml", ".yml")
    found: list[str] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP]
        for fn in files:
            if fn.endswith(_SRC):
                rel = os.path.relpath(os.path.join(base, fn), root)
                found.append(rel)
                if len(found) >= max_files:
                    break
        if len(found) >= max_files:
            break
    if not found:
        return ""
    found.sort()
    return (
        f"## Files that ALREADY EXIST on disk in this project ({os.path.basename(root)}/) "
        f"— ground truth, {len(found)} file(s). Do NOT list the directory to check; "
        "do NOT recreate these. Edit an existing file or create a NEW one the task needs:\n"
        + ", ".join(found)
    )


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

    # Build an anchored "Work State" summary from old messages. This is the
    # structure opencode + codex converge on (Objective / Completed / Blocked /
    # Next Move) — a state machine a small local model can follow, not a flat
    # "Called X, Called Y" log. Built DETERMINISTICALLY (no LLM call — matches
    # codex's no-inference compaction, which matters on slow 16GB hardware):
    # completed = files written + commands that succeeded; blocked = commands
    # that failed / errors; objective = the first user request.
    objective = ""
    files_modified: list[str] = []
    ok_cmds: list[str] = []
    failed: list[str] = []

    for m in old:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        if role == "user" and not objective and not content.startswith(("Previous", "SYSTEM:", "##")):
            objective = content[:200]
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    _raw = fn.get("arguments", "{}")
                    args = _raw if isinstance(_raw, dict) else json.loads(_raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    args = {}
                if name in ("write_file", "append_file", "edit_file", "multi_edit"):
                    p = args.get("path") or args.get("file_path")
                    if p and p not in files_modified:
                        files_modified.append(p)
                elif name == "bash":
                    ok_cmds.append(str(args.get("command", ""))[:70])
        elif role == "tool":
            c = content.strip()
            if c.startswith(("Error", "[exit code", "REJECTED")) or "Traceback " in c:
                failed.append(c.splitlines()[0][:100] if c else "")

    completed = []
    if files_modified:
        completed.append("Files created/edited: " + ", ".join(files_modified))
    if ok_cmds:
        completed.append("Commands run: " + " | ".join(ok_cmds[-5:]))
    blocked = [f for f in failed if f][-3:]

    lines = ["## Work state so far (older turns compacted — build on this, don't redo it):"]
    if objective:
        lines.append(f"Objective: {objective}")
    lines.append("Completed:")
    lines.extend(f"  - {c}" for c in completed) if completed else lines.append("  - (none yet)")
    if blocked:
        lines.append("Blocked / errors seen:")
        lines.extend(f"  - {b}" for b in blocked)
    lines.append("Next move: continue the objective from the current file state; "
                 "do not re-read files or re-run commands listed above.")

    out.print_info("Context compacted — older turns summarized to a work-state note.")

    return [
        *system,
        {"role": "user", "content": "\n".join(lines)},
        {"role": "assistant", "content": "Understood — continuing from the current state."},
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
