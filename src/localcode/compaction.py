"""Conversation compaction for long sessions.

When the running message list approaches the model's context window, we
ask the model itself to produce a structured summary of the oldest turns
and replace them with a single system-role memo. Subsequent turns then
see a short history ("Prior conversation: ...") + the most recent N
tokens verbatim.

Why this beats raw truncation:
  * Goals and decisions from early turns survive
  * File paths and error messages stay exact
  * Recent turns are preserved so the model can finish what it was doing

Design borrowed from minimal-agent/coding-agent (Mario Zechner) — the structured
sections (Goal / Progress / Key Decisions / Files / Next Steps) are the
same format; we simplified for a single-model local setup and reused our
existing runtime for the summary call (no extra model required).

This is complementary to the SSM context-checkpoint optimization on the
llama-server side. Checkpoints make multi-turn re-evaluation fast;
compaction prevents the conversation from ever exceeding the context
window (at which point checkpointing can't help anymore).
"""
from __future__ import annotations

from typing import Any

# Central per-model × per-Mac config — the single source of truth for the
# compaction tiers/thresholds below. Re-exported here under their historical
# names so call sites and tests that import them from `compaction` are
# unchanged. The window→keep-recent scaling helper also delegates.
from .model_config import (
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,
    COMPACT_THRESHOLD_FRACTION,
    KEEP_RECENT_TOKENS_DEFAULT,
    KEEP_RECENT_TOKENS_MAX,
    LLM_SUMMARY_MIN_RAM_GB,
    RESERVE_TOKENS_DEFAULT,
    keep_recent_for_window as _model_config_keep_recent_for_window,
)


def _keep_recent_for_window(context_window: int) -> int:
    """How many recent tokens to keep verbatim, scaled to the window.

    Small windows (16 GB Mac, ~64K) keep the 6K floor — there's no room
    for more. Big windows (128 GB, 256K) keep up to ~48K verbatim so a
    long session preserves far more raw recent history instead of being
    crushed to the same tiny tail as a small machine.

    Delegates to `model_config.keep_recent_for_window` (the central config).
    """
    return _model_config_keep_recent_for_window(context_window)


_SUMMARIZATION_SYSTEM = (
    "You are a context summarization assistant. You read a conversation "
    "between a user and an AI coding assistant and produce a concise "
    "structured summary. You NEVER continue the conversation; you NEVER "
    "answer questions in it. Output ONLY the structured summary."
)

_SUMMARIZATION_USER = """The messages above are the conversation to summarize. Produce a structured checkpoint another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? One or two lines.]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, or "(none)"]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Files Touched
- `<absolute or repo-relative path>`: read | created | modified | deleted

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Exact error messages, function names, API signatures worth preserving]
- [Or "(none)"]

Keep each section tight. Preserve file paths, function names, error messages VERBATIM."""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a message list.

    Good enough for deciding whether to compact. Treats non-string
    content (images, structured blocks) conservatively by counting
    their serialized length.
    """
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        else:
            # structured content (lists, dicts) — conservative serialization
            total_chars += len(str(content))
        # role tag + tool_call_id + JSON overhead
        total_chars += 20
    return total_chars // _CHARS_PER_TOKEN


def protocol_errors(messages: list[dict[str, Any]]) -> list[str]:
    """Return tool-call/tool-result trajectory violations in transcript order."""
    pending: set[str] = set()
    errors: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id") or "")
                if not call_id:
                    errors.append(f"assistant[{index}] has a tool call without an id")
                else:
                    pending.add(call_id)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending:
                errors.append(f"tool[{index}] has no preceding call: {call_id}")
            else:
                pending.remove(call_id)
    errors.extend(f"tool call has no result: {call_id}" for call_id in sorted(pending))
    return errors


def normalize_tool_protocol(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill legacy/malformed missing tool IDs deterministically in place."""
    awaiting: list[str] = []
    serial = 0
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id") or "")
                if not call_id:
                    serial += 1
                    call_id = f"localcode_call_{serial}"
                    call["id"] = call_id
                awaiting.append(call_id)
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id and awaiting:
                call_id = awaiting[0]
                message["tool_call_id"] = call_id
            if call_id in awaiting:
                awaiting.remove(call_id)
    return messages


def _compact_fraction() -> float:
    """The fraction of the usable window at which auto-compaction fires.

    Default COMPACT_THRESHOLD_FRACTION (0.70). Overridable via
    LOCALCODE_COMPACT_PCT (1-100, like Claude Code's CLAUDE_AUTOCOMPACT_PCT_
    OVERRIDE) so a user with a big machine can let context grow closer to the
    full window before summarising, or compact earlier on a small one. Always
    relative to the machine's real RAM-scaled context window — never a fixed
    token count.
    """
    import os
    raw = os.environ.get("LOCALCODE_COMPACT_PCT")
    if raw is None:
        return COMPACT_THRESHOLD_FRACTION
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        return COMPACT_THRESHOLD_FRACTION
    # Accept either a percent (1-100) or a fraction (0-1); clamp to a sane band.
    frac = pct / 100.0 if pct > 1.0 else pct
    return min(max(frac, 0.30), 0.95)


def should_compact(
    messages: list[dict[str, Any]],
    context_window: int,
    reserve_tokens: int = RESERVE_TOKENS_DEFAULT,
) -> bool:
    """Return True when the prompt is about to crowd out new generation.

    `context_window` is the number of tokens llama-server was launched
    with (`-c`) — the machine's real RAM-scaled window. `reserve_tokens` is
    headroom we promise to keep free. The trigger is always a fraction of THAT
    window (see `_compact_fraction`), so a 256K-capable machine compacts near
    256K and a 64K one near 64K.
    """
    est = estimate_tokens(messages)
    available = context_window - reserve_tokens
    if available <= 0:
        return True
    return est > int(available * _compact_fraction())


def _split_at_keep_recent(
    non_system: list[dict[str, Any]],
    keep_recent_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split non-system messages into (to_summarize, keep_verbatim).

    Walks backward from the tail until keep_recent_tokens is exceeded;
    everything before the cutoff becomes "old" and will be summarized.

    Guards against splitting a tool_call/tool_result pair: if the cutoff
    lands on a tool-role message, extend backward until we hit a
    user/assistant boundary. Breaking a tool call without its response
    confuses the next prompt.
    """
    running_chars = 0
    cutoff = len(non_system)
    for i in range(len(non_system) - 1, -1, -1):
        m = non_system[i]
        content = m.get("content", "")
        chars = len(content) if isinstance(content, str) else len(str(content))
        running_chars += chars + 20
        if running_chars > keep_recent_tokens * _CHARS_PER_TOKEN:
            cutoff = i + 1
            break
        cutoff = i  # track newest valid cutoff while accumulating

    # Don't cut mid tool-call pair. Walk forward until we land on a
    # user or assistant message, which means any tool calls before
    # the cutoff have their results also before it.
    while cutoff < len(non_system) and non_system[cutoff].get("role") == "tool":
        cutoff += 1
    return non_system[:cutoff], non_system[cutoff:]


def _llm_summary(runtime: Any, to_summarize: list[dict[str, Any]]) -> str:
    """Ask the resident model for a structured summary. Returns "" on any
    failure so the caller can fall back to the deterministic summary."""
    summary_req = (
        [{"role": "system", "content": _SUMMARIZATION_SYSTEM}]
        + to_summarize
        + [{"role": "user", "content": _SUMMARIZATION_USER}]
    )
    try:
        result = runtime.chat_once(
            summary_req,
            tools=None,
            think=False,
            num_predict=1500,
        )
    except Exception:
        return ""
    msg = (result or {}).get("message", {}) or {}
    return (msg.get("content") or "").strip()


def _deterministic_summary(to_summarize: list[dict[str, Any]]) -> str:
    """Build a structured summary with zero model calls.

    Reuses the agent's `_compact_history_summary` (files touched, commands
    run, errors to preserve, recent user intent, recent tool actions) so the
    small-RAM path and the per-round aging path produce consistent memos.
    """
    try:
        from .agent.context import _compact_history_summary
        text = _compact_history_summary(to_summarize)
    except Exception:
        text = ""
    return (text or "").strip()


def compact(
    messages: list[dict[str, Any]],
    runtime: Any,
    context_window: int,
    keep_recent_tokens: int | None = None,
    ram_gb: int | None = None,
) -> list[dict[str, Any]]:
    """Return a compacted copy of `messages`.

    Workflow:
      1. Keep every role="system" message up front (system prompt + any
         prior compaction memos).
      2. Split the remaining (user/assistant/tool) messages into
         "to-summarize" and "keep-verbatim" slices by token count.
      3. Produce a structured summary of the to-summarize slice. On a
         capable machine (`ram_gb` ≥ `LLM_SUMMARY_MIN_RAM_GB`) we ask the
         resident model for a rich summary; on a small machine — or if the
         model summary call fails/returns nothing — we fall back to an
         instant deterministic summary. This keeps a 16 GB Mac responsive
         (no mid-task generation stall, no weak-model summary) while a
         128 GB Mac spends the tokens for a better one.
      4. Return: [original system msgs] + [new system summary] +
         [keep-verbatim slice].

    `keep_recent_tokens` defaults to a window-scaled value (`None` →
    `_keep_recent_for_window`) so big windows preserve more recent history.

    If there's nothing old enough to summarize, return `messages`
    unchanged. If summarization produces nothing, return `messages`
    unchanged (fail-safe — worse to corrupt the conversation).
    """
    if keep_recent_tokens is None:
        keep_recent_tokens = _keep_recent_for_window(context_window)

    sys_msgs = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]

    to_summarize, keep = _split_at_keep_recent(others, keep_recent_tokens)
    if not to_summarize:
        return messages

    # RAM-tiered: spend a model generation only where it's worth it.
    use_llm = ram_gb is None or ram_gb >= LLM_SUMMARY_MIN_RAM_GB
    summary_text = _llm_summary(runtime, to_summarize) if use_llm else ""
    if not summary_text:
        # Small machine, or the LLM summary failed/was empty — deterministic.
        summary_text = _deterministic_summary(to_summarize)
    if not summary_text:
        return messages

    # Prepend a clear header so the downstream model knows this is a
    # compressed earlier-session note, not a direct instruction.
    # Codex-style resume framing: tell the model it has prior work + tool state
    # and must BUILD ON it, not redo it (the anti-duplication prefix that stops
    # "start from scratch" after a compaction).
    memo = (
        "Earlier turns of THIS task were compacted to save context. The summary "
        "below is the work already done — build on it and do NOT re-read files or "
        "redo steps it already covers:\n\n"
        f"{summary_text}"
    )
    compacted: list[dict[str, Any]] = list(sys_msgs)
    compacted.append({"role": "system", "content": memo})
    compacted.extend(keep)
    normalize_tool_protocol(compacted)
    if protocol_errors(compacted):
        return messages
    return compacted
