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


# Rough 4:1 char:token ratio. Good enough for thresholding — we don't
# need tokenizer-exact counts for a "should I compact?" decision.
_CHARS_PER_TOKEN = 4

# Reserve space in the context window for: (a) new user input,
# (b) new assistant generation, (c) tool results on the next round.
# Compacting must leave this much breathing room below context_window.
RESERVE_TOKENS_DEFAULT = 4096

# Keep this many tokens of most-recent history verbatim after a compact.
# Picked so the last few tool results + the user's latest ask + the
# in-progress assistant turn all survive intact.
KEEP_RECENT_TOKENS_DEFAULT = 6144

# Trigger compaction when estimated prompt exceeds this fraction of
# (context_window - reserve). 0.70 leaves enough margin for compaction
# itself to not blow the window.
COMPACT_THRESHOLD_FRACTION = 0.70


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


def should_compact(
    messages: list[dict[str, Any]],
    context_window: int,
    reserve_tokens: int = RESERVE_TOKENS_DEFAULT,
) -> bool:
    """Return True when the prompt is about to crowd out new generation.

    `context_window` is the number of tokens llama-server was launched
    with (`-c`). `reserve_tokens` is headroom we promise to keep free.
    """
    est = estimate_tokens(messages)
    available = context_window - reserve_tokens
    if available <= 0:
        return True
    return est > int(available * COMPACT_THRESHOLD_FRACTION)


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


def compact(
    messages: list[dict[str, Any]],
    runtime: Any,
    context_window: int,
    keep_recent_tokens: int = KEEP_RECENT_TOKENS_DEFAULT,
) -> list[dict[str, Any]]:
    """Return a compacted copy of `messages`.

    Workflow:
      1. Keep every role="system" message up front (system prompt + any
         prior compaction memos).
      2. Split the remaining (user/assistant/tool) messages into
         "to-summarize" and "keep-verbatim" slices by token count.
      3. Call `runtime.chat_once` with the to-summarize slice + the
         SUMMARIZATION_PROMPT to produce a structured summary.
      4. Return: [original system msgs] + [new system summary] +
         [keep-verbatim slice].

    If there's nothing old enough to summarize, return `messages`
    unchanged. If the summary call fails, return `messages` unchanged
    (fail-safe — worse to corrupt the conversation).
    """
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]

    to_summarize, keep = _split_at_keep_recent(others, keep_recent_tokens)
    if not to_summarize:
        return messages

    # Build the summarization request. The model sees prior turns as
    # normal context and then our final user-role instruction.
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
        return messages

    msg = (result or {}).get("message", {}) or {}
    summary_text = (msg.get("content") or "").strip()
    if not summary_text:
        return messages

    # Prepend a clear header so the downstream model knows this is a
    # compressed earlier-session note, not a direct instruction.
    memo = (
        "Prior conversation summary (older turns were compacted to save "
        "context; the original messages are no longer in this prompt):\n\n"
        f"{summary_text}"
    )
    compacted: list[dict[str, Any]] = list(sys_msgs)
    compacted.append({"role": "system", "content": memo})
    compacted.extend(keep)
    return compacted
