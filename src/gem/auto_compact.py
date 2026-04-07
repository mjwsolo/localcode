"""Auto-compact — summarize conversation when approaching context limit.

When token usage hits 95% of the context window, automatically:
1. Summarize the conversation so far (rule-based, no LLM)
2. Replace old messages with the summary
3. Continue seamlessly

This prevents silent degradation on long sessions with small local models.
"""
from __future__ import annotations


COMPACT_THRESHOLD = 0.90  # trigger at 90% of context budget


def should_compact(messages: list[dict], max_chars: int) -> bool:
    """Check if conversation needs compacting."""
    total = sum(len(m.get("content", "")) for m in messages)
    return total > max_chars * COMPACT_THRESHOLD


def _build_summary(old: list[dict]) -> str:
    summary_parts = []
    files_mentioned = set()
    tools_used = set()
    key_actions = []
    open_questions = []
    latest_user_goal = ""

    for msg in old:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "user":
            first_line = content.strip().split("\n")[0][:140]
            if first_line:
                latest_user_goal = first_line
                summary_parts.append(f"User asked: {first_line}")

        elif role == "assistant":
            for line in content.splitlines():
                line_lower = line.lower().strip()
                if any(w in line_lower for w in ("created", "wrote", "edited", "fixed",
                                                  "updated", "deleted", "installed", "verified",
                                                  "refined", "reviewed")):
                    key_actions.append(line.strip()[:120])
                if "todo" in line_lower or "next" in line_lower or "remaining" in line_lower:
                    open_questions.append(line.strip()[:120])
                import re
                file_match = re.findall(r'(\w+\.(?:py|js|ts|json|md|html|css|go|rs|sh|toml|yaml|yml))', line)
                files_mentioned.update(file_match)

        elif role == "tool":
            tool_name = msg.get("tool_name", "")
            if tool_name:
                tools_used.add(tool_name)
            result = str(msg.get("tool_result", ""))[:160]
            if result:
                key_actions.append(f"Tool {tool_name or 'unknown'}: {result}")

    summary_lines = ["[Conversation summary]"]
    if latest_user_goal:
        summary_lines.append(f"Current objective: {latest_user_goal}")
    if summary_parts:
        summary_lines.append("Topics discussed:")
        for s in summary_parts[-8:]:
            summary_lines.append(f"  - {s}")
    if key_actions:
        summary_lines.append("Work already completed:")
        for a in key_actions[-10:]:
            summary_lines.append(f"  - {a}")
    if open_questions:
        summary_lines.append("Open issues / pending follow-ups:")
        for q in open_questions[-6:]:
            summary_lines.append(f"  - {q}")
    if files_mentioned:
        summary_lines.append(f"Files involved: {', '.join(sorted(files_mentioned)[:20])}")
    if tools_used:
        summary_lines.append(f"Tools used: {', '.join(sorted(tools_used))}")
    return "\n".join(summary_lines)


def auto_compact(messages: list[dict], max_chars: int,
                 keep_recent: int = 4) -> tuple[list[dict], str]:
    """Compact conversation history, keeping recent messages.

    Returns (compacted_messages, summary_text).

    Strategy:
    1. Keep the system prompt (first message if role=system)
    2. Summarize old messages into one message
    3. Keep the last `keep_recent` messages verbatim
    """
    if len(messages) <= keep_recent + 1:
        return messages, ""

    # Separate system prompt
    system_msg = None
    conversation = messages
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        conversation = messages[1:]

    if len(conversation) <= keep_recent:
        return messages, ""

    # Split into old (to summarize) and recent (to keep)
    old = conversation[:-keep_recent]
    recent = conversation[-keep_recent:]

    summary_text = _build_summary(old)

    # Build compacted messages
    compacted = []
    if system_msg:
        compacted.append(system_msg)
    compacted.append({"role": "user", "content": summary_text})
    compacted.append({"role": "assistant", "content": "Understood. I have the context from our earlier conversation. How can I continue helping?"})
    compacted.extend(recent)

    return compacted, summary_text


def compact_if_needed(messages: list[dict], max_chars: int,
                      keep_recent: int = 4) -> tuple[list[dict], bool]:
    """Compact only if needed. Returns (messages, was_compacted)."""
    if should_compact(messages, max_chars):
        compacted = messages
        current_keep = keep_recent
        while should_compact(compacted, max_chars) and current_keep >= 1:
            compacted, _ = auto_compact(compacted, max_chars, current_keep)
            current_keep -= 1
        return compacted, compacted != messages
    return messages, False
