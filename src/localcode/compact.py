"""Smart context compaction — preserve important info, drop verbose noise.

Unlike crude truncation, this:
1. Keeps all user messages (intent)
2. Summarizes tool results (keep first/last lines, drop middle)
3. Drops verbose thinking from older turns
4. Preserves most recent 4 messages fully
5. Compresses older assistant responses to first paragraph
"""
from __future__ import annotations


def compact_messages(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Compact conversation history to fit within max_chars."""
    total = sum(len(msg.get("content", "")) for msg in messages)
    if total <= max_chars or len(messages) < 8:
        return messages

    # Always keep the last 8 messages in full
    keep_recent = 8
    recent = messages[-keep_recent:]
    older = messages[:-keep_recent]

    if not older:
        return messages

    # Compact older messages
    compacted: list[dict[str, str]] = []
    for msg in older:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # Keep user messages — they carry intent
            if len(content) > 1000:
                content = content[:1000] + "..."
            compacted.append({"role": role, "content": content})

        elif role == "tool":
            # Summarize tool results — keep first 2 + last 2 lines
            lines = content.splitlines()
            if len(lines) > 6:
                summary_lines = lines[:2] + ["  ..."] + lines[-2:]
                content = "\n".join(summary_lines)
            elif len(content) > 400:
                content = content[:400] + "..."
            compacted.append({"role": role, "content": content})

        elif role == "assistant":
            # Compress assistant to first paragraph or 600 chars
            # Drop thinking content entirely
            paragraphs = content.split("\n\n")
            first_para = paragraphs[0] if paragraphs else content
            if len(first_para) > 600:
                first_para = first_para[:600] + "..."
            compacted.append({"role": role, "content": first_para})

        elif role == "system":
            # Keep system messages but cap length
            if len(content) > 1000:
                content = content[:1000] + "..."
            compacted.append({"role": role, "content": content})

        else:
            compacted.append(msg)

    result = compacted + recent

    # If still too long, drop oldest messages one by one
    while sum(len(m.get("content", "")) for m in result) > max_chars and len(result) > keep_recent + 1:
        result.pop(0)

    return result
