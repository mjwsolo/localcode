"""Per-message-kind renderers — agent `src/components/messages/` pattern.

Each render method takes a ChatLog instance plus its kind-specific
arguments and writes content to the log. ChatLog acts as the dispatcher
(via `_dispatch_gap` for spacing + the per-kind method for content).

Extracted modules so far:
    diff.py          — file-edit diff with line-number gutter

To extract (follow-up — each is its own self-contained PR):
    user.py          — _render_user
    assistant.py     — _render_assistant + the streaming partial-line path
    thinking.py      — _render_thinking
    tool.py          — _render_tool, _render_tool_done, _render_tool_result
    info.py          — _render_info, _render_error, _render_approval
    turn_summary.py  — _render_turn_summary

Full extraction is a 1-day refactor — left for a quiet day. The diff
extraction here proves the pattern works and gives the most-complex
renderer its own testable home.
"""
