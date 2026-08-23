"""A bare mistyped/removed slash command must error, not go to the model.

The QA pass caught `/undo` (a removed command) being sent to the model as a
normal message - the model then apologised for having no undo command. A
leading `/` is still sent to the model on purpose so a pasted path
(`/Users/you/repo`, `/tmp`) works; the fix only catches a lone `/word` that is
neither a known command nor a real filesystem path.
"""

from __future__ import annotations

from localcode.tui.screens.chat import (
    _is_known_command,
    _looks_like_mistyped_command,
)


def test_removed_or_typod_commands_are_flagged():
    for t in ("/undo", "/celar", "/xyz", "/help"):
        assert _looks_like_mistyped_command(t), t
        assert not _is_known_command(t), t


def test_known_commands_are_not_flagged():
    for t in ("/clear", "/model qwen", "/mcp", "/exit"):
        assert not _looks_like_mistyped_command(t), t


def test_pasted_paths_and_prose_pass_through_to_the_model():
    # Multi-segment paths, real single-segment paths, and any text with a space
    # are messages, not command attempts.
    for t in ("/Users/you/repo", "/tmp", "/etc", "/path to thing",
              "fix /undo please", "hello world"):
        assert not _looks_like_mistyped_command(t), t
