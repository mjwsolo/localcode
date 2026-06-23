"""Shared types for tool modules.

Each tool lives in its own file under src/localcode/tools/ and exports two
symbols:

  SCHEMA:   dict — the OpenAI-compatible function schema handed to the
            model. See TOOL_SCHEMAS usage in src/localcode/agent.py.

  execute:  callable (ctx: ToolContext, args: dict) -> str — invoked
            when the model picks this tool. Returns the result string
            that's fed back as the `tool` message.

Per-tool files + co-located schema+exec is the minimal-agent / terminal coding tools Agent
SDK convention. Makes adding a tool a one-file change, keeps description
next to the code that implements it, and enables per-tool unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..app import LocalCodeApp
    from ..output import OutputManager


@dataclass
class ToolContext:
    """Everything a tool might need to do its job.

    Keep this deliberately small — if a tool can be done with just
    `repo` (Path) and `args` (dict), prefer that. Reach for `app` or
    `out` only when you need the full agent state or the TUI output
    manager.
    """
    app: "LocalCodeApp"
    out: "OutputManager"

    @property
    def repo(self) -> Path:
        return self.app.repo_root


@dataclass
class ToolResult:
    """Internal typed tool result.

    Tools may continue returning plain strings. The dispatcher normalizes
    both forms to this type so the agent loop can reason over facts while
    preserving the exact text sent back to the model/UI.
    """
    text: str
    ok: bool = True
    facts: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


class ToolModule(Protocol):
    """Structural type every tool module satisfies."""
    SCHEMA: dict

    @staticmethod
    def execute(ctx: ToolContext, args: dict) -> str: ...  # noqa: E704

    @staticmethod
    def is_concurrency_safe(args: dict) -> bool: ...  # noqa: E704
