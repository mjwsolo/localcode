"""Approval-related types.

Originally shipped an `ApprovalQueue` class that rendered a Rich panel and
prompted via prompt_toolkit — replaced by the TUI's `approval_callback`
path (see src/localcode/tui/bridge.py and widgets/approval.py). The queue is
deleted; only the `ApprovalItem` dataclass remains since it's still used
as a type hint in the legacy CLI approval code in app.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApprovalItem:
    label: str
    detail: str
