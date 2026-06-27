"""Tiered autonomy modes — control how much the agent can do without asking.

Three levels:
    SUGGEST   — read-only, everything needs approval. Safe for exploration.
    AUTO_EDIT — file writes auto-approved, bash/installs need approval.
    FULL_AUTO — everything auto-approved. Fast but trust the model.

The mode configures the PermissionManager behavior.
Can be set via:
    - Config: [safety] autonomy = "auto_edit"
    - CLI flag: localcode --autonomy full_auto
    - Slash command: /autonomy full_auto
    - Environment: LOCALCODE_AUTONOMY=suggest
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AutonomyLevel(Enum):
    SUGGEST = "suggest"
    AUTO_EDIT = "auto_edit"
    FULL_AUTO = "full_auto"


@dataclass
class AutonomyPolicy:
    """What each autonomy level allows."""
    level: AutonomyLevel
    auto_approve_reads: bool
    auto_approve_writes: bool
    auto_approve_bash: bool
    auto_approve_installs: bool
    auto_approve_git: bool
    description: str


POLICIES: dict[AutonomyLevel, AutonomyPolicy] = {
    AutonomyLevel.SUGGEST: AutonomyPolicy(
        level=AutonomyLevel.SUGGEST,
        auto_approve_reads=True,
        auto_approve_writes=False,
        auto_approve_bash=False,
        auto_approve_installs=False,
        auto_approve_git=False,
        description="Read-only. All writes and commands need approval.",
    ),
    AutonomyLevel.AUTO_EDIT: AutonomyPolicy(
        level=AutonomyLevel.AUTO_EDIT,
        auto_approve_reads=True,
        auto_approve_writes=True,
        auto_approve_bash=False,
        auto_approve_installs=False,
        auto_approve_git=False,
        description="File edits auto-approved. Bash and installs need approval.",
    ),
    AutonomyLevel.FULL_AUTO: AutonomyPolicy(
        level=AutonomyLevel.FULL_AUTO,
        auto_approve_reads=True,
        auto_approve_writes=True,
        auto_approve_bash=True,
        auto_approve_installs=True,
        auto_approve_git=True,
        description="Everything auto-approved. SafetyLayer still blocks dangerous commands.",
    ),
}


def get_policy(level: AutonomyLevel | str) -> AutonomyPolicy:
    """Get the policy for a given autonomy level."""
    if isinstance(level, str):
        try:
            level = AutonomyLevel(level.lower())
        except ValueError:
            level = AutonomyLevel.AUTO_EDIT  # default
    return POLICIES[level]


def apply_autonomy_to_permissions(permissions, policy: AutonomyPolicy) -> None:
    """Configure a PermissionManager based on the autonomy policy.

    Args:
        permissions: PermissionManager instance from permissions_v2.py
        policy: The autonomy policy to apply
    """
    if policy.auto_approve_writes:
        permissions._session_approved.add("write_file")
        permissions._session_approved.add("edit_file")
        permissions._session_approved.add("multi_edit")

    if policy.auto_approve_bash:
        permissions._session_approved.add("bash")

    if policy.auto_approve_installs:
        permissions._session_approved.add("pip_install")
        permissions._session_approved.add("npm_install")

    if policy.auto_approve_git:
        permissions._session_approved.add("git_commit")

    if policy.level == AutonomyLevel.FULL_AUTO:
        permissions.approve_all()
