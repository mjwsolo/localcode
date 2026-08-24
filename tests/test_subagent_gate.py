"""The `agent` tool's sub-agent loop must use the parent's guarded dispatch.

Regression tests for the sub-agent escape hatch: `tools/agent.py` used to
call each tool's raw executor directly, which bypassed
`agent.helpers._execute_tool_result` (and therefore `_safety_hard_block`,
the pre_tool_use hook, and the destructive-write guards) as well as the
`_needs_confirmation` approval gate. One `agent` call bought up to 12
rounds of unapproved shell + file writes, even in suggest mode.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localcode.tools.agent import _dispatch_guarded
from localcode.tools.base import ToolContext


class _FakeApp:
    def __init__(self, level_name="SUGGEST", repo_root=Path(".")):
        from localcode.autonomy import AutonomyLevel
        self._autonomy = getattr(AutonomyLevel, level_name)
        self._session_allow: set[str] = set()
        self.repo_root = Path(repo_root)


class _FakeOut:
    def __init__(self, verdict=None):
        self._approval_callback = (lambda name, cmd: verdict) if verdict else None

    def _stop_indicator(self):
        pass


def _ctx(app, out):
    return ToolContext(app=app, out=out)


# ── (a) hard block reaches sub-agent bash ──────────────────────────────

def test_subagent_bash_hits_the_safety_hard_block():
    # full_auto: no approval prompt at all — proves the block is the
    # autonomy-independent backstop, not the confirmation gate.
    ctx = _ctx(_FakeApp("FULL_AUTO"), _FakeOut())
    result = _dispatch_guarded(ctx, "bash", {"command": "rm -rf /"})
    assert result.startswith("REJECTED:")
    assert "blocked" in result.lower()


def test_subagent_credential_write_hits_the_safety_hard_block(tmp_path):
    ctx = _ctx(_FakeApp("FULL_AUTO", repo_root=tmp_path), _FakeOut())
    result = _dispatch_guarded(
        ctx, "write_file",
        {"path": str(Path.home() / ".ssh" / "authorized_keys"), "content": "x"},
    )
    assert result.startswith("REJECTED:")
    assert "credential" in result.lower()


# ── (b) approval gate is consulted for sub-agent tool calls ────────────

def test_subagent_tool_call_goes_through_needs_confirmation(monkeypatch):
    seen = []

    import localcode.agent.helpers as helpers

    def _fake_needs(name, args, app=None):
        seen.append((name, args, app))
        return True

    monkeypatch.setattr(helpers, "_needs_confirmation", _fake_needs)
    ctx = _ctx(_FakeApp("SUGGEST"), _FakeOut(verdict="deny"))
    result = _dispatch_guarded(ctx, "bash", {"command": "echo hi"})
    assert result == "Denied by user."
    assert seen and seen[0][0] == "bash"
    assert seen[0][1] == {"command": "echo hi"}


def test_subagent_bash_denied_in_suggest_mode_never_runs(tmp_path):
    marker = tmp_path / "pwned.txt"
    ctx = _ctx(_FakeApp("SUGGEST", repo_root=tmp_path), _FakeOut(verdict="deny"))
    result = _dispatch_guarded(ctx, "bash", {"command": f"touch {marker}"})
    assert result == "Denied by user."
    assert not marker.exists()


def test_subagent_bash_runs_when_user_approves_once(tmp_path):
    marker = tmp_path / "ok.txt"
    ctx = _ctx(_FakeApp("SUGGEST", repo_root=tmp_path), _FakeOut(verdict="once"))
    _dispatch_guarded(ctx, "bash", {"command": f"touch {marker}"})
    assert marker.exists()


def test_subagent_always_verdict_populates_session_allowlist(tmp_path):
    app = _FakeApp("SUGGEST", repo_root=tmp_path)
    ctx = _ctx(app, _FakeOut(verdict="always"))
    _dispatch_guarded(ctx, "bash", {"command": "echo hi"})
    assert "echo" in app._session_allow


def test_subagent_denies_when_no_approval_channel_is_available(monkeypatch):
    # Headless-ish: confirmation is required but there is no callback and no
    # usable terminal. Must NOT silently allow.
    import localcode.agent.helpers as helpers

    monkeypatch.setattr(helpers, "_request_approval_verdict",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no tty")))
    ctx = _ctx(_FakeApp("SUGGEST"), _FakeOut())
    assert _dispatch_guarded(ctx, "bash", {"command": "echo hi"}) == "Denied by user."


def test_readonly_subagent_tool_is_not_prompted(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\n")
    # verdict "deny" would surface if read_file were (wrongly) gated.
    ctx = _ctx(_FakeApp("SUGGEST", repo_root=tmp_path), _FakeOut(verdict="deny"))
    result = _dispatch_guarded(ctx, "read_file", {"path": str(f)})
    assert "hello" in result
