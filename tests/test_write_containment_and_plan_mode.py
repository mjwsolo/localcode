"""Two security regressions, both verified live before the fix landed.

1. Write containment. `ToolContext.resolve_path` computed `repo / raw` and
   returned it unmodified, and `Path.__truediv__` DISCARDS the left operand
   when the right one is absolute — so `write_file(path="/Users/v/.zshrc")`
   wrote exactly there. `../../..` was never normalised, and a symlink
   committed inside the repo pointing outside was a third escape with a
   fully repo-relative argument. Writes are now contained; READS are
   deliberately not, and must keep working unchanged.

2. Plan mode. plans.py promises the user that plan mode permits exactly one
   write (the plan file) and forbids edits and destructive bash, but the
   `app.plan_mode` flag was read by nothing outside the plan_mode tool
   itself. It is now enforced in `_execute_tool_result`.
"""
import types
from pathlib import Path

import pytest

from localcode.agent.helpers import (
    _execute_tool_result,
    _is_blocked_write_path,
    _needs_confirmation,
    _plan_mode_block,
    _safety_hard_block,
)
from localcode.paths import PathContainmentError, contain_write_path, is_within
from localcode.tools import dispatch_result
from localcode.tools.base import ToolContext


def _app(root: Path, **kw):
    app = types.SimpleNamespace(
        repo_root=root,
        plan_mode=False,
        plan_slug=None,
        hooks=None,
        _autonomy=None,
        _session_allow=set(),
    )
    for k, v in kw.items():
        setattr(app, k, v)
    return app


def _ctx(root: Path, **kw) -> ToolContext:
    return ToolContext(app=_app(root, **kw), out=None)  # type: ignore[arg-type]


# ── 1. containment helper ────────────────────────────────────────────


def test_contain_write_path_accepts_inside_and_rejects_outside(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    assert contain_write_path(root / "src" / "a.py", root) == (root / "src" / "a.py").resolve()
    with pytest.raises(PathContainmentError):
        contain_write_path(tmp_path / "elsewhere" / "a.py", root)


def test_is_within_follows_symlinks(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = root / "link.txt"
    link.symlink_to(outside)
    assert is_within(root / "real.txt", root)
    assert not is_within(link, root)


# ── 1. write tools are contained ─────────────────────────────────────


def test_absolute_path_no_longer_escapes(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    ctx = _ctx(root)
    victim = tmp_path / "victim.txt"
    with pytest.raises(PathContainmentError):
        ctx.resolve_write_path(str(victim))
    assert not victim.exists()


def test_dotdot_traversal_rejected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    ctx = _ctx(root)
    with pytest.raises(PathContainmentError):
        ctx.resolve_write_path("../../etc/hosts")


def test_symlink_out_of_repo_rejected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("original")
    (root / "escape.txt").symlink_to(outside)
    ctx = _ctx(root)
    with pytest.raises(PathContainmentError):
        ctx.resolve_write_path("escape.txt")


def test_inside_repo_writes_still_resolve(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    ctx = _ctx(root)
    assert ctx.resolve_write_path("src/main.py") == (root / "src" / "main.py").resolve()
    assert ctx.resolve_write_path(str(root / "src" / "main.py")) == (root / "src" / "main.py").resolve()


def test_healing_still_works_and_is_contained(tmp_path):
    # The fuzzy repo-prefix heal (small quants mangle long absolute paths)
    # must keep working; containment is applied to the HEALED result.
    root = tmp_path / "localcode-evals" / "20260713-024502" / "tempconv"
    root.mkdir(parents=True)
    ctx = _ctx(root)
    corrupted = str(tmp_path / "localcode-evals" / "2026-0713-0245-02" / "tempconv" / "pyproject.toml")
    assert ctx.resolve_write_path(corrupted) == (root / "pyproject.toml").resolve()


def test_user_approved_path_is_an_escape_hatch(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    target = tmp_path / "outside" / "notes.md"
    target.parent.mkdir()
    ctx = _ctx(root, _approved_write_paths={str(target)})
    assert ctx.resolve_write_path(str(target)) == target.resolve()


def test_reads_are_not_contained(tmp_path):
    # Reads intentionally keep full filesystem access — do not regress this.
    root = tmp_path / "proj"
    root.mkdir()
    ctx = _ctx(root)
    other = str(tmp_path / "elsewhere" / "notes.txt")
    assert str(ctx.resolve_path(other)) == other


def test_write_file_dispatch_rejects_escape(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    victim = tmp_path / "victim.txt"
    result = dispatch_result(
        "write_file", _ctx(root), {"path": str(victim), "content": "pwned"}
    )
    assert result.ok is False
    assert "REJECTED" in result.text
    assert result.facts.get("path_escape") is True
    assert not victim.exists()


@pytest.mark.parametrize("tool,args", [
    ("append_file", {"content": "x"}),
    ("edit_diff", {"diff": "x"}),
    ("edit_file", {"old_string": "a", "new_string": "b"}),
    ("multi_edit", {"edits": [{"old_string": "a", "new_string": "b"}]}),
])
def test_every_write_tool_rejects_escape(tmp_path, tool, args):
    root = tmp_path / "proj"
    root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("a")
    result = dispatch_result(tool, _ctx(root), {"path": str(victim), **args})
    assert result.ok is False
    assert result.facts.get("path_escape") is True
    assert victim.read_text() == "a"


# ── 1b. extended hard-block list ─────────────────────────────────────


@pytest.mark.parametrize("path", [
    "~/.zshrc",
    "~/.bashrc",
    "/Users/v/.bash_profile",
    "~/.profile",
    "~/Library/LaunchAgents/com.evil.plist",
    "/Users/v/proj/.git/hooks/post-checkout",
    "~/.localcode/mcp.json",
    "~/.localcode/config.toml",
    "~/.config/systemd/user/evil.service",
])
def test_persistence_targets_are_hard_blocked(path):
    assert _is_blocked_write_path(path)
    assert _safety_hard_block("write_file", {"path": path}) is not None


@pytest.mark.parametrize("path", [
    "src/hooks/use_thing.ts",        # a project `hooks/` dir is fine
    "scripts/git-hooks/pre-commit",  # hook SOURCE templates are fine
    "profile.py",
    "app/profile/page.tsx",
    "docs/launchagents.md",
])
def test_project_files_still_not_blocked(path):
    assert not _is_blocked_write_path(path)


def test_localcode_workspace_subdirs_still_writable():
    assert not _is_blocked_write_path("~/.localcode/plans/quiet-dawn.md")
    assert not _is_blocked_write_path("~/.localcode/notebook/s1/scratch.md")


# ── 2. plan mode is enforced ─────────────────────────────────────────


def test_plan_mode_off_blocks_nothing(tmp_path):
    app = _app(tmp_path)
    assert _plan_mode_block(app, "write_file", {"path": "a.py"}) is None
    assert _plan_mode_block(app, "bash", {"command": "rm -rf build"}) is None


def test_plan_mode_blocks_file_writes(tmp_path):
    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    for tool in ("write_file", "append_file", "edit_file", "multi_edit", "edit_diff"):
        reason = _plan_mode_block(app, tool, {"path": "src/main.py"})
        assert reason is not None and "plan mode" in reason


def test_plan_mode_allows_only_the_plan_file(tmp_path, monkeypatch):
    from localcode import plans

    plan_file = tmp_path / "plans" / "quiet-dawn.md"
    plan_file.parent.mkdir(parents=True)
    monkeypatch.setattr(plans, "plan_path", lambda slug: plan_file)
    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    assert _plan_mode_block(app, "write_file", {"path": str(plan_file)}) is None
    # …but only via write_file, and only for that exact path.
    assert _plan_mode_block(app, "append_file", {"path": str(plan_file)}) is not None
    assert _plan_mode_block(app, "write_file", {"path": str(tmp_path / "other.md")}) is not None


@pytest.mark.parametrize("cmd", [
    "rm -rf build",
    "git push origin main",
    "npm run deploy",
    "echo evil > ~/.zshrc",
    "cat README.md && rm -rf /tmp/x",
    "python3 -c 'x' `whoami`",
])
def test_plan_mode_blocks_mutating_shell(tmp_path, cmd):
    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    assert _plan_mode_block(app, "bash", {"command": cmd}) is not None


@pytest.mark.parametrize("cmd", [
    "ls -la src",
    "grep -rn plan_mode src | head -20",
    "git status",
    "git log --oneline -5",
    "cat pyproject.toml",
])
def test_plan_mode_allows_read_only_exploration(tmp_path, cmd):
    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    assert _plan_mode_block(app, "bash", {"command": cmd}) is None


def test_plan_mode_blocks_background_process(tmp_path):
    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    assert _plan_mode_block(app, "background_process", {"command": "npm run dev"}) is not None


def test_execute_tool_result_enforces_plan_mode(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "main.py"
    app = _app(root, plan_mode=True, plan_slug="quiet-dawn")
    result = _execute_tool_result(
        app, "write_file", {"path": "main.py", "content": "print(1)"}, None  # type: ignore[arg-type]
    )
    assert result.ok is False
    assert result.facts.get("plan_mode_blocked") is True
    assert not target.exists()


def test_exit_plan_mode_requires_confirmation(tmp_path):
    from localcode.autonomy import AutonomyLevel

    app = _app(tmp_path, plan_mode=True, plan_slug="quiet-dawn")
    app._autonomy = AutonomyLevel.AUTO_EDIT
    assert _needs_confirmation("exit_plan_mode", {}, app) is True
    # Not in plan mode → nothing to confirm.
    app.plan_mode = False
    assert _needs_confirmation("exit_plan_mode", {}, app) is False
    # FULL_AUTO still bypasses every prompt.
    app.plan_mode = True
    app._autonomy = AutonomyLevel.FULL_AUTO
    assert _needs_confirmation("exit_plan_mode", {}, app) is False
