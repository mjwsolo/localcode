"""Updater regressions: fork ancestry and the boundary around writer credentials."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()


def test_opencode_refuses_a_fork_behind_upstream_before_changing_the_pin(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "dev")
    git(upstream, "config", "user.name", "Test")
    git(upstream, "config", "user.email", "test@example.invalid")
    (upstream / "a").write_text("one")
    git(upstream, "add", "a")
    git(upstream, "commit", "-m", "initial")
    fork = tmp_path / "fork"
    subprocess.run(["git", "clone", str(upstream), str(fork)], check=True, capture_output=True)
    git(fork, "branch", "localcode")
    (upstream / "a").write_text("two")
    git(upstream, "commit", "-am", "new upstream")
    pin = ROOT / "src/localcode/ui/FORK_COMMIT"
    old = pin.read_bytes()
    result = subprocess.run(["bash", str(ROOT / "scripts/bump_opencode.sh")], env={**os.environ, "LOCALCODE_UI_FORK_REPO": str(fork), "LOCALCODE_OPENCODE_UPSTREAM_REPO": str(upstream), "LOCALCODE_UI_BUILD_DIR": str(tmp_path / "work")}, capture_output=True, text=True)
    assert result.returncode == 20
    assert "has not absorbed upstream" in result.stderr
    assert pin.read_bytes() == old


def test_upstream_build_does_not_keep_checkout_credentials():
    workflow = (ROOT / ".github/workflows/upstream-candidate.yml").read_text()
    build = workflow.split("  propose:", 1)[0]
    assert "permissions:\n  contents: read" in build
    assert "persist-credentials: false" in build
    assert "GH_TOKEN:" not in build
    assert "--include='src/localcode/ui/FORK_COMMIT'" in workflow
    assert "--include='.github" not in workflow


def test_no_workflow_runs_on_the_users_mac():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        assert "runs-on: self-hosted" not in path.read_text()
