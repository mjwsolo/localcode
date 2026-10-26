"""scripts/release.sh (the one-command flow), release_promote.py and .github/workflows/release.yml."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import release_promote  # noqa: E402

GIT = ["git", "-c", "user.name=Release tests", "-c", "user.email=tests@example.invalid"]


def git(cwd, *args):
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    work = tmp_path / "work"
    (work / "src/localcode").mkdir(parents=True)
    (work / "pyproject.toml").write_text('[project]\nname = "localcode"\nversion = "0.5.40"\n')
    (work / "src/localcode/__init__.py").write_text('try:\n    pass\nexcept Exception:\n    __version__ = "0.5.40"\n')
    (work / "CHANGELOG.md").write_text("# Changelog\n\n## 0.5.41 - unreleased\n\n- New.\n\n## 0.5.40 - 2026-10-26\n\n- Old.\n")
    shutil.copytree(SCRIPTS, work / "scripts", ignore=shutil.ignore_patterns("__pycache__", "node_modules"))
    git(work, "init", "-q", "-b", "main")
    git(work, "add", ".")
    git(work, "commit", "-qm", "base")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(origin)], check=True)
    git(work, "remote", "add", "origin", str(origin))
    git(work, "fetch", "-q", "origin")
    git(work, "branch", "-q", "--set-upstream-to=origin/main", "main")
    return work


def release(repo, tmp_path, *args, pgrep_exit=1):
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    (stubs / "pgrep").write_text(f"#!/bin/sh\nexit {pgrep_exit}\n")
    (stubs / "pgrep").chmod(0o755)
    env = {**os.environ, "RELEASE_ROOT": str(repo), "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPTS / "release.sh"), *args], env=env, capture_output=True, text=True)


def test_dry_run_prints_every_step_in_order_and_changes_nothing(repo, tmp_path):
    head = git(repo, "rev-parse", "HEAD")
    r = release(repo, tmp_path, "0.5.41", "--model", "m", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    steps = ["checks done (0 problem(s))", "tag -a v0.5.41rc1", "refs/tags/v0.5.41rc1", "== 3.", "pypi.org/pypi/localcode/0.5.41rc1/json",
             "localcode==0.5.41rc1", "localcode doctor", "ui_e2e.sh --installed", "verify_models.py --installed --strict --publish-status",
             "release_eval.sh --model m --candidate-venv", "gh workflow run release.yml --ref main -f candidate=0.5.41rc1",
             "post-publish"]
    assert "context=release eval" in out and "state=success" in out
    positions = [out.index(s) for s in steps]
    assert positions == sorted(positions), steps
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "tag", "-l") == ""
    assert 'version = "0.5.40"' in (repo / "pyproject.toml").read_text()
    assert git(repo, "ls-remote", "--tags", "origin") == ""


@pytest.mark.parametrize("what", ["dirty", "branch", "behind", "llama", "changelog", "tagged"])
def test_refuses_to_start(repo, tmp_path, what):
    pg = 1
    if what == "dirty":
        (repo / "pyproject.toml").write_text((repo / "pyproject.toml").read_text() + "\n")
    elif what == "branch":
        git(repo, "switch", "-q", "-c", "feature/x")
    elif what == "behind":
        (repo / "x.txt").write_text("x")
        git(repo, "add", "x.txt")
        git(repo, "commit", "-qm", "unpushed")
    elif what == "llama":
        pg = 0
    elif what == "changelog":
        (repo / "CHANGELOG.md").write_text("# Changelog\n")
        git(repo, "commit", "-qam", "no notes")
        git(repo, "push", "-q", "origin", "main")
    elif what == "tagged":
        git(repo, "tag", "v0.5.41")
    r = release(repo, tmp_path, "0.5.41", "--model", "m", pgrep_exit=pg)
    assert r.returncode == 2 and "refusing to start" in r.stderr, r.stdout + r.stderr
    assert git(repo, "ls-remote", "--tags", "origin") == ""
    assert "WOULD REFUSE" in release(repo, tmp_path, "0.5.41", "--model", "m", "--dry-run", pgrep_exit=pg).stdout


def test_needs_a_model_and_a_final_version(repo, tmp_path):
    assert release(repo, tmp_path, "0.5.41").returncode == 2
    assert release(repo, tmp_path, "0.5.41rc1", "--model", "m").returncode == 2


def test_promote_renames_the_heading_and_sets_both_versions(repo):
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n- New.\n\n## 0.5.40 - x\n\n- Old.\n")
    assert release_promote.main(["0.5.41", "--root", str(repo), "--date", "2026-11-01"]) == 0
    text = (repo / "CHANGELOG.md").read_text()
    assert "## 0.5.41 - 2026-11-01\n\n- New." in text and "Unreleased" not in text
    assert 'version = "0.5.41"' in (repo / "pyproject.toml").read_text()
    assert '__version__ = "0.5.41"' in (repo / "src/localcode/__init__.py").read_text()
    assert release_promote.promote_changelog("## 0.5.42rc3\n- a\n## 0.5.41 - d\n", "0.5.42", "D").startswith("## 0.5.42 - D")
    assert release_promote.promote_changelog("## 0.5.42 - d\n- a\n", "0.5.42", "D") == "## 0.5.42 - d\n- a\n"
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## 0.5.40 - x\n")
    assert release_promote.main(["0.5.42", "--root", str(repo)]) == 1
    assert 'version = "0.5.41"' in (repo / "pyproject.toml").read_text(), "nothing is half-edited on failure"


def test_release_workflow_is_owner_only_and_never_runs_on_pull_requests():
    path = next(ROOT.glob(".g*/workflows/release.yml"))
    text = path.read_text()
    wf = yaml.safe_load(text)
    on = wf.get("on", wf.get(True))
    assert set(on) == {"push", "workflow_dispatch"} and on["push"]["tags"] == ["v*rc*"]
    assert "pull_request" not in on and "pull_request_target" not in text
    guard = "github.repository_owner == github.actor && !github.event.repository.fork"
    assert "gpu-checks" not in wf["jobs"] and "self-hosted" not in text
    assert all(guard in wf["jobs"][name]["if"] for name in ("prepare", "promote"))
    promote = wf["jobs"]["promote"]
    assert set(promote["needs"]) == {"prepare", "installed-checks"} and "workflow_dispatch" in promote["if"]
    assert wf["jobs"]["installed-checks"]["runs-on"] == "macos-15"
    assert "local model gate" in text and "release eval" in text and "scripts/ui_e2e.sh --installed" in text
    assert wf["concurrency"]["group"] and wf["concurrency"]["cancel-in-progress"] is False
    for other in path.parent.glob("*.yml"):
        if other.name != "release.yml":
            assert all("self-hosted" not in str(j.get("runs-on")) for j in yaml.safe_load(other.read_text())["jobs"].values()), other.name
