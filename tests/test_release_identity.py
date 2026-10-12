"""Publishing identity and missing-tag recovery against real temporary Git repos."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release_identity.py"
WORKFLOW = ROOT / ".github/workflows/publish.yml"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Release tests")
    git(root, "config", "user.email", "tests@example.invalid")
    (root / "src/localcode").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (root / "src/localcode/__init__.py").write_text('if True:\n    __version__ = "1.2.3"\n')
    git(root, "add", ".")
    git(root, "commit", "-qm", "Fixture")
    return root


def identity(repo, ref="refs/heads/main"):
    return subprocess.run([os.sys.executable, str(SCRIPT), "--ref", ref, "--sha", git(repo, "rev-parse", "HEAD")],
                          cwd=repo, capture_output=True, text=True)


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/tags/v1.2.3"])
def test_resolves_same_identity_on_manual_tag_and_release_paths(repo, ref):
    result = identity(repo, ref)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "version=1.2.3\ntag=v1.2.3\n"


def test_accepts_existing_annotated_tag_on_exact_commit(repo):
    git(repo, "tag", "-a", "v1.2.3", "-m", "Release")
    assert identity(repo).returncode == 0


def test_rejects_existing_tag_on_another_commit_before_publication(repo):
    git(repo, "tag", "v1.2.3")
    git(repo, "commit", "--allow-empty", "-qm", "New commit without version bump")
    result = identity(repo)
    assert result.returncode == 1
    assert "different commit" in result.stderr


def test_rejects_wrong_tag_and_fallback_version(repo):
    assert identity(repo, "refs/tags/v1.2.4").returncode == 1
    (repo / "src/localcode/__init__.py").write_text('if True:\n    __version__ = "1.2.2"\n')
    assert identity(repo).returncode == 1


def ensure_tag_script():
    text = WORKFLOW.read_text()
    body = text.split("      - name: Ensure matching tag exists\n", 1)[1].split("      - name:", 1)[0]
    return "\n".join(line[10:] for line in body.split("        run: |\n", 1)[1].splitlines())


def test_workflow_creates_missing_tag_on_exact_commit_and_preserves_it(repo, tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git(repo, "remote", "add", "origin", str(remote))
    sha = git(repo, "rev-parse", "HEAD")
    env = {**os.environ, "RELEASE_TAG": "v1.2.3", "GITHUB_SHA": sha}
    for _ in range(2):
        result = subprocess.run(["bash", "-e", "-c", ensure_tag_script()], cwd=repo, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert git(repo, "rev-parse", "v1.2.3^{commit}") == sha
    assert git(remote, "rev-parse", "v1.2.3^{commit}") == sha
    git(repo, "commit", "--allow-empty", "-qm", "Later commit")
    result = subprocess.run(["bash", "-e", "-c", ensure_tag_script()], cwd=repo,
                            env={**env, "GITHUB_SHA": git(repo, "rev-parse", "HEAD")}, capture_output=True, text=True)
    assert result.returncode != 0
    assert git(remote, "rev-parse", "v1.2.3^{commit}") == sha


def test_every_publish_path_requires_verified_github_release():
    text = WORKFLOW.read_text()
    job = text.split("  github-release:\n", 1)[1].split("  verify-published:\n", 1)[0]
    assert "needs: [test, publish]" in job
    assert "    if:" not in job
    assert "needs.test.outputs.tag" in job
    assert '--draft=false' in job
    assert 'expected <= actual' in job
    assert "Require changelog notes before publishing" in text
