"""Release candidates (0.5.41rc1) flow through the same pipeline as releases, as PyPI pre-releases."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import release_eval  # noqa: E402
import release_identity  # noqa: E402
import release_notes  # noqa: E402

GIT = ["git", "-c", "user.name=Release tests", "-c", "user.email=tests@example.invalid"]



@pytest.fixture(autouse=True)
def _evals_data(tmp_path_factory, monkeypatch):
    # release.sh and release_eval.sh need the (git-ignored) Harness-Bench data.
    data = tmp_path_factory.mktemp("evals-data")
    (data / "harness-bench").mkdir()
    monkeypatch.setenv("LOCALCODE_EVALS_DATA", str(data))

def git(cwd, *args):
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src/localcode").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "localcode"\nversion = "0.5.40"\n')
    (tmp_path / "src/localcode/__init__.py").write_text('try:\n    pass\nexcept Exception:\n    __version__ = "0.5.40"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\n- Something new.\n\n## 0.5.40 - 2026-10-26\n\n- Old.\n")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def candidate(repo, *args):
    return subprocess.run(["bash", str(SCRIPTS / "release_candidate.sh"), *args], env={**os.environ, "RELEASE_CANDIDATE_ROOT": str(repo)},
                          capture_output=True, text=True)


# -- release_candidate.sh ------------------------------------------------------

def test_first_candidate_is_rc1_and_files_are_bumped_but_nothing_is_tagged_or_committed(repo):
    r = candidate(repo, "0.5.41")
    assert r.returncode == 0, r.stderr
    assert 'version = "0.5.41rc1"' in (repo / "pyproject.toml").read_text()
    assert '__version__ = "0.5.41rc1"' in (repo / "src/localcode/__init__.py").read_text()
    assert git(repo, "tag", "-l") == ""
    assert git(repo, "rev-list", "--count", "HEAD") == "1"
    assert "git push origin v0.5.41rc1" in r.stdout and "git tag -a v0.5.41rc1" in r.stdout
    assert "uv tool install --prerelease allow" in r.stdout


def test_next_free_rc_comes_from_existing_tags(repo):
    for tag in ("v0.5.41rc1", "v0.5.41rc2", "v0.5.41rc10", "v0.5.4rc7", "v0.5.410rc9"):
        git(repo, "tag", tag)
    r = candidate(repo, "0.5.41", "--dry-run")
    assert "next candidate: 0.5.41rc11" in r.stdout
    assert 'version = "0.5.40"' in (repo / "pyproject.toml").read_text(), "dry run must not edit files"


def test_refuses_a_version_that_is_already_released_or_malformed(repo):
    git(repo, "tag", "v0.5.41")
    assert candidate(repo, "0.5.41").returncode == 2
    assert candidate(repo, "0.5.42rc1").returncode == 2
    assert candidate(repo, "banana").returncode == 2


def test_refuses_with_uncommitted_tracked_changes(repo):
    (repo / "CHANGELOG.md").write_text("changed\n")
    assert candidate(repo, "0.5.41").returncode == 2
    assert 'version = "0.5.40"' in (repo / "pyproject.toml").read_text()


# -- the workflow's checks accept an rc ----------------------------------------

def test_release_identity_accepts_an_rc_and_its_tag(repo):
    candidate(repo, "0.5.41")
    sha = git(repo, "rev-parse", "HEAD")
    assert release_identity.release_identity(repo, "refs/tags/v0.5.41rc1", sha) == ("0.5.41rc1", "v0.5.41rc1")
    with pytest.raises(ValueError):
        release_identity.release_identity(repo, "refs/tags/v0.5.41", sha)


def test_main_push_that_bumps_to_an_rc_publishes(repo):
    before = git(repo, "rev-parse", "HEAD")
    candidate(repo, "0.5.41")
    out = subprocess.check_output([sys.executable, str(SCRIPTS / "release_intent.py"), "--event", "push", "--ref", "refs/heads/main", "--before", before],
                                  cwd=repo, text=True).strip()
    assert out == "publish=true"


def test_rc_needs_no_changelog_heading(repo):
    text = (repo / "CHANGELOG.md").read_text()
    assert release_notes.notes_for(text, "0.5.41rc1").startswith("Release candidate 0.5.41rc1")
    assert "Something new." in release_notes.notes_for(text, "0.5.41rc1"), "falls back to Unreleased"
    assert "Something new." not in release_notes.notes_for(text, "0.5.40")
    assert release_notes.notes_for(text, "0.5.99") is None, "a final release still needs its own notes"
    bare = "# Changelog\n\n## 0.5.40 - x\n\n- Old.\n"
    assert release_notes.notes_for(bare, "0.5.41rc2").startswith("Release candidate 0.5.41rc2")
    only_base = bare + "\n## 0.5.41 - unreleased\n\n- New.\n"
    assert "- New." in release_notes.notes_for(only_base, "0.5.41rc2")


def test_release_notes_cli_passes_check_for_an_rc_but_not_for_a_final():
    ok = subprocess.run([sys.executable, str(SCRIPTS / "release_notes.py"), "0.9.9rc1", "--check"], capture_output=True, text=True)
    bad = subprocess.run([sys.executable, str(SCRIPTS / "release_notes.py"), "0.9.9", "--check"], capture_output=True, text=True)
    assert ok.returncode == 0 and bad.returncode == 1


def test_pep440_keeps_an_rc_out_of_the_default_install():
    packaging = pytest.importorskip("packaging.version")
    rc, final = packaging.Version("0.5.41rc1"), packaging.Version("0.5.40")
    assert rc.is_prerelease and rc > final and packaging.Version("0.5.41") > rc
    from packaging.specifiers import SpecifierSet
    # What an installer does: with a final release available, an unpinned
    # requirement resolves to the final, never to the candidate. (Asserting on
    # contains() alone breaks across packaging versions.)
    assert list(SpecifierSet(">=0.5.0").filter([final, rc])) == [final], "default resolution skips pre-releases"
    assert SpecifierSet("==0.5.41rc1").contains(rc, prereleases=True), "an exact pin selects it"


def test_publish_workflow_marks_rc_releases_as_prereleases_and_tests_the_installed_package():
    text = next(ROOT.glob(".g*/workflows/publish.yml")).read_text()
    assert '[[ "${tag#v}" =~ [a-zA-Z] ]]' in text and "--prerelease" in text
    assert "ui_e2e.sh --installed" in text and "localcode\" doctor" in text.replace("/bin/localcode", "localcode")
    assert "pip index versions localcode" in text
    shell_test = subprocess.run(["bash", "-c", 'tag=v0.5.41rc1; [[ "${tag#v}" =~ [a-zA-Z] ]] && echo pre'], capture_output=True, text=True)
    assert shell_test.stdout.strip() == "pre"
    shell_test = subprocess.run(["bash", "-c", 'tag=v0.5.41; [[ "${tag#v}" =~ [a-zA-Z] ]] && echo pre; true'], capture_output=True, text=True)
    assert shell_test.stdout.strip() == ""


# -- release_eval --------------------------------------------------------------

def run(passed_scores, wall):
    return {"tasks": {f"t{i}": {"score": s, "passed": s >= 0.8, "wall_s": wall, "error": False} for i, s in enumerate(passed_scores)}}


def test_equal_runs_pass():
    r = release_eval.compare(run([1, 1, 0.5], 100), run([1, 1, 0.5], 100))
    assert r["ok"] and r["score_drop"] == 0 and r["slowdown"] == 0


def test_score_drop_beyond_margin_blocks_but_within_margin_passes():
    prev = run([1, 1, 1, 1], 100)
    assert not release_eval.compare(prev, run([1, 1, 1, 0.6], 100))["ok"]  # mean drops 0.10
    assert release_eval.compare(prev, run([1, 1, 1, 0.9], 100))["ok"]      # mean drops 0.025
    assert release_eval.compare(prev, run([1, 1, 1, 0.6], 100), score_margin=0.2)["ok"]


def test_more_than_25_percent_slower_blocks():
    prev = run([1, 1], 100)
    assert release_eval.compare(prev, run([1, 1], 125))["ok"]
    r = release_eval.compare(prev, run([1, 1], 130))
    assert not r["ok"] and "slower" in r["reasons"][0]
    assert release_eval.compare(run([1, 1], 100), run([1, 1], 60))["ok"], "faster is fine"


def test_regressed_and_fixed_tasks_are_listed():
    prev = {"tasks": {"a": {"score": 1, "passed": True, "wall_s": 1}, "b": {"score": 0, "passed": False, "wall_s": 1}}}
    cand = {"tasks": {"a": {"score": 0, "passed": False, "wall_s": 1}, "b": {"score": 1, "passed": True, "wall_s": 1}}}
    r = release_eval.compare(prev, cand, score_margin=1)
    assert r["regressed"] == ["a"] and r["fixed"] == ["b"]
    assert "passed before, fails now: a" in release_eval.render(r)


def test_different_or_empty_task_sets_cannot_be_compared():
    with pytest.raises(release_eval.Incomparable):
        release_eval.compare(run([1, 1], 1), run([1], 1))
    with pytest.raises(release_eval.Incomparable):
        release_eval.compare({"tasks": {}}, {"tasks": {}})


def test_missing_wall_time_is_not_silently_ignored():
    prev, cand = run([1], 100), run([1], 100)
    cand["tasks"]["t0"]["wall_s"] = None
    r = release_eval.compare(prev, cand)
    assert not r["ok"] and "wall time missing" in r["reasons"][0]


def test_previous_release_skips_candidates_and_newer_versions():
    versions = ["0.5.38", "0.5.39", "0.5.40", "0.5.41rc1", "0.5.41", "0.5.42", "0.5.9", "0.5.40.post1"]
    assert release_eval.previous_release("0.5.41rc2", versions) == "0.5.40"
    assert release_eval.previous_release("0.5.41", versions) == "0.5.40"
    assert release_eval.previous_release("0.5.10", versions) == "0.5.9"
    assert release_eval.previous_release("0.5.9", versions) is None


def test_default_task_set_is_the_six_fixed_ids():
    assert len(release_eval.DEFAULT_TASKS) == 6 and "120-http-cookie-jar-policy-repair" in release_eval.DEFAULT_TASKS


def test_compare_cli_exit_codes(tmp_path):
    def write(name, data):
        d = tmp_path / name
        d.mkdir()
        (d / "run.json").write_text(json.dumps(data))
        return str(d)
    prev, same, worse = write("p", run([1, 1], 100)), write("s", run([1, 1], 100)), write("w", run([0, 0], 100))
    other = write("o", run([1], 100))
    cmd = [sys.executable, str(SCRIPTS / "release_eval.py"), "compare"]
    assert subprocess.run([*cmd, prev, same], capture_output=True).returncode == 0
    assert subprocess.run([*cmd, prev, worse], capture_output=True).returncode == 1
    assert subprocess.run([*cmd, prev, other], capture_output=True).returncode == 2


# -- release_eval.sh ------------------------------------------------------------

def stub_bin(tmp_path, pgrep_exit):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    (d / "pgrep").write_text(f"#!/bin/sh\nexit {pgrep_exit}\n")
    (d / "pgrep").chmod(0o755)
    return d


def eval_script(tmp_path, *args, pgrep_exit=1, path_extra=""):
    bin_dir = stub_bin(tmp_path, pgrep_exit)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{path_extra}{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPTS / "release_eval.sh"), *args], env=env, capture_output=True, text=True)


def test_release_eval_refuses_while_a_llama_server_runs(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    r = eval_script(tmp_path, "--model", "m", "--models-dir", str(models), "--previous", "0.5.40", "--work", str(tmp_path / "w"), pgrep_exit=0)
    assert r.returncode == 2 and "llama-server" in r.stderr
    assert not (tmp_path / "w/logs-candidate").exists()


def test_release_eval_never_creates_a_models_directory(tmp_path):
    missing = tmp_path / "nope"
    r = eval_script(tmp_path, "--model", "m", "--models-dir", str(missing), "--previous", "0.5.40", "--work", str(tmp_path / "w"))
    assert r.returncode == 2 and "never downloads" in r.stderr and not missing.exists()


def test_release_eval_dry_run_prints_both_runs_and_runs_nothing(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    r = eval_script(tmp_path, "--model", "m", "--models-dir", str(models), "--previous", "0.5.40", "--work", str(tmp_path / "w"), "--dry-run")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.count("inspect eval localcode_evals/harnessbench") == 2
    assert "localcode==0.5.40" in out and "002-monorepo-affected-build" in out and "120-http-cookie-jar-policy-repair" in out
    assert out.index("== candidate ==") < out.index("== previous ==")
    for line in (l for l in out.splitlines() if l.startswith("+ env")):
        assert "LOCALCODE_HOME=" in line and "HOME=" in line and "LOCALCODE_MODEL_DIR=" in line
    assert not (tmp_path / "w/logs-candidate").exists() and not (tmp_path / "w/previous-venv").exists()


def test_release_eval_end_to_end_with_stubbed_inspect(tmp_path):
    """The whole script, with `inspect` replaced by a stub that writes a run summary."""
    models = tmp_path / "models"
    models.mkdir()
    prev_venv = tmp_path / "pv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(prev_venv)], check=True)
    site = subprocess.check_output([str(prev_venv / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True).strip()
    (Path(site) / "localcode").mkdir()
    stub = tmp_path / "stubs"
    stub.mkdir()
    (stub / "inspect").write_text(
        '#!/bin/bash\n'
        'while [ $# -gt 0 ]; do [ "$1" = "--log-dir" ] && dir="$2"; shift; done\n'
        'mkdir -p "$dir"\n'
        'case "$dir" in\n'
        f'  *logs-candidate) cat {tmp_path}/cand.json > "$dir/run.json" ;;\n'
        f'  *) cat {tmp_path}/prev.json > "$dir/run.json" ;;\n'
        'esac\n')
    (stub / "inspect").chmod(0o755)
    (tmp_path / "prev.json").write_text(json.dumps(run([1, 1, 1], 100)))

    def go(cand, name):
        (tmp_path / "cand.json").write_text(json.dumps(cand))
        return eval_script(tmp_path, "--model", "m", "--models-dir", str(models), "--previous-venv", str(prev_venv), "--work", str(tmp_path / name),
                           path_extra=f"{stub}{os.pathsep}")

    ok = go(run([1, 1, 1], 110), "w1")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "RESULT: ok" in ok.stdout and (tmp_path / "w1/comparison.json").is_file()
    assert (tmp_path / "w1/previous-repo/src/localcode").is_symlink()
    worse = go(run([1, 0, 0], 100), "w2")
    assert worse.returncode == 1 and "BLOCKED" in worse.stdout
    slow = go(run([1, 1, 1], 200), "w3")
    assert slow.returncode == 1 and "slower" in slow.stdout
