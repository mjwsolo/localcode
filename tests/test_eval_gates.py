"""Incomplete model runs must never produce a successful evaluation gate."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def row(trial=1, passed=True):
    return dict(task="fix", model="test", trial=trial, purpose="regression", **{"pass": passed})


@pytest.mark.parametrize("baseline,candidate", [
    ([], []),
    ([row()], []),
    ([row(), row(2)], [row()]),
    ([row()], [row(), row()]),
    ([row(), row()], [row()]),
    ([row()], [dict(row(), model="other")]),
    ([row()], [dict(row(), **{"pass": "false"})]),
])
def test_gate_rejects_invalid_coverage(tmp_path, baseline, candidate):
    paths = [tmp_path / name for name in ("baseline.jsonl", "candidate.jsonl")]
    for path, rows in zip(paths, (baseline, candidate)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = subprocess.run([sys.executable, str(ROOT / "evals/gate.py"), *map(str, paths)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "BLOCKED" in result.stderr


@pytest.mark.parametrize("passed,code", [(True, 0), (False, 1)])
def test_gate_accepts_matched_trials_and_detects_regression(tmp_path, passed, code):
    baseline, candidate = tmp_path / "b", tmp_path / "c"
    baseline.write_text(json.dumps(row()) + "\n")
    candidate.write_text(json.dumps(row(passed=passed)) + "\n")
    result = subprocess.run([sys.executable, str(ROOT / "evals/gate.py"), str(baseline), str(candidate)], capture_output=True, text=True)
    assert result.returncode == code


def test_nightly_propagates_failed_runner_without_reusing_old_results(tmp_path):
    harness = tmp_path / "evals"
    harness.mkdir()
    shutil.copy(ROOT / "evals/nightly.sh", harness)
    (harness / "verify_oracles.py").write_text("")
    (harness / "gate.py").write_text("from pathlib import Path\nPath('gate-was-run').touch()\n")
    previous = harness / "runs/nightly/model/20260101-000000-nightly-old"
    previous.mkdir(parents=True)
    (previous / "old.eval").write_text("")
    (previous / ".complete").touch()
    repo = tmp_path / "repo"
    (repo / ".venv/bin").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".venv/bin/python").write_text("#!/bin/sh\nexit 0\n")
    (repo / ".venv/bin/python").chmod(0o755)
    home = tmp_path / "home"
    models = home / ".local/share/localcode/models"
    models.mkdir(parents=True)
    (models / "model.gguf").touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "lsof").write_text("#!/bin/sh\nexit 1\n")
    (tools / "lsof").chmod(0o755)
    (tools / "python3").symlink_to(sys.executable)
    (tools / "inspect").write_text("#!/bin/sh\nexit 7\n")
    (tools / "inspect").chmod(0o755)
    result = subprocess.run(["bash", str(harness / "nightly.sh"), str(repo), "model"], env={**os.environ, "HOME": str(home), "PATH": str(tools) + os.pathsep + os.environ["PATH"]}, capture_output=True, text=True)
    assert result.returncode == 1
    assert "TASK RUN FAILED" in result.stdout
    assert not (harness / "gate-was-run").exists()


def test_task_stderr_survives_model_deleting_eval_directory(tmp_path):
    # Execute the runner's real post-invocation write without loading a model.
    tree = ast.parse((ROOT / "evals/localcode_evals/agent.py").read_text())
    write = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_write")
    trial = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_trial")
    statement = next(n for n in ast.walk(trial) if isinstance(n, ast.Expr) and "stderr-turn" in ast.unparse(n))
    env = {"Path": Path, "ev": tmp_path / ".eval", "i": 1, "r": SimpleNamespace(stderr="diagnostic")}
    exec(compile(ast.Module(body=[write, statement], type_ignores=[]), "runner-write", "exec"), env)
    assert (tmp_path / ".eval/stderr-turn1.txt").read_text() == "diagnostic"
