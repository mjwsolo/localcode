#!/usr/bin/env python3
"""Prove every task's grader twice: the untouched workspace must FAIL (unless
null_passes), and the oracle solution must PASS. Exit 1 if any task breaks that."""
import json, re, os, shutil, subprocess, sys, tempfile, tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parent / "tasks"

def git(ws, *a): subprocess.run(["git", "-c", "user.email=e@x", "-c", "user.name=e", *a], cwd=ws, check=True, capture_output=True)

def stage(task: Path) -> Path:
    ws = Path(tempfile.mkdtemp(prefix=f"task-{task.name}-")).resolve()
    shutil.copytree(task / "workspace", ws, dirs_exist_ok=True)
    git(ws, "init", "-q"); git(ws, "add", "-A"); git(ws, "commit", "-qm", "init", "--allow-empty")
    (ws / ".eval").mkdir(exist_ok=True)
    for f in ("commands.log", "final.txt", "ui_errors.txt", "diff.txt"): (ws / ".eval" / f).touch()
    return ws

def finish(ws: Path):
    # what the runner writes before the grader: changed paths vs the initial commit (untracked included, .eval excluded)
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ws, capture_output=True, text=True).stdout
    paths = sorted({l[3:].split(" -> ")[-1] for l in out.splitlines() if l.strip() and not l[3:].startswith(".eval/") and not l[3:].startswith(".localcode-agent/") and not re.search(r"(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/)|\.pyc$", l[3:])})
    (ws / ".eval" / "diff.txt").write_text("\n".join(paths) + ("\n" if paths else ""))

def grade(task: Path, ws: Path) -> dict:
    env = {**os.environ, "WORKSPACE": str(ws), "TASK_DIR": str(task)}
    r = subprocess.run(["bash", str(task / "tests" / "test.sh")], env=env, capture_output=True, text=True, timeout=300)
    rf = ws / ".eval" / "reward.json"
    if not rf.exists(): return {"pass": False, "notes": f"no reward.json (rc={r.returncode}) {r.stderr[-200:]}"}
    return json.loads(rf.read_text())

rows, bad = [], 0
for task in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    meta = tomllib.loads((task / "task.toml").read_text())
    ws = stage(task); finish(ws); null = grade(task, ws); shutil.rmtree(ws, ignore_errors=True)
    ws = stage(task)
    s = subprocess.run(["bash", str(task / "solution" / "solve.sh")], env={**os.environ, "WORKSPACE": str(ws), "TASK_DIR": str(task)}, capture_output=True, text=True, timeout=300)
    finish(ws); oracle = grade(task, ws); shutil.rmtree(ws, ignore_errors=True)
    null_ok = null["pass"] == bool(meta.get("null_passes", False)); oracle_ok = oracle["pass"] is True and s.returncode == 0
    bad += (not null_ok) + (not oracle_ok)
    rows.append((task.name, meta["quality"], meta["purpose"], "ok" if null_ok else f"WRONG ({null['notes'][:50]})", "ok" if oracle_ok else f"FAIL ({(oracle['notes'] or s.stderr)[:60]})"))
w = max(len(r[0]) for r in rows)
print(f"{'task':<{w}}  {'quality':<13}{'purpose':<11}{'null run':<22}oracle")
for r in rows: print(f"{r[0]:<{w}}  {r[1]:<13}{r[2]:<11}{r[3]:<22}{r[4]}")
print(f"\n{len(rows)} tasks, {bad} problems"); sys.exit(1 if bad else 0)
