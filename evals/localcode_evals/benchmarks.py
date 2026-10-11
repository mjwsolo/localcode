"""Harness-Bench 3.0, run locally through the same LocalCode UI solver. No Docker.

    inspect eval localcode_evals/harnessbench --model localcode/<model>

It measures the agent harness doing realistic jobs in a workspace (software
engineering, IT operations, compliance, data, office work), which is the product. Benchmarks that mostly rank
the model (Aider Polyglot, LiveCodeBench, SWE-bench) are deliberately not here.

The benchmark data is not in this repository and is never downloaded for you.
The task reads it from `evals/data/` (or `LOCALCODE_EVALS_DATA`) and says exactly
what is missing; `python -m localcode_evals.data` gives the command and size.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from .agent import localcode_ui
from .suite import measurements

EVALS = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LOCALCODE_EVALS_DATA") or EVALS / "data").expanduser()


class MissingData(RuntimeError):
    pass


def need(path: Path, what: str) -> Path:
    if not path.exists():
        raise MissingData(f"{what} not found at {path}. Nothing is downloaded for you: run `python -m localcode_evals.data` to see how to get it.")
    return path


async def workspace_dir() -> Path:
    return Path((await sandbox().exec(["pwd", "-P"])).stdout.strip())


def names(value) -> list[str]:
    """A task argument that is a list: Inspect passes `a,b` as a list and `a` as a string."""
    items = value.split(",") if isinstance(value, str) else list(value or [])
    return [str(i).strip() for i in items if str(i).strip()]


def pick(ids) -> set[str]:
    return set(names(ids))


# ── Harness-Bench 3.0 ─────────────────────────────────────────────────────────
# https://github.com/Qihoo360/harness-bench : each task is a directory with
# task.yaml, instruction.md, workspace/, and a hidden tests/ package whose
# test_outcome.evaluate_cases(workspace) reports one observation per business
# case; tests/cases.yaml gives each case its weight, and marks core cases and
# safety gates. The benchmark's own aggregator turns that into the score.

LOCAL_BUDGET_SECONDS = 1800

GRADER = r"""
import importlib.util, json, sys
from pathlib import Path
import yaml
task, workspace, src, outcome, cases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
sys.path[:0] = [src, str((task / outcome).parent)]
from harnessbench.grading.outcome_cases import score_workspace_from_cases
spec = importlib.util.spec_from_file_location("test_outcome", task / outcome)
module = importlib.util.module_from_spec(spec); sys.modules["test_outcome"] = module; spec.loader.exec_module(module)
result = score_workspace_from_cases(workspace, manifest=yaml.safe_load((task / cases).read_text(encoding="utf-8")),
                                    evaluate_cases=module.evaluate_cases, task_id=task.name)
print("HARNESSBENCH_RESULT " + json.dumps(result, default=str))
"""

# What the solver leaves in a workspace. The grader sees a copy without it, so a
# task that checks "nothing else was created" judges the agent, not this suite.
OURS = shutil.ignore_patterns(".git", ".eval", ".localcode-agent", "__pycache__", ".pytest_cache")


def harnessbench_root() -> Path:
    root = need(DATA / "harness-bench", "the Harness-Bench checkout")
    if not (root / "src/harnessbench/grading/outcome_cases.py").is_file():
        raise MissingData(f"{root} is not Harness-Bench 3.0 or newer (no outcome-case grader). Nothing is downloaded for you: "
                          "run `python -m localcode_evals.data` to see how to update it.")
    return root


def harnessbench_samples(ids=None, domains=None, difficulty=None, limit: int | None = None,
                         budget_seconds: int | None = None) -> tuple[list[Sample], dict[str, int]]:
    import yaml
    root = harnessbench_root()
    wanted, keep, levels, out = pick(ids), [d.lower() for d in names(domains)], pick(difficulty), []
    left_out = {"needs a container image": 0, "needs setup hooks or a service": 0}
    for d in sorted(p for p in (root / "tasks").iterdir() if (p / "task.yaml").is_file()):
        t = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8")) or {}
        tid = str(t.get("task_id") or d.name)
        if (wanted and tid not in wanted) or (keep and not any(k in str(t.get("domain", "")).lower() for k in keep)) or (levels and t.get("difficulty") not in levels):
            continue
        env = t.get("environment") or {}
        # Tasks that are only faithful inside the benchmark's own image, or that need
        # its runner to build the workspace or start a service, are left out, not scored wrong.
        if (d / "environment/Dockerfile").is_file():
            left_out["needs a container image"] += 1; continue
        workspace = d / str(t.get("workspace_dir") or "workspace")
        if (d / "environment/hooks.py").is_file() or env.get("services") or env.get("prepare") or not workspace.is_dir():
            left_out["needs setup hooks or a service"] += 1; continue
        official = int(t.get("timeout_sec") or 7200)
        out.append(Sample(
            id=tid, input=(d / str(t.get("prompt_file") or "instruction.md")).read_text(encoding="utf-8").strip(), target="task_success",
            files={str(f.relative_to(workspace)): str(f) for f in sorted(workspace.rglob("*")) if f.is_file()},
            metadata={"benchmark": "harnessbench", "id": tid, "title": t.get("title"), "domain": t.get("domain"), "difficulty": t.get("difficulty"),
                      "category": t.get("task_category"), "subcategory": t.get("task_subcategory"), "task_dir": str(d),
                      "outcome_test": str(t.get("outcome_test") or "tests/test_outcome.py"), "cases_file": str(t.get("cases_file") or "tests/cases.yaml"),
                      "official_timeout_sec": official, "budget_seconds": min(official, budget_seconds or LOCAL_BUDGET_SECONDS)}))
        if limit and len(out) >= limit:
            break
    return out, left_out


@scorer(metrics={"task_success": [mean(), stderr()], "outcome_score": [mean()]})
def harnessbench_outcome() -> Scorer:
    """The task's own outcome cases, aggregated by Harness-Bench's own code:
    `task_success` needs the pass threshold, every core case and every safety gate.
    The optional LLM semantic judge is not used, so scores are deterministic."""
    async def score(state: TaskState, target: Target) -> Score:
        ws, m = await workspace_dir(), state.metadata
        with tempfile.TemporaryDirectory(prefix="harnessbench-grade-") as tmp:
            clean = Path(tmp) / "workspace"
            shutil.copytree(ws, clean, symlinks=True, ignore=OURS)
            (Path(tmp) / "grade.py").write_text(GRADER)
            try:
                r = await sandbox().exec([sys.executable, str(Path(tmp) / "grade.py"), m["task_dir"], str(clean),
                                          str(harnessbench_root() / "src"), m["outcome_test"], m["cases_file"]], timeout=900)
                line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("HARNESSBENCH_RESULT ")), None)
            except TimeoutError:
                r, line = None, None
        if line is None:
            # A grader that cannot run is not evidence about the agent: an error, not a zero.
            raise RuntimeError(f"Harness-Bench grader failed for {state.sample_id}: {(r.stderr[-600:] if r else 'timed out after 900s')}")
        result = json.loads(line[len("HARNESSBENCH_RESULT "):])
        failed = [f"{c['label']}{' [core]' if c.get('core') else ''}{' [gate]' if c.get('gate') else ''}: {c.get('detail')}"
                  for c in result.get("checks") or [] if not c.get("pass")]
        return Score(value={"task_success": 1.0 if result.get("task_success") else 0.0, "outcome_score": float(result.get("outcome_score") or 0.0)},
                     answer=(state.output.completion or "")[:2000] if state.output else "",
                     explanation=("every case passes" if not failed else "failed cases:\n" + "\n".join(failed))[:3000],
                     metadata={k: result.get(k) for k in ("weighted_score", "pass_threshold", "failed_core_cases", "failed_gates")})
    return score


@solver
def harnessbench_reference() -> Solver:
    """Overlay the task's own reference answer instead of running a model. A grader
    that does not pass this is not working here, whatever a model scores on it:
        inspect eval localcode_evals/harnessbench --solver localcode_evals/harnessbench_reference --model none/none"""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reference = Path(state.metadata["task_dir"]) / "tests/reference"
        shutil.copytree(reference, await workspace_dir(), symlinks=True, dirs_exist_ok=True)
        return state
    return solve


@task
def harnessbench(ids: str | None = None, domains: str | None = None, difficulty: str | None = None, limit: int | None = None,
                 budget_seconds: int | None = None, ram_gb: int | None = None, env: dict | None = None) -> Task:
    """Harness-Bench 3.0: realistic jobs across software engineering, IT operations, compliance, data
    and office work, each in its own workspace and graded by the benchmark's hidden outcome cases.

    domains: words to match in a task's domain, e.g. domains=Software,Finance.
    budget_seconds: per-task time limit. The benchmark allows up to two hours a task; the default
    here is 30 minutes so a local run finishes. Pass 7200 for the official budget."""
    data, left_out = harnessbench_samples(ids, domains, difficulty, limit, budget_seconds)
    if not data:
        raise ValueError("harnessbench: the selection is empty")
    # One task whose grader cannot run must not end a run of hundreds.
    return Task(dataset=data, solver=localcode_ui(ram_gb=ram_gb, env=env), scorer=[harnessbench_outcome(), measurements()], sandbox="workspace",
                fail_on_error=False, name="harnessbench", tags=["harnessbench"] + ([f"ram{ram_gb}"] if ram_gb else []), metadata={"left_out": left_out})
