"""HarnessBench, run locally through the same LocalCode UI solver. No Docker.

    inspect eval localcode_evals/harnessbench --model localcode/<model>

It measures the agent harness doing real workspace jobs (files, terminal, data,
office documents, codebases), which is the product. Benchmarks that mostly rank
the model (Aider Polyglot, LiveCodeBench, SWE-bench) are deliberately not here.

The benchmark data is not in this repository and is never downloaded for you.
The task reads it from `evals/data/` (or `LOCALCODE_EVALS_DATA`) and says exactly
what is missing; `python -m localcode_evals.data` gives the command and size.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def build(name: str, data: list[Sample], grader: Scorer, ram_gb, env, setup: Solver | None = None, cleanup=None) -> Task:
    if not data:
        raise ValueError(f"{name}: the selection is empty")
    # One sample whose environment will not build must not end a run of hundreds.
    return Task(dataset=data, setup=setup, solver=localcode_ui(ram_gb=ram_gb, env=env), cleanup=cleanup, fail_on_error=False,
                scorer=[grader, measurements()], sandbox="local", name=name, tags=[name] + ([f"ram{ram_gb}"] if ram_gb else []))


def names(value) -> list[str]:
    """A task argument that is a list: Inspect passes `a,b` as a list and `a` as a string."""
    items = value.split(",") if isinstance(value, str) else list(value or [])
    return [str(i).strip() for i in items if str(i).strip()]


def pick(ids) -> set[str]:
    return set(names(ids))


# ── HarnessBench (workspace tasks with programmatic oracles) ──────────────────

def _harness_spec(task_dir: Path) -> SimpleNamespace:
    raw = (task_dir / "task.yaml").read_text(encoding="utf-8")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        import yaml
        d = yaml.safe_load(raw) or {}
    return SimpleNamespace(task_id=str(d["task_id"]), title=str(d.get("title", d["task_id"])), prompt_file=str(d.get("prompt_file", "prompt.txt")),
                           prompt_files=[str(x) for x in (d.get("prompt_files") or [])], fixtures_dir=str(d.get("fixtures_dir", "fixtures")),
                           oracle_module=str(d.get("oracle_module", "oracle_grade.py")), hooks_module=str(d.get("hooks_module", "hooks.py")),
                           timeout_sec=int(d.get("timeout_sec", 600)), tags=list(d.get("tags") or []), task_dir=task_dir.resolve(),
                           task_class=str(d.get("class", "")), difficulty=str(d.get("difficulty", "")))


def _load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harnessbench_samples(ids, classes: str | None, limit: int | None) -> tuple[list[Sample], dict[str, int]]:
    root = need(DATA / "harness-bench/tasks", "the HarnessBench checkout")
    wanted, keep, out, skipped = pick(ids), pick(classes), [], {"needs a public tunnel": 0, "needs per-round hooks": 0}
    for d in sorted(p for p in root.iterdir() if (p / "task.yaml").is_file()):
        t = _harness_spec(d)
        if (wanted and t.task_id not in wanted) or (keep and not any(k.lower() in t.task_class.lower() for k in keep)):
            continue
        hooks = (d / t.hooks_module).read_text(encoding="utf-8") if (d / t.hooks_module).is_file() else ""
        # Two kinds of task cannot run faithfully here yet, and are left out rather than scored wrong.
        if "trycloudflare" in hooks or "PUBLIC_URL" in hooks:
            skipped["needs a public tunnel"] += 1; continue
        if "def after_round" in hooks:
            skipped["needs per-round hooks"] += 1; continue
        prompts = t.prompt_files or [t.prompt_file]
        texts = [(d / p).read_text(encoding="utf-8").strip() for p in prompts]
        prompt = texts[0] if len(texts) == 1 else "\n\n".join(f"## Turn {n}\n\n{x}" for n, x in enumerate(texts, 1))
        fixtures = d / t.fixtures_dir
        files = {str(f.relative_to(fixtures)): str(f) for f in sorted(fixtures.rglob("*")) if f.is_file()} if fixtures.is_dir() else {}
        out.append(Sample(id=t.task_id, input=prompt, target="oracle outcome_score of 1.0", files=files, metadata={
            "benchmark": "harnessbench", "title": t.title, "class": t.task_class, "difficulty": t.difficulty, "task_dir": str(d),
            "has_hooks": bool(hooks), "budget_seconds": t.timeout_sec}))
        if limit and len(out) >= limit:
            break
    return out, skipped


_harness_state: dict[str, tuple] = {}


@solver
def harnessbench_prepare() -> Solver:
    """The task's own prepare_runtime hook (mock servers and the like), and the
    prompt's $WORKSPACE and runtime placeholders."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ws = await workspace_dir()
        (ws / "in").mkdir(exist_ok=True); (ws / "out").mkdir(exist_ok=True)
        t = _harness_spec(Path(state.metadata["task_dir"]))
        runtime_env: dict[str, str] = {}
        if state.metadata.get("has_hooks"):
            hooks = _load(t.task_dir / t.hooks_module, f"harnessbench_hooks_{t.task_id.replace('-', '_')}")
            ctx = {"task": t, "sandbox": ws.parent, "workspace": ws}
            started = hooks.prepare_runtime(ctx) if callable(getattr(hooks, "prepare_runtime", None)) else None
            started = started if isinstance(started, dict) else {}
            runtime_env = {k: v for k, v in started.items() if isinstance(v, str)}
            _harness_state[str(state.uuid)] = (hooks, ctx, started)
        text = state.user_prompt.text.replace("$WORKSPACE", str(ws))
        for k, v in runtime_env.items():
            text = text.replace(f"${k}", v)
        state.user_prompt.text = text
        return state
    return solve


async def harnessbench_cleanup(state: TaskState) -> None:
    hooks, ctx, started = _harness_state.pop(str(state.uuid), (None, None, None))
    if hooks is not None and callable(getattr(hooks, "cleanup_runtime", None)):
        hooks.cleanup_runtime(ctx, started)


@scorer(metrics={"solved": [mean(), stderr()], "outcome_score": [mean()]})
def harnessbench_oracle() -> Scorer:
    """The task's own oracle_grade.score_workspace: weighted programmatic checks.
    The benchmark's optional LLM quality rubric is not used, so scores are deterministic."""
    async def score(state: TaskState, target: Target) -> Score:
        ws = await workspace_dir()
        t = _harness_spec(Path(state.metadata["task_dir"]))
        try:
            result = dict(_load(t.task_dir / t.oracle_module, f"harnessbench_oracle_{t.task_id.replace('-', '_')}").score_workspace(ws))
        except Exception as e:  # noqa: BLE001 - a crashing oracle is a failed sample, not a failed run
            return Score(value={"solved": 0.0, "outcome_score": 0.0}, explanation=f"oracle raised {type(e).__name__}: {e}"[:600])
        outcome = float(result.get("outcome_score") or 0.0)
        failed = [f"{c.get('label') or c.get('id')}: {c.get('detail')}" for c in result.get("checks") or [] if not c.get("pass")]
        return Score(value={"solved": 1.0 if outcome >= 0.999 else 0.0, "outcome_score": outcome},
                     answer=(state.output.completion or "")[:2000] if state.output else "",
                     explanation=("all checks pass" if not failed else "failed checks:\n" + "\n".join(failed))[:3000],
                     metadata={"grade": result.get("grade"), "checks": len(result.get("checks") or [])})
    return score


@task
def harnessbench(ids: str | None = None, classes: str | None = None, limit: int | None = None, ram_gb: int | None = None, env: dict | None = None) -> Task:
    """HarnessBench: office, data, SRE, knowledge and codebase tasks in a real workspace, graded by each
    task's programmatic oracle. `classes` filters by a word in the task class, e.g. classes=Software,SRE."""
    data, skipped = harnessbench_samples(ids, classes, limit)
    t = build("harnessbench", data, harnessbench_oracle(), ram_gb, env, setup=harnessbench_prepare(), cleanup=harnessbench_cleanup)
    t.metadata = {"left_out": skipped}
    return t
