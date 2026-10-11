"""LocalCode's task suite for Inspect AI.

    pip install -e evals
    inspect eval localcode_evals/regression --model localcode/<alias>
    inspect eval localcode_evals/tracking   --model localcode/<alias> --epochs 3
    inspect eval localcode_evals/regression --model localcode/<alias> -T ram_gb=16
    inspect eval localcode_evals/localcode  --model localcode/<alias> -T ids=fix-one-bug,add-test
    inspect eval-set localcode_evals/regression --model localcode/<a>,localcode/<b> --log-dir logs/x
    inspect view

Each directory under tasks/ is one sample: instruction.md is the input,
workspace/ is copied into the sample's sandbox, tests/test.sh is the hidden
grader and solution/solve.sh is the oracle (verify_oracles.py proves every
grader fails an untouched workspace and passes the oracle).
"""
from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from .agent import localcode_ui

TASKS = Path(__file__).resolve().parent.parent / "tasks"


def samples(purpose: str | None = None, ids: str | list[str] | None = None) -> list[Sample]:
    wanted = set(ids.split(",") if isinstance(ids, str) else ids or [])
    out = []
    for d in sorted(p for p in TASKS.iterdir() if (p / "task.toml").is_file()):
        meta = tomllib.loads((d / "task.toml").read_text())
        if "kind=check" in meta.get("notes", ""):           # needs no model: lives in checks/
            continue
        if wanted and d.name not in wanted:
            continue
        if not wanted and purpose and meta["purpose"] != purpose:
            continue
        ws = d / "workspace"
        out.append(Sample(
            id=d.name, input=(d / "instruction.md").read_text(), target="hidden tests pass",
            files={str(f.relative_to(ws)): str(f) for f in sorted(ws.rglob("*")) if f.is_file()},
            metadata={**meta, "task_dir": str(d)},
        ))
    missing = wanted - {s.id for s in out}
    if missing or not out:
        raise ValueError(f"unknown or empty task selection: {sorted(missing) or purpose}")
    return out


@scorer(metrics=[accuracy(), stderr()])
def hidden_tests() -> Scorer:
    """The task's own grader: tests/test.sh run against the finished workspace."""
    async def score(state: TaskState, target: Target) -> Score:
        m = state.store.get("metrics") or {}
        workspace = (await sandbox().exec(["pwd", "-P"])).stdout.strip()
        task_dir = state.metadata["task_dir"]
        run = await sandbox().exec(["bash", f"{task_dir}/tests/test.sh"], env={"WORKSPACE": workspace, "TASK_DIR": task_dir}, timeout=300)
        try:
            import json
            reward = json.loads(await sandbox().read_file(".eval/reward.json"))
        except Exception:  # noqa: BLE001
            reward = {"pass": False, "notes": f"grader wrote no reward (rc={run.returncode}): {run.stderr[-300:]}"}
        notes = str(reward.get("notes", ""))
        if m.get("timed_out"): notes = f"timed out after {m.get('wall_s')}s; {notes}"
        if m.get("runtime_failed"): notes = f"UI exited non-zero; {notes}"
        ok = bool(reward.get("pass")) and run.returncode == 0 and not m.get("timed_out") and not m.get("runtime_failed")
        return Score(value="C" if ok else "I", answer=(state.output.completion or "")[:2000], explanation=notes,
                     metadata={k: m.get(k) for k in ("changed_files", "diff", "tool_invalid_samples", "ui_error_lines", "stderr_tail", "compactions")})
    return score


NUMBERS = ("tool_validity", "steps", "tool_calls", "wall_s", "reread_tokens_max", "prompt_tokens_max", "decode_tps", "prefill_tps")


@scorer(metrics={k: [mean()] for k in NUMBERS})
def measurements() -> Scorer:
    """How the run went, apart from whether it passed: valid tool calls, steps,
    time, tokens re-read after the first turn, and the server's own speeds."""
    async def score(state: TaskState, target: Target) -> Score:
        m = state.store.get("metrics") or {}
        later = (m.get("reread_tokens") or [])[1:]
        value = {**{k: m.get(k) for k in NUMBERS}, "reread_tokens_max": max(later) if later else None}
        return Score(value={k: (float("nan") if v is None else float(v)) for k, v in value.items()},
                     metadata={**{k: v for k, v in m.items() if k not in ("diff",)}, "runtime": state.store.get("runtime")})
    return score


def build(purpose: str | None, ids: str | None, ram_gb: int | None, env: dict | None, name: str) -> Task:
    data = samples(purpose, ids)
    tags = sorted({s.metadata["quality"] for s in data}) + ([f"ram{ram_gb}"] if ram_gb else [])
    return Task(dataset=data, solver=localcode_ui(ram_gb=ram_gb, env=env), scorer=[hidden_tests(), measurements()],
                sandbox="workspace", name=name, tags=tags)


@task
def regression(ram_gb: int | None = None, env: dict | None = None) -> Task:
    """Tasks a healthy build passes nearly every time. A drop here blocks a release."""
    return build("regression", None, ram_gb, env, "regression")


@task
def tracking(ram_gb: int | None = None, env: dict | None = None) -> Task:
    """Harder tasks that inform rather than gate: packages, injection, browser, long context."""
    return build("tracking", None, ram_gb, env, "tracking")


@task
def localcode(ids: str, ram_gb: int | None = None, env: dict | None = None) -> Task:
    """Named tasks only: -T ids=fix-one-bug,add-test"""
    return build(None, ids, ram_gb, env, "localcode")
