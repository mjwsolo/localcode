"""Public benchmarks, run locally through the same LocalCode UI solver. No Docker.

    inspect eval localcode_evals/polyglot       --model localcode/<model> -T languages=python,rust
    inspect eval localcode_evals/livecodebench  --model localcode/<model> -T limit=50
    inspect eval localcode_evals/swebench       --model localcode/<model> -T repos=pallets/flask
    inspect eval localcode_evals/harnessbench   --model localcode/<model>

The benchmark data is not in this repository and is never downloaded for you.
Each task reads it from `evals/data/` (or `LOCALCODE_EVALS_DATA`) and says exactly
what is missing; `python -m localcode_evals.data` lists the commands and sizes.

Every benchmark carries a reference solution, so its grader can be proven
without a model:

    inspect eval localcode_evals/polyglot --solver localcode_evals/reference_solution --model none/none
"""
from __future__ import annotations

import json
import os
import shutil
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
PYTHON = os.environ.get("LOCALCODE_EVALS_TEST_PYTHON") or "python3"


class MissingData(RuntimeError):
    pass


def has(tool: str) -> bool:
    """Installed and able to run: macOS ships a `java` stub that only prints an error."""
    import subprocess
    if not shutil.which(tool):
        return False
    try:
        return subprocess.run([tool, "-version" if tool == "java" else "--version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def need(path: Path, what: str) -> Path:
    if not path.exists():
        raise MissingData(f"{what} not found at {path}. Nothing is downloaded for you: run `python -m localcode_evals.data` to see how to get it.")
    return path


async def workspace_dir() -> Path:
    return Path((await sandbox().exec(["pwd", "-P"])).stdout.strip())


def graded(passed: int, total: int, notes: str, state: TaskState, **metadata) -> Score:
    """`solved` is all-or-nothing, the number every published leaderboard uses;
    `tests_passed` is the fraction, which shows near misses."""
    total = max(total, 1)
    return Score(value={"solved": 1.0 if passed >= total else 0.0, "tests_passed": passed / total},
                 answer=(state.output.completion or "")[:2000] if state.output else "", explanation=notes[-3000:], metadata=metadata or None)


BOTH = {"solved": [mean(), stderr()], "tests_passed": [mean()]}


@solver
def reference_solution() -> Solver:
    """Apply the benchmark's own reference answer instead of running a model.
    A grader that does not pass this is broken, whatever a model scores on it."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ws = await workspace_dir()
        ref = state.metadata.get("reference") or {}
        for dest, src in (ref.get("files") or {}).items():
            (ws / dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, ws / dest)
        for dest, text in (ref.get("text") or {}).items():
            (ws / dest).parent.mkdir(parents=True, exist_ok=True)
            (ws / dest).write_text(text)
        if ref.get("patch"):
            (ws / ".reference.patch").write_text(ref["patch"])
            r = await sandbox().exec(["git", "apply", ".reference.patch"])
            if r.returncode != 0:
                raise RuntimeError(f"reference patch did not apply: {r.stderr[-300:]}")
            (ws / ".reference.patch").unlink()
        if not ref:
            raise RuntimeError(f"{state.sample_id} has no reference solution")
        return state
    return solve


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


# ── Aider Polyglot (Exercism practice exercises) ──────────────────────────────

BOOST = any(Path(p, "boost/version.hpp").is_file() for p in ("/opt/homebrew/include", "/usr/local/include", "/usr/include"))

POLYGLOT_ADDENDUM = """

Use the above instructions to modify the supplied files: {files}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests.
Only use standard libraries, don't suggest installing any packages.
The tests are already in the workspace and are correct: do not change them. Run them and fix the code until they pass."""

# How each language's tests run here. A language whose toolchain is absent is
# refused with a message rather than scored as failures.
POLYGLOT_TESTS = {
    "python": {"needs": [PYTHON], "run": lambda cfg: [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *cfg["files"]["test"]]},
    "rust": {"needs": ["cargo"], "run": lambda cfg: ["cargo", "test", "--offline", "--quiet", "--", "--include-ignored"]},
    # Exercism's CMakeLists names its target after the directory, so build in a copy named for the exercise.
    "cpp": {"needs": ["cmake"], "run": lambda cfg: ["bash", "-c", 'd="$(mktemp -d)/$0" && mkdir -p "$d" && cp -R . "$d" && cd "$d" && rm -rf build .git .eval && '
                                                    "cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=1 >/dev/null && cmake --build build", cfg["exercise"]]},
    "go": {"needs": ["go"], "run": lambda cfg: ["go", "test", "./..."]},
    "java": {"needs": ["java", "gradle"], "run": lambda cfg: ["gradle", "test", "--offline", "-q"]},
    "javascript": {"needs": ["node"], "run": lambda cfg: ["node_modules/.bin/jest", "./*", "--ci"], "node_modules": True},
}


def _reference_files(ex: Path, cfg: dict) -> dict[str, str]:
    """Each example file replaces the solution file of the same type (a header-only
    C++ answer has one example for two solution files)."""
    out, left = {}, list(cfg["solution"])
    for src in cfg["example"]:
        same = [d for d in left if Path(d).suffix == Path(src).suffix]
        dst = (same or left or [None])[0]
        if dst:
            out[dst] = str(ex / src); left.remove(dst)
    return out


def polyglot_samples(languages: str, ids, limit: int | None) -> list[Sample]:
    root = need(DATA / "polyglot-benchmark", "the Aider polyglot-benchmark checkout")
    wanted, out = pick(ids), []
    for lang in names(languages):
        spec = POLYGLOT_TESTS.get(lang)
        if spec is None:
            raise ValueError(f"unknown language {lang!r}; choose from {sorted(POLYGLOT_TESTS)}")
        absent = [t for t in spec["needs"] if not has(t)]
        if absent:
            raise MissingData(f"polyglot {lang}: {', '.join(absent)} is not installed on this machine")
        if spec.get("node_modules"):
            need(DATA / "polyglot-jsdeps/node_modules/.bin/jest", "the shared jest install for the JavaScript exercises")
        kept = 0
        for ex in sorted(p for p in need(root / lang / "exercises/practice", f"polyglot {lang} exercises").iterdir() if p.is_dir()):
            sid = f"{lang}/{ex.name}"
            cfg_file = ex / ".meta/config.json"
            if not cfg_file.is_file() or (wanted and sid not in wanted):
                continue
            cfg = json.loads(cfg_file.read_text())["files"]
            if not cfg.get("solution") or not cfg.get("test") or not cfg.get("example"):
                continue
            if lang == "cpp" and "find_package(Boost" in (ex / "CMakeLists.txt").read_text() and not BOOST:
                continue                                        # cannot build here: Boost is not installed
            docs = [ex / ".docs" / n for n in ("introduction.md", "instructions.md", "instructions.append.md")]
            prompt = "\n\n".join(d.read_text() for d in docs if d.is_file()) + POLYGLOT_ADDENDUM.format(files=" ".join(cfg["solution"]))
            # .meta holds the reference solution and .approaches the write-ups: neither reaches the workspace.
            files = {str(f.relative_to(ex)): str(f) for f in sorted(ex.rglob("*"))
                     if f.is_file() and not {".meta", ".approaches", ".docs"} & set(f.relative_to(ex).parts)}
            out.append(Sample(id=sid, input=prompt, target="all tests pass", files=files, metadata={
                "benchmark": "polyglot", "language": lang, "exercise": ex.name, "exercise_dir": str(ex), "tests": cfg["test"], "files": {"test": cfg["test"]},
                "allowed_paths": cfg["solution"], "budget_seconds": 900,
                "reference": {"files": _reference_files(ex, cfg)}}))
            kept += 1
            if limit and kept >= limit:
                break
    return out


@scorer(metrics=BOTH)
def polyglot_tests() -> Scorer:
    """Restore the exercise's own test files, then run them."""
    async def score(state: TaskState, target: Target) -> Score:
        ws, m = await workspace_dir(), state.metadata
        for rel in m["tests"]:                                   # an edited test cannot help
            shutil.copyfile(Path(m["exercise_dir"]) / rel, ws / rel)
        spec = POLYGLOT_TESTS[m["language"]]
        if spec.get("node_modules") and not (ws / "node_modules").exists():
            (ws / "node_modules").symlink_to(DATA / "polyglot-jsdeps/node_modules")
        try:
            r = await sandbox().exec(spec["run"](m), env={"CI": "1"}, timeout=600)
            ok, out = r.returncode == 0, (r.stdout + r.stderr)
        except TimeoutError:
            ok, out = False, "tests timed out after 600s"
        return graded(1 if ok else 0, 1, "all tests pass" if ok else out, state)
    return score


@task
def polyglot(languages: str = "python", ids: str | None = None, limit: int | None = None, ram_gb: int | None = None, env: dict | None = None) -> Task:
    """Aider Polyglot: Exercism practice exercises, tests visible, fix until green."""
    return build("polyglot", polyglot_samples(languages, ids, limit), polyglot_tests(), ram_gb, env)


# ── LiveCodeBench (dated competitive-programming problems) ────────────────────

LCB_RUNNER = r'''
import json, subprocess, sys
cases = json.load(open(sys.argv[2])); passed = 0
for c in cases:
    try:
        r = subprocess.run([sys.executable, sys.argv[1]], input=c["input"], capture_output=True, text=True, timeout=15)
        passed += r.stdout.strip() == str(c["output"]).strip()
    except Exception:
        pass
print(json.dumps({"passed": passed, "total": len(cases)}))
'''


def livecodebench_samples(since: str, until: str | None, ids, limit: int | None) -> list[Sample]:
    index = need(DATA / "livecodebench/index.jsonl", "the LiveCodeBench index (built by `python -m localcode_evals.data livecodebench`)")
    wanted, out = pick(ids), []
    for line in index.read_text().splitlines():
        p = json.loads(line)
        if wanted and p["id"] not in wanted:
            continue
        if not wanted and (p["date"] < since or (until and p["date"] > until)):
            continue
        prompt = ("Write a Python program `solution.py` that reads from standard input and writes the answer to standard output "
                  "for the following problem. Test it against the samples in the statement before you finish.\n\n" + p["content"].strip())
        out.append(Sample(id=p["id"], input=prompt, target="every public test case passes", metadata={
            "benchmark": "livecodebench", "platform": p["platform"], "date": p["date"], "difficulty": p.get("difficulty"),
            "cases": p["cases"], "allowed_paths": ["solution.py"], "budget_seconds": 600}))
        if limit and len(out) >= limit:
            break
    return out


@scorer(metrics=BOTH)
def livecodebench_cases() -> Scorer:
    """Run solution.py on every public stdin case and compare stdout."""
    async def score(state: TaskState, target: Target) -> Score:
        ws = await workspace_dir()
        if not (ws / "solution.py").is_file():
            return graded(0, 1, "solution.py missing", state)
        grader = ws / ".eval"; grader.mkdir(exist_ok=True)
        (grader / "lcb_run.py").write_text(LCB_RUNNER); (grader / "lcb_cases.json").write_text(json.dumps(state.metadata["cases"]))
        r = await sandbox().exec([PYTHON, ".eval/lcb_run.py", "solution.py", ".eval/lcb_cases.json"], timeout=600)
        try:
            got = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return graded(0, 1, f"grader failed: {r.stderr[-300:]}", state)
        return graded(got["passed"], got["total"], f"{got['passed']}/{got['total']} public cases", state)
    return score


@task
def livecodebench(since: str = "2000-01-01", until: str | None = None, ids: str | None = None, limit: int | None = None,
                  ram_gb: int | None = None, env: dict | None = None) -> Task:
    """LiveCodeBench stdin problems. Each has a release date: set `since` past the model's training cutoff
    for a contamination-free score, which needs a release file that recent (`python -m localcode_evals.data`)."""
    return build("livecodebench", livecodebench_samples(str(since), str(until) if until else None, ids, limit), livecodebench_cases(), ram_gb, env)


# ── SWE-bench Verified (real GitHub issues), environments built with uv ───────

LIGHT_REPOS = "pallets/flask,psf/requests,pytest-dev/pytest,pylint-dev/pylint"


def swebench_samples(repos: str, ids, limit: int | None, unproven: bool = False) -> list[Sample]:
    index = need(DATA / "swebench/instances.jsonl", "the SWE-bench Verified index (built by `python -m localcode_evals.data swebench`)")
    wanted, keep, out = pick(ids), pick(repos), []
    proven = DATA / "swebench/builds.json"                      # written by `python -m localcode_evals.data swebench-builds <log>`
    builds = set(json.loads(proven.read_text())) if proven.is_file() and not unproven else None
    for line in index.read_text().splitlines():
        i = json.loads(line)
        if (wanted and i["instance_id"] not in wanted) or (not wanted and i["repo"] not in keep):
            continue
        if builds is not None and not wanted and i["instance_id"] not in builds:
            continue
        pins = " ".join(f"'{p}'" for p in i["pip_packages"])
        # Instances predate pytest 8's API removals; pytest's own repository tests itself.
        runner = "" if i["repo"] == "pytest-dev/pytest" else "'pytest>=7,<8'"
        mirror = DATA / "swebench/repos" / (i["repo"].replace("/", "__") + ".git")
        # A bare mirror is cloned once per repository and reused; the environment is
        # built in the workspace by uv and kept out of git, so it never shows in the diff.
        setup = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p '{mirror.parent}'
until mkdir '{mirror}.lock' 2>/dev/null; do sleep 2; done      # one clone per repository, even with parallel samples
trap "rmdir '{mirror}.lock'" EXIT
if [ ! -d '{mirror}' ]; then git clone -q --bare https://github.com/{i['repo']} '{mirror}.partial' && mv '{mirror}.partial' '{mirror}'; fi
rmdir '{mirror}.lock'; trap - EXIT
git init -q . && git remote add origin '{mirror}' && git fetch -q --tags origin '+refs/heads/*:refs/remotes/origin/*'
git checkout -q {i['base_commit']}
printf '.v/\\n.eval/\\n' >> .git/info/exclude
uv venv -q --python {i['python']} .v
uv pip install -q --python .v/bin/python -e . {pins} {runner} >/dev/null 2>&1 || true
"""
        out.append(Sample(id=i["instance_id"], input=i["problem_statement"].strip() + "\n\nFix this in the repository in the current directory. "
                          "A Python environment for it is at `.v/bin/python`. Do not edit or add tests.",
                          target="FAIL_TO_PASS tests pass", setup=setup, metadata={
                              "benchmark": "swebench", "repo": i["repo"], "test_patch": i["test_patch"], "fail_to_pass": i["fail_to_pass"],
                              "budget_seconds": 1800, "reference": {"patch": i["patch"]}}))
        if limit and len(out) >= limit:
            break
    return out


@scorer(metrics=BOTH)
def swebench_tests() -> Scorer:
    """Apply the instance's hidden test patch and run its FAIL_TO_PASS tests."""
    async def score(state: TaskState, target: Target) -> Score:
        ws, m = await workspace_dir(), state.metadata
        (ws / ".eval").mkdir(exist_ok=True); (ws / ".eval/test.patch").write_text(m["test_patch"])
        applied = await sandbox().exec(["bash", "-c", "git apply .eval/test.patch 2>/dev/null || git apply --3way .eval/test.patch"])
        if not (ws / ".v/bin/python").exists():
            return graded(0, len(m["fail_to_pass"]), "the environment did not build on this machine", state, built=False)
        r = await sandbox().exec([".v/bin/python", "-m", "pytest", "-rA", "-p", "no:cacheprovider", *m["fail_to_pass"]], timeout=1200)
        passed = sum(1 for line in r.stdout.splitlines() if line.startswith("PASSED "))
        tail = "\n".join(r.stdout.strip().splitlines()[-15:])
        return graded(passed, len(m["fail_to_pass"]), f"{passed}/{len(m['fail_to_pass'])} FAIL_TO_PASS" + ("" if applied.returncode == 0 else "; test patch did not apply") + "\n" + tail,
                      state, built=True)
    return score


@task
def swebench(repos: str = LIGHT_REPOS, ids: str | None = None, limit: int | None = None, unproven: bool = False,
             ram_gb: int | None = None, env: dict | None = None) -> Task:
    """SWE-bench Verified without Docker. Each instance's pinned environment is built with uv, which
    works for the pure-Python repositories; the default selection is those. An instance whose gold
    patch does not pass here does not build locally and tells you nothing about a model, so once the
    reference solver has been run and recorded (`python -m localcode_evals.data swebench-builds <log>`)
    only the proven instances are scored. `unproven=true` runs them all, for that reference run."""
    return build("swebench", swebench_samples(repos, ids, limit, unproven), swebench_tests(), ram_gb, env)


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
