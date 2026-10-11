"""Where the benchmark data must be, and how to get it. Nothing here downloads.

    python -m localcode_evals.data                  # what is present, what is missing, and the command for each
    python -m localcode_evals.data livecodebench    # build the slim index from a test.jsonl already on disk
    python -m localcode_evals.data swebench         # build the instance index from the parquet already on disk
    python -m localcode_evals.data swebench-builds <reference log>   # record which instances build on this machine

The data lives in `evals/data/` (gitignored) or `LOCALCODE_EVALS_DATA`. A symlink
to a copy you already have is fine.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .benchmarks import DATA, has

HF = Path.home() / ".cache/huggingface/hub"

SOURCES = {
    "polyglot": (DATA / "polyglot-benchmark", "about 40 MB",
                 "git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark {path}"),
    "polyglot javascript": (DATA / "polyglot-jsdeps/node_modules", "about 150 MB, JavaScript exercises only",
                            "mkdir -p {parent} && cd {parent} && npm init -y && npm install jest@29 babel-jest@29 @babel/core @exercism/babel-preset-javascript core-js@3.37"),
    "harnessbench": (DATA / "harness-bench", "about 35 MB",
                     "git clone --depth 1 https://github.com/Qihoo360/harness-bench {path}"),
    "livecodebench": (DATA / "livecodebench/index.jsonl", "source file is about 2.3 GB, the index a few MB",
                      "hf download livecodebench/code_generation_lite test.jsonl --repo-type dataset && python -m localcode_evals.data livecodebench"),
    "swebench": (DATA / "swebench/instances.jsonl", "source file is about 4 MB; each repository is cloned once on first use",
                 "pip install pyarrow swebench && hf download SWE-bench/SWE-bench_Verified --repo-type dataset && python -m localcode_evals.data swebench"),
}


def cached(repo: str, name: str) -> Path | None:
    hits = sorted((HF / f"datasets--{repo.replace('/', '--')}/snapshots").glob(f"*/{name}"))
    return hits[-1] if hits else None


def status() -> int:
    print(f"benchmark data directory: {DATA}\n")
    for name, (path, size, how) in SOURCES.items():
        mark = "present" if path.exists() else "MISSING"
        print(f"{name:22} {mark:8} {path}")
        if not path.exists():
            print(f"{'':22} {size}\n{'':22} {how.format(path=path, parent=path.parent)}")
    print("\ntoolchains for polyglot: " + ", ".join(f"{t} {'ok' if has(t) else 'absent'}" for t in ("python3", "cargo", "cmake", "node", "go", "java")))
    return 0


def build_livecodebench() -> int:
    src = cached("livecodebench/code_generation_lite", "test.jsonl")
    if src is None:
        print("test.jsonl is not in the Hugging Face cache; see `python -m localcode_evals.data`", file=sys.stderr); return 1
    out = DATA / "livecodebench/index.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src.open() as lines, out.open("w") as index:
        for line in lines:
            # The private test cases make each line megabytes long; only stdin problems are kept.
            if '"testtype": "stdin"' not in line and '\\"testtype\\": \\"stdin\\"' not in line:
                continue
            p = json.loads(line)
            try:
                cases = json.loads(p["public_test_cases"]) if isinstance(p["public_test_cases"], str) else p["public_test_cases"]
            except Exception:  # noqa: BLE001
                continue
            if not cases or any(c.get("testtype") != "stdin" for c in cases):
                continue
            index.write(json.dumps({"id": re.sub(r"[^A-Za-z0-9_.-]", "-", str(p.get("question_id") or p.get("question_title"))),
                                    "platform": p.get("platform"), "date": str(p.get("contest_date", ""))[:10], "difficulty": p.get("difficulty"),
                                    "content": p.get("question_content", ""), "cases": [{"input": c["input"], "output": c["output"]} for c in cases]}) + "\n")
            kept += 1
    rows = sorted((json.loads(l) for l in out.read_text().splitlines()), key=lambda r: (r["date"], r["id"]))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{kept} stdin problems, {rows[0]['date']} to {rows[-1]['date']} -> {out}")
    return 0


def build_swebench() -> int:
    src = cached("SWE-bench/SWE-bench_Verified", "data/test-00000-of-00001.parquet")
    if src is None:
        print("the SWE-bench Verified parquet is not in the Hugging Face cache; see `python -m localcode_evals.data`", file=sys.stderr); return 1
    try:
        import pyarrow.parquet as pq
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except ImportError as e:
        print(f"building this index needs pyarrow and swebench ({e}); nothing else here does", file=sys.stderr); return 1
    out = DATA / "swebench/instances.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w") as index:
        for i in pq.read_table(src).to_pylist():
            spec = MAP_REPO_VERSION_TO_SPECS.get(i["repo"], {}).get(i["version"])
            if not spec or not spec.get("python"):
                continue
            f2p = i["FAIL_TO_PASS"]
            index.write(json.dumps({"instance_id": i["instance_id"], "repo": i["repo"], "version": i["version"], "base_commit": i["base_commit"],
                                    "problem_statement": i["problem_statement"], "patch": i["patch"], "test_patch": i["test_patch"],
                                    "fail_to_pass": json.loads(f2p) if isinstance(f2p, str) else list(f2p),
                                    "python": spec["python"], "pip_packages": list(spec.get("pip_packages") or [])}) + "\n")
            kept += 1
    print(f"{kept} instances -> {out}")
    return 0


def record_swebench_builds() -> int:
    """Keep the instances whose gold patch passed in a reference-solver log."""
    from inspect_ai.log import read_eval_log
    if len(sys.argv) < 3:
        print("usage: python -m localcode_evals.data swebench-builds <reference log> [more logs]", file=sys.stderr); return 1
    out = DATA / "swebench/builds.json"
    ok = set(json.loads(out.read_text())) if out.is_file() else set()
    seen = 0
    for path in sys.argv[2:]:
        log = read_eval_log(path)
        if log.eval.solver != "localcode_evals/reference_solution":
            print(f"{path} was not run with the reference solver", file=sys.stderr); return 1
        for s in log.samples or []:
            seen += 1
            grade = (s.scores or {}).get("swebench_tests")
            if not s.error and grade and grade.value.get("solved") == 1:
                ok.add(str(s.id))
            else:
                ok.discard(str(s.id))
    out.write_text(json.dumps(sorted(ok), indent=1))
    print(f"{len(ok)} instances build and pass their gold patch here ({seen} checked) -> {out}")
    return 0


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"": status, "livecodebench": build_livecodebench, "swebench": build_swebench, "swebench-builds": record_swebench_builds}.get(what, status)())
