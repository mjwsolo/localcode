"""HumanEval runner against a local llama-server at 127.0.0.1:8081.

Usage:
    python benchmarks/humaneval_runner.py --model-label gemma-iq3s [--subset 20]

Generates completions via /v1/chat/completions, then scores pass@1 by running
each completion against the HumanEval `test` block in a sandboxed subprocess.
Writes per-problem JSONL + a summary line to benchmarks/results/.

Why chat-format: our models are instruct-tuned; /completion bypasses the chat
template and produces worse numbers than users actually see. We benchmark the
stack as it runs in production.

Why subprocess eval: generated code can infinite-loop or call os._exit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8081"
HUMANEVAL_PATH = Path("/tmp/HumanEval.jsonl")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a Python expert. You will be given a Python function signature "
    "and docstring. Write the complete function implementation. Output ONLY "
    "the Python code for the function, starting with `def `. No explanations, "
    "no markdown fences, no example usage."
)

# Match ```python ... ``` or ``` ... ``` blocks.
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(response: str, entry_point: str) -> str:
    """Extract Python code from a model response. Models love markdown fences
    and extra prose; we strip both and return the function definition.
    """
    # 1. If there's a fenced block, prefer that.
    m = _FENCE_RE.search(response)
    if m:
        response = m.group(1)

    # 2. Find the first `def <entry_point>(` — cut everything before it.
    def_marker = f"def {entry_point}("
    idx = response.find(def_marker)
    if idx >= 0:
        response = response[idx:]

    # 3. Strip trailing prose: stop at first line that's unindented and starts
    #    with non-whitespace that's not 'def ', 'class ', '#', or '@' — that's
    #    typically the model explaining what it just wrote.
    lines = response.split("\n")
    out = []
    seen_def = False
    for ln in lines:
        stripped = ln.rstrip()
        if not seen_def:
            out.append(ln)
            if stripped.startswith("def "):
                seen_def = True
            continue
        # Stop at unindented non-code line (e.g. "This function...")
        if stripped and not stripped.startswith((" ", "\t", "#", "@", "def ", "class ")):
            # Tolerate a trailing standalone blank-return like "return x" only
            # if it's at module level after def — extremely rare; just break.
            break
        out.append(ln)
    return "\n".join(out)


def call_model(prompt: str, max_tokens: int = 768, temperature: float = 0.0) -> tuple[str, dict]:
    """Send chat-completion, return (content, timings)."""
    import urllib.request
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete this function:\n\n{prompt}"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # Our instruct models default to thinking-on; for HumanEval we want
        # the direct answer. Gemma 4 + Qwen3.6 both honor this template kw.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    elapsed = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    timings = {
        "wall_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    return content, timings


def run_test(full_program: str, timeout_s: int = 10) -> tuple[bool, str]:
    """Execute the program in a subprocess; return (passed, message)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", full_program],
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or r.stdout or "").strip().split("\n")[-1][:200]
        return False, err
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"harness-err: {e}"


def run_problem(problem: dict) -> dict:
    task_id = problem["task_id"]
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]
    test_block = problem["test"]

    t0 = time.time()
    try:
        raw, timings = call_model(prompt)
    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "error": f"model-call: {e}",
            "wall_s": time.time() - t0,
        }

    code = extract_code(raw, entry_point)

    # Build runnable program: the model's code already contains the def; if it
    # somehow omitted imports from the prompt, include the prompt preamble.
    if f"def {entry_point}" not in code:
        # Fallback: append raw response to prompt
        full_code = prompt + "\n" + raw
    elif code.startswith("from ") or code.startswith("import "):
        full_code = code
    else:
        # Keep prompt imports/preamble, then replace the def forward with model's
        preamble = prompt.split(f"def {entry_point}")[0]
        full_code = preamble + code

    program = full_code + "\n\n" + test_block + f"\n\ncheck({entry_point})\n"
    passed, err = run_test(program)

    return {
        "task_id": task_id,
        "passed": passed,
        "error": err,
        "wall_s": time.time() - t0,
        "gen_timings": timings,
        "extracted_code": code[:200],  # first 200 chars for debugging
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-label", required=True,
                    help="Label for this run (e.g. 'gemma-iq3s', 'qwen-iq2m')")
    ap.add_argument("--subset", type=int, default=None,
                    help="Only run first N problems (default: all 164)")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip first N problems (for resume)")
    args = ap.parse_args()

    # Health check
    import urllib.request
    try:
        urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5).read()
    except Exception as e:
        print(f"ERROR: llama-server not reachable at {SERVER_URL}: {e}", file=sys.stderr)
        return 1

    problems = [json.loads(l) for l in HUMANEVAL_PATH.read_text().splitlines() if l.strip()]
    problems = problems[args.start:]
    if args.subset is not None:
        problems = problems[:args.subset]

    out_file = RESULTS_DIR / f"humaneval_{args.model_label}.jsonl"
    summary_file = RESULTS_DIR / f"humaneval_{args.model_label}_summary.json"
    print(f"Running {len(problems)} problems against {SERVER_URL}")
    print(f"Results → {out_file}")

    passed_count = 0
    total_wall = 0.0
    total_tokens = 0
    t_start = time.time()
    with open(out_file, "w") as f:
        for i, prob in enumerate(problems, 1):
            r = run_problem(prob)
            f.write(json.dumps(r) + "\n")
            f.flush()
            passed_count += int(r["passed"])
            total_wall += r["wall_s"]
            ct = (r.get("gen_timings") or {}).get("completion_tokens") or 0
            total_tokens += ct
            status = "PASS" if r["passed"] else f"FAIL ({r['error'][:50]})"
            print(f"  [{i:3d}/{len(problems)}] {r['task_id']:15s} {r['wall_s']:5.1f}s {ct:4d}tok  {status}")

    total = len(problems)
    pass_at_1 = passed_count / total if total else 0.0
    summary = {
        "model_label": args.model_label,
        "n_problems": total,
        "passed": passed_count,
        "pass_at_1": pass_at_1,
        "avg_wall_s_per_problem": total_wall / total if total else 0.0,
        "total_wall_s": time.time() - t_start,
        "completion_tokens_total": total_tokens,
        "ts": time.time(),
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== {args.model_label} ===")
    print(f"pass@1: {passed_count}/{total} = {pass_at_1*100:.1f}%")
    print(f"total wall: {summary['total_wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
