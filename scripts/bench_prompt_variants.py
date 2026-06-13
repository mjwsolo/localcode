"""A/B benchmark: SYSTEM_PROMPT (current) vs SYSTEM_PROMPT_V2 (leaner).

Isolates the PROMPT variable: same model, same tasks, same sampling — only
the system prompt changes. Measures pass@1 (code correctness) and tokens/sec
on a real server model, so the "is v2 better for our context" question gets a
measured answer instead of intuition.

Reuses the task set + code extractor from bench_diffusion_quants.py.

Usage:
    PYTHONPATH=src python scripts/bench_prompt_variants.py [model_filename] [k]
    # default model: gemma-4-12b-it-UD-Q4_K_XL.gguf, k=3 samples/task
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_V2
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway

# Reuse the exact tasks + extractor the diffusion bench already validated.
from bench_diffusion_quants import TASKS, extract_python_code  # noqa: E402

PORT = 8198
DEFAULT_MODEL = "gemma-4-12b-it-UD-Q4_K_XL.gguf"

VARIANTS = {
    "current": SYSTEM_PROMPT,
    "v2_lean": SYSTEM_PROMPT_V2,
}


# A coding-AGENT system prompt pushes the model to USE TOOLS (write_file /
# bash) to create files — correct agent behavior, but it means a "write this
# function" task with tools=None yields a tool-call string, not inline code,
# scoring 0 (this tanked Gemma-12B to 16.7% while it was actually behaving
# agentically). A code-gen micro-bench must ask for inline code explicitly,
# identically for both variants, so we measure code-gen — not tool-eagerness.
CODE_ONLY = (
    "\n\nRespond with ONLY the Python code in a single ```python block. "
    "Do not call any tools, create files, or add explanation."
)


def _fill(prompt_tmpl: str) -> str:
    return prompt_tmpl.format(
        cwd=os.getcwd(),
        network_status="Network: ONLINE.",
        reasoning_rules="",
        project_instructions="",
        skills_block="",
    )


def _score(code: str, asserts: list) -> bool:
    # asserts are (expression_string, label) tuples. The expression is what we
    # actually check — NOT the whole tuple. `exec(f"assert {a}")` with a tuple
    # `a` compiles to `assert (expr, label)`, a non-empty tuple that is ALWAYS
    # true, so the original scoring only verified the code ran, never that it
    # was correct. Pull out a[0] so pass@1 reflects real correctness.
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 — benchmarking generated code by design
        for a in asserts:
            expr = a[0] if isinstance(a, (tuple, list)) else a
            exec(f"assert ({expr})", ns)  # noqa: S102
        return True
    except Exception:
        return False


def _wait_health(proc, url, timeout=180):
    for _ in range(timeout):
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


def main():
    model_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model_path = Path(catalog.model_dir()) / model_file
    if not model_path.is_file():
        print(f"SKIP: model not downloaded: {model_path}")
        return

    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(model_path)
    gw = LocalCodeRuntimeGateway(cfg)
    cmd = gw.llama_server_command(str(model_path))
    if "--port" in cmd:
        cmd[cmd.index("--port") + 1] = str(PORT)
    gw.config.base_url = f"http://localhost:{PORT}"
    print(f"Starting server for {model_file} on :{PORT} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: dict = {}
    try:
        _wait_health(proc, f"http://localhost:{PORT}/health")
        print("server up.\n")
        for vname, vprompt in VARIANTS.items():
            sys_msg = {"role": "system", "content": _fill(vprompt)}
            per_task = {}
            for task in TASKS:
                passes, secs = 0, []
                for _ in range(k):
                    t0 = time.monotonic()
                    content = []
                    for ev in gw.stream_chat_events(
                        [sys_msg, {"role": "user", "content": task["prompt"] + CODE_ONLY}],
                        tools=None, num_predict=512,
                    ):
                        if ev.get("type") == "content":
                            content.append(ev.get("content", ""))
                    secs.append(time.monotonic() - t0)
                    code = extract_python_code("".join(content))
                    if _score(code, task["asserts"]):
                        passes += 1
                per_task[task["id"]] = {"pass": passes, "k": k,
                                          "mean_sec": sum(secs) / len(secs)}
                print(f"[{vname}] {task['id']:16} {passes}/{k}  "
                      f"{per_task[task['id']]['mean_sec']:.1f}s")
            tot_pass = sum(t["pass"] for t in per_task.values())
            tot = sum(t["k"] for t in per_task.values())
            mean_sec = sum(t["mean_sec"] for t in per_task.values()) / len(per_task)
            results[vname] = {"pass": tot_pass, "total": tot,
                              "pass_at_1": tot_pass / tot, "mean_sec": mean_sec,
                              "per_task": per_task}
            print(f"--> {vname}: pass@1 {tot_pass}/{tot} = {tot_pass/tot:.1%}, "
                  f"mean {mean_sec:.1f}s\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    out = Path("bench_prompt_results.json")
    out.write_text(json.dumps({"model": model_file, "k": k, "results": results}, indent=2))
    print("\n=== SUMMARY ===")
    print(f"model: {model_file}, k={k}")
    for v, r in results.items():
        print(f"  {v:9} pass@1 {r['pass']}/{r['total']} = {r['pass_at_1']:.1%}  "
              f"mean {r['mean_sec']:.1f}s")
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
