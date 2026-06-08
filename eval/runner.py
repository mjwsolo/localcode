"""localcode proficiency benchmark runner.

Two modes:
  manual   — user drives localcode TUI manually; harness times + verifies.
             Use this until localcode grows a headless entrypoint.
  headless — programmatic: localcode runs the goal end-to-end and exits.
             Requires `localcode run --goal "..." --cwd <path>` (not built yet).

Usage:
    python eval/runner.py --task 01-fizzbuzz --model qwen2.5-coder-32b-q4 --mode manual
    python eval/runner.py --all --model qwen2.5-coder-32b-q4 --mode manual
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"

# Per-task wall-clock ceiling for headless runs. CPU-only CI is slow, so
# this is generous; the agent's own turn limits usually finish well under it.
HEADLESS_TIMEOUT_S = 1800


def detect_machine() -> dict:
    chip = platform.processor() or platform.machine()
    ram_gb = None
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            ram_gb = round(int(out) / (1024**3))
        elif sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024**2))
                        break
    except Exception:
        pass
    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def machine_slug(m: dict) -> str:
    chip = (m.get("chip") or "unknown").replace(" ", "-").lower()
    ram = f"{m.get('ram_gb') or 'x'}gb"
    return f"{chip}__{ram}"


def list_tasks() -> list[str]:
    return sorted(p.name for p in TASKS_DIR.iterdir() if p.is_dir())


def run_task(task: str, model: str, mode: str) -> dict:
    task_dir = TASKS_DIR / task
    if not task_dir.exists():
        sys.exit(f"unknown task: {task}")
    task_md = (task_dir / "task.md").read_text()
    goal = task_md.split("## Goal", 1)[1].split("##", 1)[0].strip()

    workdir = Path(tempfile.mkdtemp(prefix=f"eval-{task}-"))
    setup = subprocess.run(["bash", str(task_dir / "setup.sh"), str(workdir)], capture_output=True, text=True)
    if setup.returncode != 0:
        sys.exit(f"setup failed: {setup.stderr}")

    machine = detect_machine()
    print(f"\n{'='*60}\nTASK: {task}\nMODEL: {model}\nMACHINE: {machine}\nWORKDIR: {workdir}\n{'='*60}\n")
    print("GOAL (paste into localcode):\n")
    print(goal)
    print(f"\n{'='*60}\n")

    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    if mode == "manual":
        print(f"Open a new terminal, cd to {workdir}, run: localcode --model {model}")
        print("Paste the goal above. Let it finish. Then come back here and press ENTER.")
        input("> press ENTER when the agent has finished its turn...")
    elif mode == "headless":
        # Drive localcode non-interactively via the `run` entrypoint.
        cmd = [sys.executable, "-m", "localcode"]
        if model and model != "default":
            cmd += ["--model", model]
        cmd += ["--cwd", str(workdir), "run", "--goal", goal, "--quiet",
                "--timeout", str(HEADLESS_TIMEOUT_S)]
        print(f"running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=HEADLESS_TIMEOUT_S + 60)
            print(proc.stdout[-2000:])
            if proc.returncode != 0:
                print(f"[run exited {proc.returncode}]\n{proc.stderr[-1000:]}")
        except subprocess.TimeoutExpired:
            print(f"[run timed out after {HEADLESS_TIMEOUT_S}s]")
    else:
        sys.exit(f"unknown mode: {mode}")

    wall_clock = time.time() - started
    finished_iso = datetime.now(timezone.utc).isoformat()

    verify = subprocess.run(["bash", str(task_dir / "verify.sh"), str(workdir)], capture_output=True, text=True)
    verify_exit = verify.returncode
    verify_out = (verify.stdout + verify.stderr).strip()
    print(f"\nVERIFY exit={verify_exit}\n{verify_out}\n")

    if mode == "manual":
        notes = input("Notes (what worked, what broke, any stalls)? ").strip()
    else:
        notes = ""

    result = {
        "task": task,
        "model": model,
        "mode": mode,
        "machine": machine,
        "started_at": started_iso,
        "finished_at": finished_iso,
        "wall_clock_s": round(wall_clock, 1),
        "verify_exit": verify_exit,
        "verify_reason": verify_out,
        "workdir": str(workdir),
        "notes": notes,
        "tok_total": None,  # populated when headless mode wires up
        "tok_per_s_avg": None,
        "rounds": None,
        "stalls": None,
        "bailouts": None,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{model}__{machine_slug(machine)}__{task}__{ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", help="task name (e.g. 01-fizzbuzz)")
    p.add_argument("--all", action="store_true", help="run all tasks")
    p.add_argument("--model", required=True, help="model tag (informational, recorded in results)")
    p.add_argument("--mode", choices=["manual", "headless"], default="manual")
    p.add_argument("--list", action="store_true", help="list tasks and exit")
    args = p.parse_args()

    if args.list:
        for t in list_tasks():
            print(t)
        return

    if not args.task and not args.all:
        p.error("--task or --all required")

    tasks = list_tasks() if args.all else [args.task]
    results = [run_task(t, args.model, args.mode) for t in tasks]

    if len(results) > 1:
        passed = sum(1 for r in results if r["verify_exit"] == 0)
        print(f"\n{'='*60}\nPROFICIENCY REPORT — {args.model}\n{'='*60}")
        print(f"Passed: {passed}/{len(results)}")
        for r in results:
            status = "PASS" if r["verify_exit"] == 0 else "FAIL"
            print(f"  {r['task']:25s}  {status}  {r['wall_clock_s']}s")


if __name__ == "__main__":
    main()
