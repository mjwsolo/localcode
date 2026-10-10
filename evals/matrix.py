#!/usr/bin/env python3
"""Coverage matrix: every catalog model x every RAM tier, through the task harness.

    python3 evals/matrix.py --repo <checkout> [--tiers 16,24,32,64,this] [--models key,...]
                            [--set regression] [--trials 1] [--execute]

For each (model, tier) the supervisor is started with LOCALCODE_RAM_GB=<tier>, so
the picker chooses the quant, context and slots a Mac of that size would get; the
model it picks is then run through the task set with run_tasks.py, labelled
`tier<N>`. "this" means the real machine. The host must actually hold what the
smaller tier picks (it always does: smaller tiers pick smaller quants), and
nothing here measures real memory pressure on small hardware; it measures the
quality of the choices made for it. Results land in evals/runs/ like any run,
so dashboard.py and gate.py read them.

The default prints the plan. --execute loads models (hours).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def picked_alias(repo: Path, py: Path, group: str, ram_gb: int | None) -> tuple[str, str] | None:
    """Ask a throwaway supervisor (no model loaded) which quant it would pick for this
    group on a machine with ram_gb of memory. Returns (alias, filename)."""
    env = {**os.environ, "HOME": str(Path(os.environ.get("TMPDIR", "/tmp")) / f"lc-matrix-home-{os.getpid()}")}
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("LOCALCODE_CONTROL_TOKEN", "matrix"); env.setdefault("LOCALCODE_SERVER_KEY", "matrix")
    if ram_gb: env["LOCALCODE_RAM_GB"] = str(ram_gb)
    else: env.pop("LOCALCODE_RAM_GB", None)
    mport, cport = free_port(), free_port()
    proc = subprocess.Popen([str(py), "-m", "localcode.ui.supervisor", "--model", "", "--port", str(mport), "--control-port", str(cport),
                             "--server", str(repo / "src/localcode/bin/llama-server"), "--models-dir", str(Path.home() / ".local/share/localcode/models"),
                             "--parent-pid", str(os.getpid())], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    hdr = {"X-Localcode-Token": env["LOCALCODE_CONTROL_TOKEN"]}
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{cport}/status", headers=hdr), timeout=1); break
            except Exception:
                time.sleep(0.3)
        else:
            return None
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{cport}/quants?group={group}", headers=hdr), timeout=60) as r:
            data = json.load(r)
        rows = data.get("quants") or []
        # The picker marks its own choice; that is the quant a user of this tier gets.
        chosen = [q for q in rows if q.get("recommended")]
        if not chosen:
            return None
        return chosen[0]["alias"], chosen[0]["filename"]
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tiers", default="16,24,32,64,this")
    ap.add_argument("--models", default="", help="catalog group keys, default: every group")
    ap.add_argument("--set", default="regression")
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default: the set")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    py = repo / ".venv/bin/python" if (repo / ".venv/bin/python").exists() else Path(sys.executable)
    sys.path.insert(0, str(repo / "src"))
    from localcode.models_catalog import MODEL_GROUPS  # type: ignore
    groups = [g for g in MODEL_GROUPS if not a.models or g.key in a.models.split(",")]
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    plan = []
    for g in groups:
        for t in tiers:
            ram = None if t == "this" else int(t)
            pick = picked_alias(repo, py, g.key, ram)
            plan.append((g.key, t, pick))
            print(f"{g.key:22} tier {t:>4}: {pick[0] if pick else 'nothing fits'}", flush=True)
    if not a.execute:
        print("\nplan only; add --execute to run (hours)")
        return
    for key, t, pick in plan:
        if not pick:
            continue
        alias, _ = pick
        present = (Path.home() / ".local/share/localcode/models" / f"{alias}.gguf").exists()
        if not present:
            print(f"== skip {key} tier {t}: {alias} is not downloaded (no silent downloads)", flush=True)
            continue
        env = {**os.environ}
        if t != "this": env["LOCALCODE_RAM_GB"] = t
        if "compaction-depth" in (a.tasks or ""): env.setdefault("LOCALCODE_OVERRIDE_NCTX", "16384")
        cmd = [sys.executable, str(HERE / "run_tasks.py"), "--repo", str(repo), "--model", alias, "--set", a.set, "--trials", str(a.trials), "--label", f"tier{t}"]
        if a.tasks: cmd += ["--tasks", a.tasks]
        print(f"== {key} tier {t} -> {alias} {time.strftime('%H:%M')}", flush=True)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
