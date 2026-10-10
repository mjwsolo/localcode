#!/usr/bin/env python3
"""Run Tasks against the real localcode UI runtime and a real model.

  run_tasks.py --repo <checkout> --model <alias> [--tasks a,b,...] [--set regression|tracking|all]
               [--trials N] [--out runs/<label>] [--label <sha>]

One isolated supervisor + llama-server (throwaway HOME, spare ports, single slot) serves
every trial. Per trial: stage the task workspace as a fresh git repo, run each turn with
`localcode-ui run --format json`, write the .eval/ contract (diff.txt, final.txt,
commands.log, metrics.json, ui_errors.txt), run the hidden grader, append one row to
<out>/results.jsonl. Prints a table at the end.
"""
import argparse, json, os, re, shutil, signal, socket, subprocess, sys, tempfile, time, tomllib, urllib.request
from pathlib import Path
from performance import summarize, ResourceMeter
from process import run_bounded

HERE = Path(__file__).resolve().parent; TASKS = HERE / "tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tasks", default=""); ap.add_argument("--set", default="regression", choices=["regression", "tracking", "all"])
ap.add_argument("--trials", type=int, default=1); ap.add_argument("--out", default=""); ap.add_argument("--label", default="")
ap.add_argument("--keep", action="store_true")
a = ap.parse_args()
if a.trials < 1: ap.error("--trials must be positive")
REPO = Path(a.repo).resolve(); UI = REPO / "src/localcode/bin/localcode-ui"; SERVER = REPO / "src/localcode/bin/llama-server"
PY = Path(os.environ.get("LOCALCODE_PY") or (REPO / ".venv/bin/python" if (REPO / ".venv/bin/python").exists() else sys.executable))
label = a.label or subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip() or "local"
OUT = Path(a.out or (HERE / "runs" / f"{time.strftime('%Y%m%d-%H%M')}-{label}-{a.model}")).resolve(); OUT.mkdir(parents=True, exist_ok=False, mode=0o700)
HOME = OUT / "home"; HOME.mkdir(exist_ok=True); RUN_DIR = HOME / "run"; RUN_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path.home() / ".local/share/localcode/models"   # the real models, read-only for us
if Path(a.model).name != a.model or not (MODELS_DIR / f"{a.model}.gguf").is_file():
    sys.exit("--model must name an already-installed GGUF (without .gguf)")
# Only model files are shared; caches, history, configuration and warm-up state
# live in this run's own HOME. Do not link the user's entire application folder.
model_links = HOME / ".local/share/localcode/models"
model_links.mkdir(parents=True)
for model_file in MODELS_DIR.glob("*.gguf"):
    if model_file.is_file(): (model_links / model_file.name).symlink_to(model_file)

isolated_env = {"HOME": str(HOME), "XDG_CONFIG_HOME": str(HOME / ".config"),
                "XDG_DATA_HOME": str(HOME / ".local/share"), "XDG_CACHE_HOME": str(HOME / ".cache"),
                "XDG_STATE_HOME": str(HOME / ".local/state")}

def log(*x): print(time.strftime("%H:%M:%S"), *x, flush=True)
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
# Session secrets (ui/auth.py): the supervisor and the UI inherit them from our environment.
import secrets as _secrets
os.environ.setdefault("LOCALCODE_CONTROL_TOKEN", _secrets.token_urlsafe(32))
os.environ.setdefault("LOCALCODE_SERVER_KEY", _secrets.token_urlsafe(32))
_CTL = {"X-Localcode-Token": os.environ["LOCALCODE_CONTROL_TOKEN"]}
def get(url, timeout=2):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_CTL), timeout=timeout) as r: return json.load(r)
def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **_CTL})
    with urllib.request.urlopen(req, timeout=10) as r: return json.load(r)

# ── task selection ────────────────────────────────────────────────────────────────
wanted = set(filter(None, a.tasks.split(",")))
tasks = []
for d in sorted(p for p in TASKS.iterdir() if p.is_dir()):
    meta = tomllib.loads((d / "task.toml").read_text())
    if "kind=check" in meta.get("notes", ""): continue
    if wanted and d.name not in wanted: continue
    if not wanted and a.set != "all" and meta["purpose"] != a.set: continue
    tasks.append((d, meta))
if not tasks or (wanted and wanted != {d.name for d, _ in tasks}):
    sys.exit("select valid model tasks; empty/unknown/check-only task selections are rejected")


# ── one supervisor + model for the whole run ──────────────────────────────────────
mport, cport = free_port(), free_port()
sup_env = {**os.environ, **isolated_env, "PYTHONPATH": str(REPO / "src"), "HOME": str(HOME), "LOCALCODE_AGENT_RUN_DIR": str(RUN_DIR), "LOCALCODE_PARALLEL": "1"}
sup_log = open(OUT / "supervisor.log", "ab")
sup = subprocess.Popen([str(PY), "-m", "localcode.ui.supervisor", "--model", a.model, "--port", str(mport), "--control-port", str(cport),
                        "--server", str(SERVER), "--models-dir", str(model_links), "--parent-pid", str(os.getpid())],
                       env=sup_env, stdout=sup_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
import atexit
def _stop():
    if sup.poll() is None:
        sup.terminate()
        try: sup.wait(timeout=15)
        except Exception: sup.kill()
atexit.register(_stop)
for _ in range(100):
    try: get(f"http://127.0.0.1:{cport}/status"); break
    except Exception: time.sleep(0.3)
else: sys.exit("supervisor did not start (is another localcode running under this HOME?)")
log(f"loading installed {a.model} on port {mport}")
for _ in range(600):
    if sup.poll() is not None: sys.exit("supervisor exited; inspect supervisor.log")
    st = get(f"http://127.0.0.1:{cport}/status")
    if st.get("state") == "ready" and st.get("current") == a.model: break
    if st.get("state") == "error": sys.exit(f"model failed to load: {st}")
    time.sleep(1)
else: sys.exit("model did not become ready")
time.sleep(3)
props_req = urllib.request.Request(f"http://127.0.0.1:{mport}/props", headers={"Authorization": "Bearer " + os.environ["LOCALCODE_SERVER_KEY"]})
try:
    with urllib.request.urlopen(props_req, timeout=5) as response: props = json.load(response)
except (OSError, ValueError):
    props = {}
(OUT / "runtime.json").write_text(json.dumps({"model": a.model, "context": st.get("ctx"), "slots": props.get("total_slots"), "generation_context": (props.get("default_generation_settings") or {}).get("n_ctx")}, indent=2))
binary_sha = subprocess.run(["shasum", "-a", "256", str(UI)], capture_output=True, text=True).stdout[:12]
cfg = HOME / "session.json"
subprocess.run([str(PY), "-c", f"import sys; sys.path.insert(0, {str(REPO / 'src')!r}); from pathlib import Path; from localcode.ui.launch import write_config; write_config(Path({str(cfg)!r}), port={mport}, ctx={int(st.get('ctx') or 32768)}, alias={a.model!r})"], check=True)

def turns_of(task: Path):
    text = (task / "instruction.md").read_text()
    if "## Turn 1" in text:
        return [s.strip() for s in re.split(r"^## Turn \d+\s*$", text, flags=re.M) if s.strip()]
    return [text.strip()]

def server_prompt_tokens():
    """(count, list of prompt-eval token counts) from the shared server log so far."""
    f = RUN_DIR / "server.log"
    if not f.exists(): return []
    return [int(m) for m in re.findall(r"prompt eval time = +[0-9.]+ ms / +(\d+) tokens", f.read_text(errors="replace"))]

def server_prompt_lengths():
    """Prompt length at the end of every request so far (the server's n_tokens line)."""
    f = RUN_DIR / "server.log"
    if not f.exists(): return []
    return [int(m) for m in re.findall(r"stop processing: n_tokens = (\d+)", f.read_text(errors="replace"))]

def control_budget():
    try:
        j = get(f"http://127.0.0.1:{cport}/status")
        return {"ctx": j.get("ctx"), "budget": j.get("budget"), "budget_info": j.get("budget_info")}
    except Exception:
        return {}

def _write(path: Path, text: str) -> None:
    """The .eval/ folder lives in the workspace, and a model told to clean caches
    may delete it mid-run (seen on forbidden-commands). Recreate it on every write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_trial(task: Path, meta: dict, trial: int) -> dict:
    ws = Path(tempfile.mkdtemp(prefix=f"lc-task-{task.name}-")).resolve()
    shutil.copytree(task / "workspace", ws, dirs_exist_ok=True)
    g = lambda *x: subprocess.run(["git", "-c", "user.email=e@x", "-c", "user.name=e", *x], cwd=ws, capture_output=True)
    g("init", "-q"); g("add", "-A"); g("commit", "-qm", "init", "--allow-empty")
    ev = ws / ".eval"; ev.mkdir()
    env = {**os.environ, **isolated_env, "HOME": str(HOME), "LOCALCODE_TEST_HOME": str(HOME), "PWD": str(ws), "LOCALCODE_CONFIG": str(cfg),
           "LOCALCODE_CONTROL_URL": f"http://127.0.0.1:{cport}", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "LOCALCODE_PARALLEL": "1",
           "LOCALCODE_AGENT_RUN_DIR": str(RUN_DIR)}
    steps = tool_calls = 0; commands = []; final = ""; reread = []; wall = 0.0; ui_errors = []; timed_out = False; runtime_failed = False; calls_per_turn = []
    lengths_before = len(server_prompt_lengths()); budgets = []
    server_log = RUN_DIR / "server.log"
    log_offset = server_log.stat().st_size if server_log.exists() else 0
    all_events = []
    ui_log_dir = HOME / ".local/share/localcode-agent/log"
    ui_offsets = {p: p.stat().st_size for p in ui_log_dir.glob("*.log")}
    meter = ResourceMeter(sup.pid)
    atexit.register(meter.finish)
    budget = int(meta.get("budget_seconds", 900))
    for i, text in enumerate(turns_of(task), 1):
        before = len(server_prompt_tokens())
        args = [str(UI), "run", "--format", "json", "-m", f"localcode/{a.model}"] + (["--continue"] if i > 1 else []) + [text]
        t0 = time.time()
        try:
            r = run_bounded(args, cwd=ws, env=env, text=True, timeout=budget - wall, stdin=subprocess.DEVNULL)
            out = r.stdout
            runtime_failed = runtime_failed or r.returncode != 0
            (ev / f"stderr-turn{i}.txt").write_text(r.stderr)
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""; timed_out = True
            if isinstance(out, bytes): out = out.decode(errors="replace")
        wall += time.time() - t0
        # The client exits on the last streamed token; the server prints that
        # request's timing and release lines a few ms later. Wait for the log to
        # settle so the request is attributed to this turn, not the next.
        for _ in range(60):
            txt = (RUN_DIR / "server.log").read_text(errors="replace") if (RUN_DIR / "server.log").exists() else ""
            if txt.count("launch_slot_: id") <= txt.count("stop processing: n_tokens"): break
            time.sleep(0.05)
        _write(ev / f"events-turn{i}.jsonl", out)
        turn_text = []; turn_calls = 0
        for line in out.splitlines():
            try: e = json.loads(line)
            except Exception: continue
            all_events.append(e)
            t = e.get("type"); p = e.get("part") or {}
            if t == "step_start": steps += 1
            elif t == "tool_use":
                tool_calls += 1; turn_calls += 1
                if p.get("tool") == "bash": commands.append(str((p.get("state") or {}).get("input", {}).get("command", "")))
            elif t == "text": turn_text.append(str(p.get("text") or e.get("text") or ""))
        final = "\n".join(turn_text).strip() or final
        calls_per_turn.append(turn_calls)
        _write(ev / f"final_turn{i}.txt", "\n".join(turn_text).strip())
        toks = server_prompt_tokens()[before:]
        agent_toks = [n for n in toks if n > 20]        # drop the tiny title/side requests
        reread.append(agent_toks[0] if agent_toks else 0)
        budgets.append((control_budget() or {}).get("budget"))
        if timed_out: break
    resources = meter.finish()
    atexit.unregister(meter.finish)
    # the .eval contract
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ws, capture_output=True, text=True).stdout
    paths = sorted({l[3:].split(" -> ")[-1] for l in st.splitlines() if l.strip() and not l[3:].startswith(".eval/") and not l[3:].startswith(".localcode-agent/") and not re.search(r"(^|/)(__pycache__/|\.pytest_cache/|\.mypy_cache/)|\.pyc$", l[3:])})
    _write(ev / "diff.txt", "\n".join(paths) + ("\n" if paths else ""))
    _write(ev / "final.txt", final); _write(ev / "commands.log", "\n".join(commands) + ("\n" if commands else ""))
    ui_text = ""
    for path in ui_log_dir.glob("*.log"):
        with path.open("rb") as stream:
            stream.seek(ui_offsets.get(path, 0))
            ui_text += stream.read().decode(errors="replace")
    ui_errors = [line for line in ui_text.splitlines() if "level=ERROR" in line]
    (ev / "ui_errors.txt").write_text("\n".join(ui_errors))
    lengths = server_prompt_lengths()[lengths_before:]
    compactions = sum("agent=compaction" in line and "message=stream" in line for line in ui_text.splitlines())
    with server_log.open("rb") as stream:
        stream.seek(log_offset)
        performance = summarize(all_events, stream.read().decode(errors="replace"))
    metrics = {"steps": steps, "tool_calls": tool_calls, "tool_calls_per_turn": calls_per_turn, "wall_s": round(wall, 1), "reread_tokens": reread,
               "prompt_tokens_max": max(lengths) if lengths else 0, "prompt_tokens_by_request": lengths, "budget_by_turn": budgets,
               **resources, **performance, "compactions": compactions, "context": control_budget(), "timed_out": timed_out, "runtime_failed": runtime_failed}
    (ev / "metrics.json").write_text(json.dumps(metrics))
    gr = subprocess.run(["bash", str(task / "tests/test.sh")], env={**os.environ, "WORKSPACE": str(ws), "TASK_DIR": str(task)}, capture_output=True, text=True, timeout=300)
    reward = json.loads((ev / "reward.json").read_text()) if (ev / "reward.json").exists() else {"pass": False, "notes": f"grader wrote no reward (rc={gr.returncode})"}
    keep = OUT / "trials" / f"{task.name}-{trial}"; keep.parent.mkdir(exist_ok=True)
    shutil.copytree(ev, keep, dirs_exist_ok=True)
    if paths: (keep / "changed.diff").write_text(subprocess.run(["git", "diff"], cwd=ws, capture_output=True, text=True).stdout)
    if not a.keep: shutil.rmtree(ws, ignore_errors=True)
    return {"label": label, "task": task.name, "quality": meta["quality"], "component": meta.get("component", "harness"), "purpose": meta["purpose"],
            "model": a.model, "trial": trial, "pass": bool(reward.get("pass")) and not timed_out and not runtime_failed and gr.returncode == 0, "notes": reward.get("notes", ""), **metrics, "binary_sha": binary_sha, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

rows = []
with open(OUT / "results.jsonl", "a") as f:
    for task, meta in tasks:
        for trial in range(1, max(a.trials, int(meta.get("trials", 1))) + 1):
            try:
                row = run_trial(task, meta, trial)
            except Exception as e:  # noqa: BLE001 - one broken trial must not end the run
                row = {"label": label, "task": task.name, "quality": meta["quality"], "component": meta.get("component", "harness"), "purpose": meta["purpose"],
                       "model": a.model, "trial": trial, "pass": False, "notes": f"runner error: {type(e).__name__}: {e}"[:300], "steps": 0, "tool_calls": 0,
                       "wall_s": 0.0, "reread_tokens": [], "binary_sha": binary_sha, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            rows.append(row); f.write(json.dumps(row) + "\n"); f.flush()
            log(f"{task.name:<20} trial {trial}  {'PASS' if row['pass'] else 'FAIL'}  steps={row['steps']} tools={row['tool_calls']} wall={row['wall_s']}s reread={row['reread_tokens']}  {row['notes'][:70]}")
by = {}
for r in rows: by.setdefault(r["task"], []).append(r["pass"])
print(f"\n{'task':<20}{'trials':>7}{'pass@k':>8}{'pass^k':>8}")
for t, ps in by.items(): print(f"{t:<20}{len(ps):>7}{('yes' if any(ps) else 'no'):>8}{('yes' if all(ps) else 'no'):>8}")
tot = sum(r["pass"] for r in rows); print(f"\n{tot}/{len(rows)} trials passed  ->  {OUT / 'results.jsonl'}")
