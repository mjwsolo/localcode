#!/usr/bin/env python3
"""Drive the real localcode UI runtime headlessly against canned servers and
collect every model request plus the UI log. No model, no supervisor.

  run_checks.py <repo-root> <outdir> [--slots N] [--plugin path]

<repo-root> must have src/localcode/bin/localcode-ui; localcode must be importable by
LOCALCODE_PY, <repo-root>/.venv/bin/python, or this interpreter (in that order).
Writes <outdir>/req-NN.json, <outdir>/ui.log, <outdir>/events-turnN.jsonl, <outdir>/summary.json.
"""
import argparse, json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("out"); ap.add_argument("--slots", type=int, default=1); ap.add_argument("--plugin"); ap.add_argument("--keep", action="store_true")
a = ap.parse_args()
REPO = Path(a.repo).resolve(); OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("req-*.json"): f.unlink()
UI = REPO / "src/localcode/bin/localcode-ui"
# The python that can import localcode from <repo>/src: LOCALCODE_PY, else the
# checkout's .venv, else the interpreter running this script (CI installs -e .).
PY = Path(os.environ.get("LOCALCODE_PY") or (REPO / ".venv/bin/python" if (REPO / ".venv/bin/python").exists() else sys.executable))
HERE = Path(__file__).resolve().parent
home = Path(tempfile.mkdtemp(prefix="lc-check-home-")).resolve(); ws = Path(tempfile.mkdtemp(prefix="lc-check-ws-")).resolve()
(ws / "README.md").write_text("# demo\n\nA tiny project used by localcode Checks.\n")
(ws / "a.py").write_text("def add(a, b):\n    return a + b\n")
subprocess.run(["git", "init", "-q"], cwd=ws); subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c", "add", "-A"], cwd=ws); subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c", "commit", "-qm", "init"], cwd=ws)

def free_port():
    import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
mport, cport = free_port(), free_port(); alias = "canned-model"
env0 = {**os.environ, "CANNED_SLOTS": str(a.slots), "CANNED_WORKSPACE": str(ws)}
import atexit
srv_log = open(OUT / "servers.log", "ab")
model = subprocess.Popen([sys.executable, str(HERE / "canned_server.py"), str(mport), str(OUT)], env=env0, stdout=srv_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
ctrl = subprocess.Popen([sys.executable, str(HERE / "control_server.py"), str(cport), str(mport), alias], stdout=srv_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
def _cleanup():
    for p in (model, ctrl):
        if p.poll() is None:
            p.terminate()
            try: p.wait(timeout=3)
            except Exception: p.kill()
atexit.register(_cleanup)
def wait(url):
    for _ in range(50):
        try: urllib.request.urlopen(url, timeout=0.5); return
        except Exception: time.sleep(0.1)
    raise SystemExit(f"server not up: {url}")
wait(f"http://127.0.0.1:{mport}/health"); wait(f"http://127.0.0.1:{cport}/status")

cfg = home / "session.json"
code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(REPO / 'src')!r})
from localcode.ui.launch import write_config
write_config(Path({str(cfg)!r}), port={mport}, ctx=32768, alias={alias!r})
c = json.load(open({str(cfg)!r}))
plugin = {a.plugin!r}
if plugin: c["plugin"] = [plugin]
json.dump(c, open({str(cfg)!r}, "w"), indent=2)
"""
subprocess.run([str(PY), "-c", code], check=True)

env = {**os.environ, "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1", "HOME": str(home), "LOCALCODE_CONFIG": str(cfg), "LOCALCODE_CONTROL_URL": f"http://127.0.0.1:{cport}",
       "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "LOCALCODE_PARALLEL": str(a.slots), "LOCALCODE_TEST_HOME": str(home), "PWD": str(ws), "LOCALCODE_AGENT_RUN_DIR": str(home / "run")}
turns = ["Read README.md and tell me in one sentence what this project is.",
         "Now plan two steps with todowrite: add a docstring to add() in a.py, then run the tests. Do step 1."]
summary = {"turns": [], "slots": a.slots}
for i, text in enumerate(turns, 1):
    args = [str(UI), "run", "--format", "json", "-m", f"localcode/{alias}"] + (["--continue"] if i > 1 else []) + [text]
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=ws, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -1, (e.stdout or ""), (e.stderr or "") + "\nTIMEOUT"
    (OUT / f"events-turn{i}.jsonl").write_text(out); (OUT / f"stderr-turn{i}.txt").write_text(err)
    summary["turns"].append({"text": text, "rc": rc, "seconds": round(time.time() - t0, 1), "events": out.count("\n")})
log = next(iter((home / ".local/share/localcode-agent/log").glob("*.log")), None) if (home / ".local/share/localcode-agent/log").exists() else None
shutil.copy(log, OUT / "ui.log") if log else (OUT / "ui.log").write_text("")
plog = ws / ".localcode-agent/localcode-plugin.log"
shutil.copy(plog, OUT / "plugin.log") if plog.exists() else (OUT / "plugin.log").write_text("")
summary["requests"] = sorted(p.name for p in OUT.glob("req-*.json")); summary["workspace_files_changed"] = [l for l in subprocess.run(["git", "status", "--short"], cwd=ws, capture_output=True, text=True).stdout.strip().splitlines() if ".localcode-agent/" not in l]
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
if not a.keep: shutil.rmtree(home, ignore_errors=True); shutil.rmtree(ws, ignore_errors=True)
print(json.dumps(summary, indent=2))
