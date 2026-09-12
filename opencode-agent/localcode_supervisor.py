#!/usr/bin/env python3
"""localcode's model supervisor for the fork-based front ends (codex, opencode).

Owns the bundled llama-server and exposes a tiny localhost control API that
the in-TUI `/model` picker talks to. The picker itself is NOT reimplemented
here: the catalog comes from localcode's own modules (MODEL_GROUPS, recommend,
hf_quants.fetch_quants, fit_badge, estimate_decode_tok_s) and downloads flow
through bootstrap.download_model — the same code the Textual picker uses.

    GET  /catalog            level 1: every catalog model, ★ from recommend()
    GET  /quants?group=KEY   level 2: every quant the HF repo ships
    POST /select {"group","filename"}   download if needed, then restart the
                                        server on the SAME port with that gguf
    GET  /status             {"state": idle|downloading|loading|ready|error, ...}
    POST /cancel             stop the download in progress (the .part is discarded)
    GET  /models_dir         {"path", "free_gb"}
    POST /models_dir {"path"}   change where GGUFs download to (persisted in localcode's config)

The inference port never changes across a switch, so the front end's
base_url stays valid; only the model alias changes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localcode.models_catalog import (  # noqa: E402
    MODEL_GROUPS, choice_for_quant, estimate_decode_tok_s, recommend,
    _system_ram_gb, model_dir,
)
from localcode.hf_quants import fetch_quants, fit_badge  # noqa: E402
from localcode import bootstrap  # noqa: E402

MIN_SPEED_FRACTION = 0.5  # same rule as tui/screens/model_picker.py
HERE = Path(__file__).resolve().parent


def _bandwidth() -> float:
    try:
        from localcode.performance import apple_silicon_bandwidth_gbps
        return apple_silicon_bandwidth_gbps()
    except Exception:  # noqa: BLE001
        return 150.0


def _alias(filename: str) -> str:
    return filename[:-5] if filename.endswith(".gguf") else filename


class Supervisor:
    def __init__(self, server_bin: str, port: int, models_dir: Path, ctx: int) -> None:
        self.server_bin, self.port, self.models_dir, self.ctx = server_bin, port, models_dir, ctx
        self.proc: subprocess.Popen | None = None
        self.current: str | None = None           # alias of the loaded model
        self.lock = threading.Lock()
        self.state = {"state": "idle", "model": None, "detail": "", "pct": None}
        self.cancel = threading.Event()
        self.active: dict = {}  # {"group": key, "filename": ..} while a download/switch runs
        self.ram_gb = _system_ram_gb()
        self.bandwidth = _bandwidth()
        self.log = open(HERE / ".run" / "server.log", "ab", buffering=0)

    # ---- llama-server lifecycle -------------------------------------------
    def start(self, alias: str, wait_s: int = 240) -> bool:
        gguf = self.models_dir / f"{alias}.gguf"
        self.stop()
        # localcode's own per-machine server command (RAM-tier context, KV
        # compression, flash-attn, checkpoints) — nothing hardcoded here.
        from server_cmd import server_command
        cmd = server_command(str(gguf), self.port, alias)
        cmd[0] = self.server_bin
        self.ctx = int(cmd[cmd.index("--ctx-size") + 1])
        self.log.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n".encode())
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(f"supervisor: started llama-server pid {self.proc.pid} for {alias}", file=sys.stderr, flush=True)
        for _ in range(wait_s):
            if self.proc.poll() is not None:
                return False
            if self.healthy():
                self.current = alias
                return True
            time.sleep(1)
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            print(f"supervisor: stopping llama-server pid {self.proc.pid}", file=sys.stderr, flush=True)
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    # ---- catalog ------------------------------------------------------------
    def _rec_repo(self) -> str | None:
        try:
            return recommend(self.ram_gb).hf_repo
        except Exception:  # noqa: BLE001
            return None

    def _current_repo(self) -> str | None:
        if not self.current:
            return None
        from localcode.models_catalog import group_for_filename
        g = group_for_filename(self.current + ".gguf")
        return g.hf_repo if g else None

    def catalog(self) -> dict:
        rec, cur = self._rec_repo(), self._current_repo()
        groups = []
        for g in MODEL_GROUPS:
            groups.append({
                "key": g.key, "display_name": g.display_name, "maker": g.maker,
                "license": g.license, "hf_repo": g.hf_repo,
                "recommended": g.hf_repo == rec, "current": g.hf_repo == cur,
                "downloading": self.state["state"] == "downloading" and self.active.get("group") == g.key,
                "pct": self.state["pct"] if self.active.get("group") == g.key else None,
            })
        return {"ram_gb": self.ram_gb, "current": self.current, "groups": groups,
                "models_dir": str(self.models_dir)}

    def quants(self, key: str) -> dict:
        g = next((g for g in MODEL_GROUPS if g.key == key), None)
        if g is None:
            return {"error": f"unknown group {key}"}
        rows = [q for q in fetch_quants(g.hf_repo) if not q.is_mmproj]
        rows.sort(key=lambda q: q.size_gb)
        rec_idx = self._recommended_quant_idx(rows, g.display_name)
        out = []
        for i, q in enumerate(rows):
            spd = estimate_decode_tok_s(q.size_gb, g.display_name, self.bandwidth)
            out.append({
                "filename": q.filename, "alias": _alias(q.filename), "label": q.label,
                "size_gb": round(q.size_gb, 1), "fit": fit_badge(q.size_gb, self.ram_gb),
                "tok_s": spd, "recommended": i == rec_idx,
                "downloaded": (self.models_dir / q.filename).exists(),
                "current": _alias(q.filename) == self.current,
                "downloading": self.state["state"] == "downloading" and self.active.get("filename") == q.filename,
                "pct": self.state["pct"] if self.active.get("filename") == q.filename else None,
            })
        return {"group": g.key, "display_name": g.display_name, "maker": g.maker,
                "license": g.license, "ram_gb": self.ram_gb, "quants": out}

    def _recommended_quant_idx(self, rows, name: str) -> int | None:
        if not rows:
            return None
        fitting = [i for i, q in enumerate(rows) if q.size_gb <= 0.55 * self.ram_gb]
        if not fitting:
            return 0
        speeds = {i: (estimate_decode_tok_s(rows[i].size_gb, name, self.bandwidth) or 0)
                  for i in fitting}
        fastest = max(speeds.values()) or 0
        if fastest <= 0:
            return max(fitting, key=lambda i: rows[i].size_gb)
        bar = fastest * MIN_SPEED_FRACTION
        responsive = [i for i in fitting if speeds[i] >= bar]
        return max(responsive or fitting, key=lambda i: rows[i].size_gb)

    # ---- select (download + switch), runs on its own thread ------------------
    def select(self, key: str, filename: str) -> dict:
        g = next((g for g in MODEL_GROUPS if g.key == key), None)
        if g is None:
            return {"error": f"unknown group {key}"}
        if not self.lock.acquire(blocking=False):
            return {"error": "a model switch is already in progress"}
        alias = _alias(filename)
        self.cancel.clear()
        self.active = {"group": key, "filename": filename}
        self.state = {"state": "downloading" if not (self.models_dir / filename).exists() else "loading",
                      "model": alias, "detail": "", "pct": None}
        threading.Thread(target=self._select_worker, args=(g, filename, alias), daemon=True).start()
        return {"ok": True, "model": alias, "state": self.state["state"]}

    def _select_worker(self, g, filename: str, alias: str) -> None:
        try:
            if not (self.models_dir / filename).exists():
                size = next((q.size_gb for q in fetch_quants(g.hf_repo) if q.filename == filename), 0.0)
                choice = choice_for_quant(g, filename, size)

                def on_progress(msg: str) -> None:
                    pct = None
                    if "(" in msg and "%)" in msg:
                        try:
                            pct = int(msg.rsplit("(", 1)[1].split("%")[0])
                        except ValueError:
                            pct = None
                    self.state = {"state": "downloading", "model": alias, "detail": msg, "pct": pct}

                ok, res = bootstrap.download_model(choice, on_progress=on_progress, cancel_event=self.cancel)
                if self.cancel.is_set():
                    for part in (self.models_dir / filename, choice.local_path):
                        for cand in (part.with_suffix(part.suffix + ".part"), Path(str(part) + ".part")):
                            try: cand.unlink()
                            except FileNotFoundError: pass
                    self.state = {"state": "idle", "model": self.current, "detail": "download cancelled", "pct": None}
                    return
                if not ok:
                    self.state = {"state": "error", "model": alias, "detail": res, "pct": None}
                    return
                got = Path(res)
                if got.parent != self.models_dir and not (self.models_dir / filename).exists():
                    # download_model saved under localcode's model_dir(); link it here.
                    os.symlink(got, self.models_dir / filename)
            # Vision projector (mmproj sidecar): small, one per family; without it the
            # server cannot take images. Failure here only means text-only.
            try:
                from server_cmd import catalog_choice
                choice = catalog_choice(str(self.models_dir / filename))
                if choice is not None and getattr(choice, "supports_vision", False):
                    dest = choice.mmproj_path
                    if dest is None or not dest.is_file():
                        self.state = {"state": "downloading", "model": alias, "detail": "downloading vision projector…", "pct": None}
                        ok, msg = bootstrap.download_mmproj(choice, on_progress=lambda m: None)
                        if not ok:
                            print(f"supervisor: no vision projector for {alias}: {msg}", file=sys.stderr, flush=True)
                    if dest is not None and dest.is_file() and not (self.models_dir / dest.name).exists() and dest.parent != self.models_dir:
                        os.symlink(dest, self.models_dir / dest.name)
            except Exception as e:  # noqa: BLE001
                print(f"supervisor: vision projector step failed: {e}", file=sys.stderr, flush=True)
            self.state = {"state": "loading", "model": alias, "detail": "loading model…", "pct": None}
            if self.start(alias):
                self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
            else:
                self.state = {"state": "error", "model": alias,
                              "detail": "llama-server failed to load the model (see .run/server.log)",
                              "pct": None}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.state = {"state": "error", "model": alias, "detail": str(e), "pct": None}
        finally:
            self.active = {}
            print(f"supervisor: switch to {alias} -> {self.state['state']}", file=sys.stderr, flush=True)
            self.lock.release()

    def vision(self) -> bool:
        if not self.current:
            return False
        from server_cmd import mmproj_for
        return mmproj_for(str(self.models_dir / f"{self.current}.gguf")) is not None

    def cancel_download(self) -> dict:
        if self.state["state"] != "downloading":
            return {"error": "no download in progress"}
        self.cancel.set()
        return {"ok": True}

    def models_dir_info(self) -> dict:
        try:
            st = os.statvfs(self.models_dir if self.models_dir.exists() else self.models_dir.parent)
            free = round(st.f_bavail * st.f_frsize / 1e9, 1)
        except OSError:
            free = None
        return {"path": str(self.models_dir), "free_gb": free}

    def set_models_dir(self, raw: str) -> dict:
        if self.state["state"] in ("downloading", "loading"):
            return {"error": "wait for the current download/switch to finish"}
        path = Path(os.path.expanduser(raw.strip())).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            (path / ".localcode-write-test").touch(); (path / ".localcode-write-test").unlink()
        except OSError as e:
            return {"error": f"cannot use {path}: {e}"}
        self.models_dir = path
        # localcode's own download path reads model_dir() -> config runtime.model_dir /
        # LOCALCODE_MODEL_DIR, so persist it where every front end will see it.
        os.environ["LOCALCODE_MODEL_DIR"] = str(path)
        try:
            from localcode.config import load_config, save_config
            cfg = load_config(); cfg.runtime.model_dir = str(path); save_config(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"supervisor: could not persist model_dir: {e}", file=sys.stderr, flush=True)
        return dict(self.models_dir_info(), ok=True)


def make_handler(sup: Supervisor):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/catalog":
                return self._json(sup.catalog())
            if u.path == "/quants":
                key = parse_qs(u.query).get("group", [""])[0]
                return self._json(sup.quants(key))
            if u.path == "/status":
                return self._json(dict(sup.state, current=sup.current, port=sup.port, ctx=sup.ctx,
                                       group=sup.active.get("group"), filename=sup.active.get("filename"),
                                       vision=sup.vision()))
            if u.path == "/models_dir":
                return self._json(sup.models_dir_info())
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if u.path == "/select":
                res = sup.select(str(body.get("group", "")), str(body.get("filename", "")))
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/cancel":
                res = sup.cancel_download()
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/models_dir":
                res = sup.set_models_dir(str(body.get("path", "")))
                return self._json(res, 200 if "error" not in res else 400)
            self._json({"error": "not found"}, 404)
    return H


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="alias (gguf filename without .gguf); empty = start idle, the TUI picks")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--control-port", type=int, required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--models-dir", default=os.environ.get(
        "LOCALCODE_MODELS_DIR", str(Path.home() / ".local/share/localcode/models")))
    a = ap.parse_args()

    (HERE / ".run").mkdir(exist_ok=True)
    sup = Supervisor(a.server, a.port, Path(a.models_dir), 0)
    httpd = ThreadingHTTPServer(("127.0.0.1", a.control_port), make_handler(sup))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def bye(signum, frame):
        import traceback
        print(f"supervisor: got signal {signum}, exiting", file=sys.stderr)
        traceback.print_stack(frame, file=sys.stderr)
        sup.stop()
        httpd.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGHUP, bye)
    import atexit
    atexit.register(sup.stop)

    if a.model:
        sup.state = {"state": "loading", "model": a.model, "detail": "", "pct": None}
        if not sup.start(a.model):
            print(f"llama-server failed to load {a.model}", file=sys.stderr)
            return 1
        sup.state = {"state": "ready", "model": a.model, "detail": "", "pct": None}
    else:
        # First-run journey: the TUI opens first and its /models picker loads a model.
        sup.state = {"state": "idle", "model": None, "detail": "no model loaded", "pct": None}
    print("ready", flush=True)
    # NOT signal.pause(): it returns on ANY signal, including the SIGCHLD from
    # a llama-server we just stopped, which made the supervisor exit mid-switch.
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
