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
    GET  /progress           prompt-fill progress of the running request (from llama-server /slots)
    GET  /voice/status       {"ready", "recording", "detail"}   (whisper env + STT model)
    POST /voice/start        start recording the default mic (ffmpeg, 16 kHz mono wav)
    POST /voice/stop         stop, transcribe locally (whisper.cpp), {"text"}
    POST /voice/speak {"text"}   read text aloud with macOS `say`
    POST /models_dir {"path"}   change where GGUFs download to (persisted in localcode's config)

The inference port never changes across a switch, so the front end's
base_url stays valid; only the model alias changes.
"""
from __future__ import annotations

import json
import os
import shutil
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
    def progress(self) -> dict:
        """What the server is doing right now: reading the prompt (with a fraction) or
        generating. Lets the TUI show 'reading context 40%' instead of a bare spinner."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/slots", timeout=1) as r:
                slots = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            return {"phase": "unknown"}
        slot = next((x for x in slots if x.get("is_processing")), None)
        if slot is None:
            return {"phase": "idle"}
        total = int(slot.get("n_prompt_tokens") or 0)
        cache = int(slot.get("n_prompt_tokens_cache") or 0)
        done = int(slot.get("n_prompt_tokens_processed") or 0)
        nt = slot.get("next_token") or []
        decoded = int((nt[0] if isinstance(nt, list) and nt else nt or {}).get("n_decoded") or 0)
        todo = max(total - cache, 0)
        if decoded > 0 or (todo and done >= todo):
            return {"phase": "generating", "decoded": decoded, "prompt_tokens": total}
        if todo:
            return {"phase": "reading", "done": done, "todo": todo, "cached": cache, "pct": int(100 * done / todo)}
        return {"phase": "reading", "done": 0, "todo": 0, "cached": cache, "pct": 0}

    def vision_info(self) -> dict:
        """What the loaded model could see with, and whether it is on disk. Nothing
        is downloaded here: the user asks via POST /vision/install (the TUI's /vision)."""
        if not self.current:
            return {"vision": False, "vision_available": False, "vision_size_gb": 0.0}
        from server_cmd import catalog_choice, mmproj_for
        choice = catalog_choice(str(self.models_dir / f"{self.current}.gguf"))
        available = bool(choice is not None and getattr(choice, "supports_vision", False))
        return {"vision": mmproj_for(str(self.models_dir / f"{self.current}.gguf")) is not None,
                "vision_available": available,
                "vision_size_gb": round(float(getattr(choice, "mmproj_size_gb", 0.0) or 0.0), 2) if available else 0.0}

    def vision_install(self) -> dict:
        """User-requested: download the projector for the loaded model, then restart
        the server with it. Progress is in /status like any download."""
        if not self.current:
            return {"error": "no model loaded"}
        if not self.lock.acquire(blocking=False):
            return {"error": "a model switch is already in progress"}
        alias = self.current
        def work() -> None:
            try:
                self.active = {"group": None, "filename": None}
                self.ensure_mmproj(alias)
                self.state = {"state": "loading", "model": alias, "detail": "restarting with vision…", "pct": None}
                if self.start(alias):
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                else:
                    self.state = {"state": "error", "model": alias, "detail": "llama-server failed to restart", "pct": None}
            finally:
                self.active = {}
                self.lock.release()
        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    def ensure_mmproj(self, alias: str) -> None:
        """Fetch the model's vision projector (mmproj sidecar). Only called on an
        explicit user request (vision_install); failure only means text-only."""
        try:
            from server_cmd import catalog_choice
            choice = catalog_choice(str(self.models_dir / f"{alias}.gguf"))
            if choice is None or not getattr(choice, "supports_vision", False):
                return
            dest = choice.mmproj_path
            if dest is None:
                return
            if not dest.is_file():
                def prog(msg: str) -> None:
                    pct = None
                    if "(" in msg and "%)" in msg:
                        try: pct = int(msg.rsplit("(", 1)[1].split("%")[0])
                        except ValueError: pct = None
                    self.state = {"state": "downloading", "model": alias, "detail": f"vision projector: {msg}", "pct": pct}
                self.state = {"state": "downloading", "model": alias, "detail": "downloading vision projector…", "pct": None}
                ok, msg = bootstrap.download_mmproj(choice, on_progress=prog)
                if not ok:
                    print(f"supervisor: no vision projector for {alias}: {msg}", file=sys.stderr, flush=True)
            if dest.is_file() and dest.parent != self.models_dir and not (self.models_dir / dest.name).exists():
                os.symlink(dest, self.models_dir / dest.name)
        except Exception as e:  # noqa: BLE001
            print(f"supervisor: vision projector step failed: {e}", file=sys.stderr, flush=True)

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
                "license": g.license, "ram_gb": self.ram_gb, "quants": out,
                "vision_size_gb": round(float(getattr(g, "mmproj_size_gb", 0.0) or 0.0), 2) if getattr(g, "mmproj_filename", None) else 0.0}

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

    # ---- voice: push-to-talk STT + read-aloud, all local ---------------------
    voice_proc: "subprocess.Popen | None" = None
    voice_recorder = ""  # "portaudio" (bundled) or "ffmpeg" (fallback) while recording
    voice_wav = HERE / ".run" / "voice.wav"
    voice_detail = ""
    voice_setup_lock = threading.Lock()

    @staticmethod
    def _voice_venv() -> Path:
        return HERE / ".run" / "voice-venv"

    def _voice_python(self) -> Path | None:
        py = self._voice_venv() / "bin" / "python"
        if not py.exists():
            return None
        try:
            subprocess.run([str(py), "-c", "import pywhispercpp"], check=True, capture_output=True, timeout=60)
            return py
        except Exception:  # noqa: BLE001
            return None

    def _voice_recorder_ok(self) -> bool:
        """The bundled (PortAudio) recorder is installed in the venv. ffmpeg is only a
        fallback at record time, never a reason to skip installing this."""
        py = self._voice_venv() / "bin" / "python"
        if not py.exists():
            return False
        try:
            subprocess.run([str(py), "-c", "import sounddevice"], check=True, capture_output=True, timeout=30)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _voice_model_ready(self) -> bool:
        from localcode.voice import VoiceState, stt_model_ready
        return stt_model_ready(VoiceState())

    def voice_status(self) -> dict:
        from localcode.voice import DEFAULT_STT_MODEL_SIZE_MB
        runtime_ok = self._voice_python() is not None and self._voice_recorder_ok()
        model_ok = self._voice_model_ready()
        return {"ready": runtime_ok and model_ok,
                "setup_needed": not (runtime_ok and model_ok),
                "needs": {"runtime_mb": 0 if runtime_ok else 60, "model_mb": 0 if model_ok else int(DEFAULT_STT_MODEL_SIZE_MB)},
                "recording": self.voice_proc is not None and self.voice_proc.poll() is None,
                "recorder": self.voice_recorder,
                "detail": self.voice_detail}

    def voice_setup(self) -> tuple[bool, str]:
        """One-time: whisper.cpp wheel into a private venv + the STT model. User-triggered (/voice)."""
        with self.voice_setup_lock:
            if self._voice_python() is None or not self._voice_recorder_ok():
                self.voice_detail = "installing whisper.cpp + microphone recorder into .run/voice-venv…"
                venv = self._voice_venv()
                try:
                    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True, timeout=300)
                    subprocess.run([str(venv / "bin" / "pip"), "install", "-q", "pywhispercpp", "sounddevice"], check=True, capture_output=True, timeout=900)
                except Exception as e:  # noqa: BLE001
                    self.voice_detail = f"whisper install failed: {e}"
                    return False, self.voice_detail
            if not self._voice_model_ready():
                from localcode.voice import VoiceState, ensure_stt_model
                def prog(m: str) -> None:
                    self.voice_detail = f"downloading speech model: {m}"
                ok, msg = ensure_stt_model(VoiceState(), on_progress=prog)
                if not ok:
                    self.voice_detail = f"speech model download failed: {msg}"
                    return False, self.voice_detail
            self.voice_detail = ""
            return True, "ready"

    @staticmethod
    def _mic_device() -> str:
        """ffmpeg avfoundation audio device index: LOCALCODE_MIC, else the built-in mic, else 0."""
        env = os.environ.get("LOCALCODE_MIC")
        if env:
            return env
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                                 capture_output=True, text=True, timeout=15).stderr
            audio = out.split("audio devices:", 1)[1] if "audio devices:" in out else ""
            for line in audio.splitlines():
                if "[" in line and "]" in line and ("MacBook" in line or "Built-in" in line):
                    return line.rsplit("[", 1)[1].split("]")[0]
        except Exception:  # noqa: BLE001
            pass
        return "0"

    # Recorder that runs inside the voice venv: PortAudio ships in the sounddevice
    # wheel, so no ffmpeg/Homebrew is needed. Records 16 kHz mono until stdin closes.
    _RECORDER = """
import sys, wave, sounddevice as sd
path = sys.argv[1]
frames = []
def cb(indata, n, t, status):
    frames.append(bytes(indata))
with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', callback=cb):
    sys.stdout.write('recording\\n'); sys.stdout.flush()
    sys.stdin.read()
with wave.open(path, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b''.join(frames))
"""

    def voice_start(self) -> dict:
        if self.voice_proc is not None and self.voice_proc.poll() is None:
            return {"error": "already recording"}
        py = self._voice_python()
        if py is None or not self._voice_model_ready():
            return {"error": "setup_needed"}
        self.voice_wav.unlink(missing_ok=True)
        try:
            subprocess.run([str(py), "-c", "import sounddevice"], check=True, capture_output=True, timeout=30)
            cmd = [str(py), "-c", self._RECORDER, str(self.voice_wav)]
            self.voice_recorder = "portaudio"
        except Exception:  # noqa: BLE001
            if not shutil.which("ffmpeg"):
                return {"error": "no recorder available: run /voice setup again (installs the PortAudio recorder) or brew install ffmpeg"}
            self.voice_recorder = "ffmpeg"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-i", f":{self._mic_device()}",
                   "-ac", "1", "-ar", "16000", "-y", str(self.voice_wav)]
        self.voice_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.8)
        if self.voice_proc.poll() is not None:
            err = (self.voice_proc.stderr.read() if self.voice_proc.stderr else b"").decode(errors="replace").strip()
            self.voice_proc = None
            return {"error": f"could not open the microphone: {err or 'unknown error'} (grant microphone access to your terminal in System Settings → Privacy)"}
        return {"ok": True}

    def voice_stop(self) -> dict:
        p = self.voice_proc
        if p is None or p.poll() is not None:
            return {"error": "not recording"}
        try:
            p.stdin.write(b"q"); p.stdin.flush(); p.stdin.close()  # ffmpeg: 'q' quits; recorder: stdin EOF stops
        except Exception:  # noqa: BLE001
            p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        self.voice_proc = None
        if not self.voice_wav.exists() or self.voice_wav.stat().st_size < 1000:
            return {"error": "no audio captured"}
        return self.voice_transcribe(self.voice_wav)

    def voice_transcribe(self, wav: Path) -> dict:
        if self._voice_python() is None or not self._voice_model_ready():
            return {"error": "setup_needed"}
        py = self._voice_python()
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(HERE.parent / 'src')!r})\n"
            "from pathlib import Path\n"
            "from localcode.voice import VoiceState, transcribe\n"
            f"ok, text = transcribe(VoiceState(), Path({str(wav)!r}))\n"
            "print(json.dumps({'ok': ok, 'text': text}))\n"
        )
        try:
            r = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=300)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
            out = json.loads(line)
        except Exception as e:  # noqa: BLE001
            return {"error": f"transcription failed: {e}"}
        if not out.get("ok"):
            return {"error": out.get("text") or "transcription failed"}
        return {"ok": True, "text": out.get("text", "")}

    @staticmethod
    def voice_speak(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "nothing to say"}
        if not shutil.which("say"):
            return {"error": "macOS `say` not available"}
        subprocess.Popen(["say", text[:4000]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True}

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
                                       **sup.vision_info()))
            if u.path == "/models_dir":
                return self._json(sup.models_dir_info())
            if u.path == "/voice/status":
                return self._json(sup.voice_status())
            if u.path == "/progress":
                return self._json(sup.progress())
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
            if u.path == "/vision/install":
                res = sup.vision_install()
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/voice/setup":
                ok, msg = sup.voice_setup()
                return self._json({"ok": True} if ok else {"error": msg}, 200 if ok else 500)
            if u.path == "/voice/start":
                res = sup.voice_start()
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/voice/stop":
                res = sup.voice_stop()
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/voice/transcribe":
                res = sup.voice_transcribe(Path(str(body.get("path", ""))))
                return self._json(res, 200 if "error" not in res else 400)
            if u.path == "/voice/speak":
                res = sup.voice_speak(str(body.get("text", "")))
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
