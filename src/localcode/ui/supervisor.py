"""localcode's model supervisor for the fork-based front ends (codex, opencode).

Owns the bundled llama-server and exposes a tiny localhost control API that
the in-TUI `/model` picker talks to. The picker itself is NOT reimplemented
here: the catalog comes from localcode's own modules (MODEL_GROUPS, recommend,
hf_quants.fetch_quants, quant_plan, estimate_decode_tok_s) and downloads flow
through bootstrap.download_model — the same code the Textual picker uses.

    GET  /catalog            level 1: every catalog model, ★ from recommend()
    GET  /quants?group=KEY   level 2: every quant the HF repo ships
    POST /select {"group","filename"}   download if needed, then restart the
                                        server on the SAME port with that gguf
    GET  /status             {"state": idle|downloading|loading|ready|error, ...}
    POST /cancel             stop the download in progress (the .part is discarded)
    POST /warmup/cancel      stop the prompt warm-up prefill (the user just sent a turn)
    POST /vision/load        an image was sent: restart with the (downloaded) projector
    GET  /models_dir         {"path", "free_gb"}
    GET  /progress           prompt-fill progress of the running request (from llama-server /slots)
    GET  /voice/status       {"ready", "recording", "detail"}   (whisper env + STT model)
    POST /voice/start        open the default mic (PortAudio, 16 kHz mono) and start live dictation
    GET  /voice/partial      {"text", "recording"}: what has been heard so far, while recording
    POST /voice/stop         stop, {"text"}: the final transcript (whisper.cpp, local)
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
from localcode.ui.http import local_urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import localcode
from localcode import bootstrap
from localcode.model_files import model_files, find_model
from localcode.hf_quants import fetch_quants
from localcode.models_catalog import (
    MODEL_GROUPS,
    _system_ram_gb,
    choice_for_quant,
    estimate_decode_tok_s,
    is_degraded,
    quant_plan,
    recommend,
    recommended_quant_index,
)

MIN_SPEED_FRACTION = 0.5  # same rule as tui/screens/model_picker.py
from localcode.ui import run_dir
from localcode.ui.auth import KEY_ENV, control_token, new_secret, request_allowed, server_headers, server_key, write_auth_file

HERE = run_dir()


def log(message: str) -> None:
    """Diagnostics must never paint over the terminal UI."""
    try:
        HERE.mkdir(parents=True, exist_ok=True)
        with (HERE / "supervisor.log").open("a") as stream:
            stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def _bandwidth() -> float:
    try:
        from localcode.performance import apple_silicon_bandwidth_gbps
        return apple_silicon_bandwidth_gbps()
    except Exception:  # noqa: BLE001
        return 150.0


def _alias(filename: str) -> str:
    return filename.removesuffix(".gguf")


class Supervisor:
    def __init__(self, server_bin: str, port: int, models_dir: Path, ctx: int) -> None:
        self.server_bin, self.port, self.models_dir, self.ctx = server_bin, port, models_dir, ctx
        self.proc: subprocess.Popen | None = None
        self.current: str | None = None           # alias of the loaded model
        self.lock = threading.Lock()
        self._process_lock = threading.RLock()
        self._closing = threading.Event()
        self.speech_lock = threading.Lock()
        self.speech_proc: subprocess.Popen | None = None
        self.state = {"state": "idle", "model": None, "detail": "", "pct": None}
        self.cancel = threading.Event()
        self.active: dict = {}  # {"group": key, "filename": ..} while a download/switch runs
        self.ram_gb = _system_ram_gb()
        self.bandwidth = _bandwidth()
        # Lazy vision projector: the server starts text-only; the first image
        # in a session asks for the projector (POST /vision/load) and the
        # server restarts with it for the rest of that model's load.
        self.vision_wanted: str | None = None   # alias whose projector was asked for
        self.vision_loaded = False              # the running server has --mmproj
        self.log = open(HERE / "server.log", "ab", buffering=0)  # noqa: SIM115 (lives with the server)
        self._drop_retired_sidecars()
        # Context budget (ui/context_budget.py): the per-slot context the server
        # really loaded, and how much of it this machine can use at speed.
        self.ctx_total = ctx
        self.slots = 1
        self.pp_prior: float | None = None     # prefill tokens/s from the warm-up replay
        self._timing = None                    # context_budget.TimingTable for the loaded model
        self._log_offset = 0                   # server.log bytes already fed to it
        self._budget: int | None = None
        self._budget_info: dict = {}
        self._budget_lock = threading.Lock()
        self._warm_stop = threading.Event()
        self._warm_status = "idle"
        self._context_ready = False
        self._warm_replay = None  # warmup.Replay while the prefix is being pre-read

    # ---- llama-server lifecycle -------------------------------------------
    def _drop_retired_sidecars(self) -> None:
        """Delete drafter files an earlier release downloaded that the catalog
        has since replaced. Only names in models_catalog.RETIRED_SIDECARS."""
        try:
            removed = bootstrap.remove_retired_sidecars(self.models_dir)
        except Exception as e:  # noqa: BLE001
            log(f"supervisor: retired sidecar cleanup skipped: {e}")
            return
        for name in removed:
            log(f"supervisor: removed retired drafter {name} (replaced by a newer catalog drafter)")

    def _server_failure(self) -> dict:
        """Describe an unexpected exit without treating an intentional stop as a crash."""
        process = getattr(self, "proc", None)
        if process is None or getattr(self, "state", {}).get("state") != "ready":
            return {}
        code = process.poll()
        if code is None:
            return {}
        name = None
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
        reason = name or f"exit code {code}"
        verb = "crashed" if code < 0 else "exited unexpectedly"
        return {"exit_code": code, "exit_signal": name,
                "process_id": getattr(process, "pid", None),
                "detail": f"Model server {verb} ({reason}). Open /server to restart or choose another model. See server.log for diagnostics."}

    def progress(self) -> dict:
        """What the server is doing right now: reading the prompt (with a fraction) or
        generating. Lets the TUI show 'reading context 40%' instead of a bare spinner."""
        failure = self._server_failure()
        if failure:
            return {"phase": "error", **failure}
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/slots", headers=server_headers())
            with local_urlopen(req, timeout=1) as r:
                slots = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            return {"phase": "unknown"}
        slot = next((x for x in slots if x.get("is_processing")), None)
        if slot is None:
            return {"phase": "idle"}
        request = {"request_id": f"{slot.get('id')}:{slot.get('id_task')}",
                   "warmup": getattr(self, "_warm_replay", None) is not None,
                   "context_ready": getattr(self, "_context_ready", False)}
        total = int(slot.get("n_prompt_tokens") or 0)
        cache = int(slot.get("n_prompt_tokens_cache") or 0)
        done = int(slot.get("n_prompt_tokens_processed") or 0)
        nt = slot.get("next_token") or []
        decoded = int((nt[0] if isinstance(nt, list) and nt else nt or {}).get("n_decoded") or 0)
        todo = max(total - cache, 0)
        if decoded > 0 or (todo and done >= todo):
            if slot.get("id") == 0 and not request["warmup"]:
                self._context_ready = True
                request["context_ready"] = True
            return {"phase": "generating", "decoded": decoded, "prompt_tokens": total, **request}
        if todo:
            return {"phase": "reading", "done": done, "todo": todo, "cached": cache, "pct": int(100 * done / todo), **request}
        return {"phase": "reading", "done": 0, "todo": 0, "cached": cache, "pct": 0, **request}

    def vision_info(self) -> dict:
        """What the loaded model could see with, and whether it is on disk. Nothing
        is downloaded here: the user asks via POST /vision/install (the TUI's /vision).

        `vision` means images work: the projector is on disk and loads with the
        first image (`vision_loaded` says whether it is in memory now)."""
        if not self.current:
            return {"vision": False, "vision_available": False, "vision_size_gb": 0.0, "vision_loaded": False}
        from localcode.ui.server_cmd import catalog_choice, mmproj_for
        choice = catalog_choice(str(self.models_dir / f"{self.current}.gguf"))
        available = bool(choice is not None and getattr(choice, "supports_vision", False))
        return {"vision": mmproj_for(str(self.models_dir / f"{self.current}.gguf")) is not None,
                "vision_available": available,
                "vision_size_gb": round(float(getattr(choice, "mmproj_size_gb", 0.0) or 0.0), 2) if available else 0.0,
                "vision_loaded": bool(getattr(self, "vision_loaded", False) and getattr(self, "vision_wanted", None) == self.current)}

    def vision_install(self) -> dict:
        """User-requested: download the projector for the loaded model. It loads
        with the first image (vision_load); a running server is not restarted.
        Progress is in /status like any download."""
        if not self.current:
            return {"error": "no model loaded"}
        if not self.lock.acquire(blocking=False):
            return {"error": "a model switch is already in progress"}
        alias = self.current
        def work() -> None:
            try:
                self.active = {"group": None, "filename": None}
                self.ensure_mmproj(alias)
                process = getattr(self, "proc", None)
                if process is not None and process.poll() is None and self.healthy():
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                elif self.start(alias):
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                else:
                    self.state = {"state": "error", "model": alias, "detail": "llama-server failed to restart", "pct": None}
            finally:
                self.active = {}
                self.lock.release()
        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    def vision_load(self) -> dict:
        """An image was sent: restart the server with the projector that is
        already on disk. Never downloads (a missing projector stays the
        ask-first /vision flow). Returns at once; /status reports `loading`
        and then `vision_loaded`."""
        if not self.current:
            return {"error": "no model loaded"}
        info = self.vision_info()
        if info["vision_loaded"]:
            return {"ok": True, "loaded": True}
        if not info["vision"]:
            return {"error": "This model's image projector is not downloaded. Run /vision to download it.",
                    "vision_available": info["vision_available"], "vision_size_gb": info["vision_size_gb"]}
        if not self.lock.acquire(blocking=False):
            if self.state.get("state") == "loading" and self.vision_wanted == self.current:
                return {"ok": True, "loading": True}
            return {"error": "a model switch is already in progress"}
        alias = self.current
        self.vision_wanted = alias
        self.state = {"state": "loading", "model": alias, "detail": "loading the image projector…", "pct": None}
        def work() -> None:
            try:
                # The request that carries the image waits on this; a reply
                # still streaming (a tool step that just ended) finishes first.
                try:
                    self.cancel_warmup()
                except Exception:  # noqa: BLE001, S110
                    pass
                for _ in range(60):
                    if self.progress().get("phase") == "idle":
                        break
                    time.sleep(0.5)
                if self.start(alias):
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                    log(f"supervisor: image projector loaded for {alias}")
                else:
                    self.vision_wanted = None
                    self.state = {"state": "error", "model": alias, "detail": "llama-server failed to restart with the image projector", "pct": None}
            except Exception as exc:  # noqa: BLE001
                self.vision_wanted = None
                log(f"supervisor: image projector load failed: {type(exc).__name__}: {exc}")
                self.state = {"state": "error", "model": alias, "detail": "Could not load the image projector; see supervisor.log", "pct": None}
            finally:
                self.lock.release()
        threading.Thread(target=work, daemon=True).start()
        return {"ok": True, "loading": True, "size_gb": info["vision_size_gb"]}

    def ensure_mmproj(self, alias: str) -> None:
        """Fetch the model's vision projector (mmproj sidecar). Only called on an
        explicit user request (vision_install); failure only means text-only."""
        try:
            from localcode.ui.server_cmd import catalog_choice
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
                    log(f"supervisor: no vision projector for {alias}: {msg}")
            if dest.is_file() and dest.parent != self.models_dir and not (self.models_dir / dest.name).exists():
                os.symlink(dest, self.models_dir / dest.name)
        except Exception as e:  # noqa: BLE001
            log(f"supervisor: vision projector step failed: {e}")

    def decision(self, body) -> dict:
        from localcode.decision import DecisionError, post, supported, validate_request, validate_response
        data = validate_request(body)
        if self.state.get("state") != "ready" or not self.current or self.proc is None:
            raise DecisionError("Load a decision model before evaluating", 409)
        if not supported(self.current):
            raise DecisionError("This model does not support decisions. Select OpenJev with /models", 409)
        model, process = self.current, self.proc
        result = post(f"http://127.0.0.1:{self.port}/v1/systemone", data, server_headers())
        if self.current != model or self.proc is not process or self.state.get("state") != "ready":
            raise DecisionError("The model changed during the request; evaluate again", 409)
        validate_response(result, data)
        return {**result, "model": model}

    def start(self, alias: str, wait_s: int = 240) -> bool:
        if self._closing.is_set():
            return False
        gguf = self.models_dir / f"{alias}.gguf"
        # localcode's own per-machine server command (RAM-tier context, KV
        # compression, flash-attn, checkpoints) — nothing hardcoded here.
        from localcode.ui.server_cmd import server_command
        cmd = server_command(str(gguf), self.port, alias, vision=getattr(self, "vision_wanted", None) == alias)
        with self._process_lock:
            if self._closing.is_set():
                return False
            self.stop()
            cmd[0] = self.server_bin
            self.ctx = self.ctx_total = int(cmd[cmd.index("--ctx-size") + 1])
            self.slots = int(cmd[cmd.index("--parallel") + 1]) if "--parallel" in cmd else 1
            from localcode.ui.auth import redact_command
            self.log.write(f"\n=== {time.ctime()} {' '.join(redact_command(cmd))}\n".encode())
            with self._budget_lock:
                from localcode.ui.context_budget import TimingTable
                self._timing = TimingTable()
                self._log_offset = self.log.tell()
                self.pp_prior = None
                self._budget = None
                self._budget_info = {}
            from localcode.ui import warmup
            self._launch_key = warmup.launch_key(cmd)
            from localcode.ui.process_guard import spawn_owned
            self.vision_loaded = "--mmproj" in cmd
            self.proc = spawn_owned(cmd, stdout=self.log, lease_fd=getattr(self, "lease_fd", None))
            process = self.proc
            log(f"supervisor: started model process guard pid {self.proc.pid} for {alias}")
        for _ in range(wait_s):
            if self._closing.is_set() or process.poll() is not None:
                return False
            if self.healthy():
                self.current = alias
                self._probe_props()
                self._start_warmup(alias)
                return True
            time.sleep(1)
        return False

    # ---- prompt warm-up (see ui/warmup.py) ----------------------------------
    def _start_warmup(self, alias: str) -> None:
        self._context_ready = False
        from localcode.ui import warmup
        if not warmup.enabled():
            self._warm_status = "disabled"
            return
        self._warm_status = "checking"
        self._warm_stop = threading.Event()
        threading.Thread(target=self._warmup_worker, args=(alias, self._warm_stop), daemon=True).start()

    def _warmup_worker(self, alias: str, stop: threading.Event) -> None:
        from localcode.ui import warmup
        port = self.port
        baseline: int | None = None
        model_path = self.models_dir / f"{alias}.gguf"
        launch_key = getattr(self, "_launch_key", "")
        try:
            toks = warmup.load(alias, model_path)
            if stop.is_set():
                return
            self._warm_status = "warming" if toks else "cold"
            state = warmup.saved_state(alias, model_path, launch_key) if toks and launch_key else None
            if state:
                # A kept slot state of this exact launch: restoring it is a
                # file read, with the checkpoints the next turn resumes from.
                t0 = time.time()
                try:
                    n = warmup.restore(port, state)
                except Exception as e:  # noqa: BLE001
                    n = 0
                    log(f"warmup: restore failed for {alias}, replaying tokens: {e}")
                if n > 0:
                    toks = None
                    if not stop.is_set():
                        self._warm_status = "warm"
                        self._context_ready = True
                    log(f"warmup: restored {n} tokens for {alias} in {time.time() - t0:.2f}s")
            if toks:
                t0 = time.time()
                rp = warmup.Replay(port)
                self._warm_replay = rp
                try:
                    tm = rp.run(toks)
                    if not stop.is_set():
                        self._warm_status = "warm"
                        self._context_ready = True
                    log(f"warmup: replayed {len(toks)} tokens for {alias} in {time.time() - t0:.1f}s "
                        f"(server prompt_ms={tm.get('prompt_ms')}, cache_n={tm.get('cache_n')})")
                    try:
                        ms = float(tm.get("prompt_ms") or 0)
                        if ms > 0 and len(toks) >= 512:
                            self.pp_prior = len(toks) / (ms / 1000.0)
                    except (TypeError, ValueError):
                        pass
                except Exception as e:  # noqa: BLE001
                    if rp.cancelled or stop.is_set():
                        if not stop.is_set():
                            self._warm_status = "interrupted"
                        log(f"warmup: replay cancelled after {time.time() - t0:.1f}s (user turn started)")
                    else:
                        log(f"warmup: replay failed for {alias}, forgetting it: {e}")
                        self._warm_status = "failed"
                        warmup.forget(alias)
                finally:
                    if self._warm_replay is rp:
                        self._warm_replay = None
            if stop.is_set():
                return
            for s in warmup.slots(port) or []:
                try:
                    baseline = max(baseline or -1, int(s.get("id_task") if s.get("id_task") is not None else -1))
                except (TypeError, ValueError):
                    pass
        except Exception as e:  # noqa: BLE001
            if not stop.is_set():
                self._warm_status = "failed"
            log(f"warmup: skipped: {e}")
        # Capture this session's first turn so the next load of this model is warm.
        tracker = warmup.CaptureTracker(baseline_task=baseline)
        deadline = time.time() + warmup.CAPTURE_WINDOW_S
        while not stop.is_set() and time.time() < deadline:
            snap = warmup.slots(port)
            if snap is not None:
                slot_id = tracker.observe(snap, time.time())
                if slot_id is not None:
                    try:
                        new = warmup.capture(port, slot_id, keep_as=warmup.state_name(alias))
                        merged = warmup.merge_tokens(warmup.load(alias, model_path), new)
                        warmup.store(alias, merged, model_path, state_key=launch_key)
                        log(f"warmup: captured {len(new)} tokens for {alias}, stored {len(merged)}")
                    except Exception as e:  # noqa: BLE001
                        log(f"warmup: capture failed for {alias}: {e}")
                    return
            stop.wait(0.5)

    def cancel_warmup(self) -> dict:
        """The user submitted a turn: stop pre-reading so their request is next."""
        rp = self._warm_replay
        if rp is not None:
            rp.cancel()
            return {"ok": True, "cancelled": True}
        return {"ok": True, "cancelled": False}

    def shutdown(self) -> None:
        """Permanently forbid new model workers before stopping the current one."""
        self._closing.set()
        self.cancel.set()
        self.stop()

    def stop(self) -> None:
        with self._process_lock:
            self._warm_stop.set()
            self.cancel_warmup()
            if self.proc and self.proc.poll() is None:
                log(f"supervisor: stopping model service pid {self.proc.pid}")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            owner_pipe = getattr(self.proc, "stdin", None)
            if owner_pipe is not None:
                owner_pipe.close()
            self.proc = None

    def _probe_props(self) -> None:
        """The context the server really gives each request. With --parallel N the
        flag's --ctx-size is split N ways; /props reports the per-slot figure."""
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/props", headers=server_headers())
            with local_urlopen(req, timeout=3) as r:
                props = json.loads(r.read().decode())
            n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
            slots = int(props.get("total_slots") or 0)
        except Exception:  # noqa: BLE001
            n_ctx, slots = 0, 0
        if slots > 0:
            self.slots = slots
        if n_ctx > 0:
            self.ctx = n_ctx
        elif self.slots > 1:
            self.ctx = max(1, self.ctx_total // self.slots)
        log(f"supervisor: context per slot {self.ctx} (total {self.ctx_total}, slots {self.slots})")

    def _log_request_perf(self, samples: list, added: int) -> None:
        """One supervisor.log line per finished request, so a slow-down over a
        long session can be attributed: context depth, the idle gap before the
        request (macOS unwires idle GPU buffers), power source and swap. Debug
        diagnostics only; nothing here changes behaviour."""
        try:
            from localcode.ui.diagnostics import machine_observations
            machine = machine_observations()
        except Exception:  # noqa: BLE001
            machine = {}
        start = len(samples) - added
        for i in range(max(0, start), len(samples)):
            s, prev = samples[i], samples[i - 1] if i > 0 else None
            idle = None
            if s.t_end is not None and prev is not None and prev.t_end is not None:
                idle = max(0.0, s.t_end - s.busy_s - prev.t_end)
            pp = f"{s.pp_tps:.0f}" if s.pp_tps else "-"
            tg = f"{s.tg_tps:.1f}" if s.tg_tps else "-"
            swap = machine.get("swap_used_bytes")
            log(f"perf: ctx={s.n_tokens} prefill={s.pp_n}@{pp}/s decode={s.tg_n}@{tg}/s "
                f"idle_before={'-' if idle is None else f'{idle:.0f}s'} "
                f"power={'battery' if machine.get('on_battery') else 'ac' if machine.get('on_battery') is False else '?'} "
                f"swap={'-' if swap is None else f'{swap / 2**30:.1f}G'}")

    def context_info(self) -> dict:
        """ctx (per slot), the speed-derived budget, and the measurements behind it.
        Reads whatever llama-server has appended to server.log since the last call."""
        from localcode.ui import context_budget as cb
        with self._budget_lock:
            table = self._timing
            if table is not None:
                try:
                    with open(HERE / "server.log", "rb") as f:
                        f.seek(self._log_offset)
                        chunk = f.read()
                    self._log_offset += len(chunk)
                    if chunk:
                        added = table.feed(chunk.decode("utf-8", "replace"))
                        if added:
                            self._log_request_perf(table.samples, added)
                except OSError:
                    pass
            samples = table.samples if table is not None else []
            reserve = cb.step_reserve(self.ctx, min(8192, max(1, self.ctx // 4)), 50 * 1024)
            # LOCALCODE_CONTEXT_BUDGET=off publishes no budget (the runtime then
            # compacts at ctx - output as before); LOCALCODE_REREAD_MAX_S tunes
            # the one latency constant. Both are escape hatches, not settings.
            if os.environ.get("LOCALCODE_CONTEXT_BUDGET", "").strip().lower() in {"0", "off", "false", "no"}:
                return {"ctx": self.ctx, "ctx_total": self.ctx_total, "slots": self.slots}
            try:
                reread_s = float(os.environ.get("LOCALCODE_REREAD_MAX_S") or cb.REREAD_MAX_S)
            except ValueError:
                reread_s = cb.REREAD_MAX_S
            new, info = cb.budget(self.ctx, reserve, samples, self.pp_prior, reread_max_s=reread_s)
            step = max(1, self.ctx // cb.GRID)
            if self._budget is None or abs(new - self._budget) >= step:
                if self._budget is not None:
                    log(f"supervisor: context budget {self._budget} -> {new} ({info.get('basis')}, {info.get('limit', 'no limit hit')})")
                self._budget = new
            self._budget_info = info
            return {"ctx": self.ctx, "ctx_total": self.ctx_total, "slots": self.slots,
                    "budget": self._budget, "budget_info": info}

    def healthy(self) -> bool:
        try:
            with local_urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def crash_watchdog(self, interval: float = 1.0, cooldown: float = 120.0) -> None:
        """Restart the loaded model once after an unexpected exit.

        A crash mid-reply otherwise leaves every open chat retrying a dead server. One
        automatic restart lets the runtime's retry succeed; a second crash within the
        cooldown stays an error so a model that keeps crashing does not loop.
        """
        last_restart = float("-inf")
        while not self._closing.is_set():
            time.sleep(interval)
            failure = self._server_failure()
            alias = getattr(self, "current", None)
            if not failure or not alias or time.monotonic() - last_restart < cooldown:
                continue
            if not self.lock.acquire(blocking=False):
                continue
            try:
                if not self._server_failure():
                    continue
                last_restart = time.monotonic()
                log(f"supervisor: {failure['detail'].split('. ')[0]}; restarting {alias} once")
                self.state = {"state": "loading", "model": alias, "detail": "restarting after a crash", "pct": None}
                if self.start(alias):
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                    log(f"supervisor: {alias} restarted after a crash")
                else:
                    self.state = {"state": "error", "model": alias, "pct": None,
                                  "detail": "Model server crashed and could not restart. Open /server to retry or choose another model."}
            finally:
                self.lock.release()

    def server_status(self) -> dict:
        """Report current liveness, not just the last successful model load."""
        state = dict(self.state)
        process = self.proc
        running = process is not None and process.poll() is None
        ready = running and state["state"] == "ready" and self.healthy()
        failure = self._server_failure()
        if failure:
            state.update(state="error", **failure)
        elif state["state"] == "ready" and not ready:
            state.update(state="error", detail="Model server is unavailable")
        return dict(state, server_running=running, server_ready=ready,
                    version=localcode.__version__, process_id=getattr(process, "pid", None),
                    warmup=getattr(self, "_warm_status", "idle"))

    def server_action(self, action: str, *, force: bool = False) -> dict:
        """Queue an explicit lifecycle action; retain session and selected model."""
        if action not in ("start", "stop"):
            return {"error": "Unknown server action"}
        if not self.lock.acquire(blocking=False):
            return {"error": "A model operation is in progress; wait and refresh /server"}
        queued = False
        previous = dict(self.state)
        try:
            if self.state["state"] in ("loading", "downloading", "stopping"):
                return {"error": "A model operation is in progress; wait and refresh /server"}
            running = self.proc is not None and self.proc.poll() is None
            if action == "stop":
                if running and not force:
                    phase = self.progress().get("phase")
                    if phase != "idle":
                        return {"error": "A request is active." if phase in ("reading", "generating", "prefill") else "Server activity could not be checked.",
                                "force_required": True, "phase": phase}
                if not running:
                    self.state = {"state": "stopped", "model": self.current, "detail": "Model server stopped. Use /server to start it.", "pct": None}
                    return {"ok": True, "state": "stopped"}
                self.state = {"state": "stopping", "model": self.current, "detail": "Stopping model server…", "pct": None}
            else:
                if running:
                    return {"error": "Model server is already running. Stop it first to restart."}
                if not self.current:
                    return {"error": "No previous model. Select a model using /model first."}
                if not _safe_filename(self.current + ".gguf") or not (self.models_dir / (self.current + ".gguf")).is_file():
                    return {"error": "The previous model file is missing. Select an installed model using /model."}
                self.state = {"state": "loading", "model": self.current, "detail": "Reloading model…", "pct": None}
            threading.Thread(target=self._server_action_worker, args=(action, force), daemon=True).start()
            queued = True
            return {"ok": True, "state": self.state["state"]}
        except Exception:
            self.state = previous
            log("Could not start server lifecycle worker")
            return {"error": "Could not begin the operation. Refresh /server and retry."}
        finally:
            if not queued:
                self.lock.release()

    def _server_action_worker(self, action: str, force: bool = False) -> None:
        try:
            if action == "stop":
                # Recheck immediately before stopping, including other API clients.
                if not force and self.proc is not None and self.proc.poll() is None and self.progress().get("phase") != "idle":
                    self.state = {"state": "ready", "model": self.current, "detail": "Stop cancelled: a request started. Cancel it and retry.", "pct": None}
                    return
                self.stop()
                self.state = {"state": "stopped", "model": self.current, "detail": "Model server stopped. Use /server to start it.", "pct": None}
            elif self.start(self.current):
                self.state = {"state": "ready", "model": self.current, "detail": "", "pct": None}
            else:
                self.stop()
                self.state = {"state": "error", "model": self.current, "detail": "Model failed to load. Retry Start or select another model with /model.", "pct": None}
        except Exception as exc:
            log(f"Server {action} failed: {exc}")
            try:
                self.stop()
            except Exception:
                log("Could not clean up model process after lifecycle failure")
            self.state = {"state": "error", "model": self.current, "detail": "Model operation failed. Refresh /server and retry; see supervisor.log.", "pct": None}
        finally:
            self.lock.release()

    # ---- catalog ------------------------------------------------------------
    def _rec_repo(self) -> str | None:
        try:
            return recommend(self.ram_gb, self.bandwidth).hf_repo
        except Exception:  # noqa: BLE001
            return None

    def _current_repo(self) -> str | None:
        if not self.current:
            return None
        from localcode.models_catalog import group_for_filename
        g = group_for_filename(self.current + ".gguf")
        return g.hf_repo if g else None

    def _installed_models(self) -> dict[str, Path]:
        # The picker polls frequently. Avoid recursively rescanning external
        # caches on every poll while noticing removals immediately.
        cached = getattr(self, "_model_file_cache", None)
        now = time.monotonic()
        if cached and cached[0] == self.models_dir and now - cached[1] < 3:
            return {name: path for name, path in cached[2].items() if path.is_file()}
        files = model_files(self.models_dir)
        self._model_file_cache = (self.models_dir, now, files)
        return files

    def catalog(self) -> dict:
        from localcode.models_catalog import group_for_filename
        installed = {}
        for file in self._installed_models().values():
            if not file.is_file() or file.name.lower().startswith("mmproj"):
                continue
            group = group_for_filename(file.name)
            if group:
                installed[group.key] = installed.get(group.key, 0) + 1
        rec, cur = self._rec_repo(), self._current_repo()
        process = getattr(self, "proc", None)
        loaded = process is not None and process.poll() is None and self.state["state"] == "ready"
        groups = []
        for g in MODEL_GROUPS:
            groups.append({
                "key": g.key, "display_name": g.display_name, "maker": g.maker,
                "license": g.license, "hf_repo": g.hf_repo,
                "recommended": g.hf_repo == rec, "current": g.hf_repo == cur,
                "loaded": loaded and g.hf_repo == cur,
                "installed_count": installed.get(g.key, 0),
                "downloading": self.state["state"] == "downloading" and self.active.get("group") == g.key,
                "pct": self.state["pct"] if self.active.get("group") == g.key else None,
            })
        return {"ram_gb": self.ram_gb, "current": self.current, "groups": groups,
                "models_dir": str(self.models_dir)}

    def _quant_rows(self, g, refresh: bool = False):
        from localcode.hf_quants import Quant, _parse_label, _read_cache
        from localcode.models_catalog import known_quants
        cached = _read_cache(g.hf_repo)
        memory = self.__dict__.setdefault("_quant_memory", {})
        rows, fetched = memory.get(g.hf_repo, cached or ([], 0))
        if cached and cached[1] > fetched:
            rows, fetched = cached
        if refresh and time.time() - fetched >= 86400:
            lock = self.__dict__.setdefault("_quant_lock", threading.Lock())
            with lock:
                pending = self.__dict__.setdefault("_quant_pending", set())
                attempts = self.__dict__.setdefault("_quant_attempts", {})
                if g.hf_repo not in pending and time.monotonic() - attempts.get(g.hf_repo, -60) >= 60:
                    pending.add(g.hf_repo)
                    attempts[g.hf_repo] = time.monotonic()
                    def fetch():
                        try:
                            memory[g.hf_repo] = (fetch_quants(g.hf_repo), time.time())
                        except Exception as exc:
                            log(f"Quant refresh failed: {type(exc).__name__}")
                        finally:
                            with lock:
                                pending.discard(g.hf_repo)
                    try:
                        threading.Thread(target=fetch, daemon=True).start()
                    except RuntimeError:
                        pending.discard(g.hf_repo)
        # Trusted curated choices and the recorded repo sizes keep a
        # first-time/offline picker usable too.
        known = {q.filename: q for q in rows if not q.is_mmproj}
        for filename, size_gb in known_quants(g):
            label = _parse_label(filename)
            if label:
                known.setdefault(filename, Quant(filename, size_gb, label, False))
        return list(known.values())

    def quants(self, key: str) -> dict:
        g = next((g for g in MODEL_GROUPS if g.key == key), None)
        if g is None:
            return {"error": f"unknown group {key}"}
        rows = self._quant_rows(g, refresh=True)
        # Local weights remain selectable even when HF is offline, or its
        # current listing no longer contains a previously downloaded quant.
        from localcode.hf_quants import Quant, _parse_label
        from localcode.models_catalog import group_for_filename
        installed = self._installed_models()
        listed = {q.filename for q in rows}
        for filename, path in installed.items():
            label = _parse_label(filename)
            if label and filename not in listed and group_for_filename(filename) == g:
                try:
                    rows.append(Quant(filename, path.stat().st_size / 1e9, label, False))
                except OSError:
                    continue
        rows.sort(key=lambda q: q.size_gb)
        rec_idx = self._recommended_quant_idx(rows, g)
        process = getattr(self, "proc", None)
        loaded = process is not None and process.poll() is None and self.state["state"] == "ready"
        out = []
        # A vendor-trained drafter ships with the model and loads with it when
        # it fits beside the quant (quant_plan, the launcher's own rule); the
        # size the user sees includes it only then.
        draft_gb = float(getattr(getattr(g, "drafter", None), "size_gb", 0.0) or 0.0)
        for i, q in enumerate(rows):
            spd = estimate_decode_tok_s(q.size_gb, g.display_name, self.bandwidth)
            plan = quant_plan(g, q.size_gb, self.ram_gb)
            degraded = is_degraded(q.label, g)
            out.append({
                "filename": q.filename, "alias": _alias(q.filename),
                # Below UD-Q2_K_XL agent quality falls off; say so where the user picks.
                "label": f"{q.label} · degraded" if degraded else q.label,
                "degraded": degraded, "drafter": plan["drafter"],
                "size_gb": round(plan["size_gb"], 1), "fit": plan["fit"],
                "tok_s": spd, "recommended": i == rec_idx,
                "downloaded": q.filename in installed,
                "current": _alias(q.filename) == self.current,
                "loaded": loaded and _alias(q.filename) == self.current,
                "downloading": self.state["state"] == "downloading" and self.active.get("filename") == q.filename,
                "pct": self.state["pct"] if self.active.get("filename") == q.filename else None,
            })
        return {"group": g.key, "display_name": g.display_name, "maker": g.maker,
                "license": g.license, "ram_gb": self.ram_gb, "quants": out,
                "listing": g.hf_repo in getattr(self, "_quant_pending", set()),
                "drafter_size_gb": round(draft_gb, 2),
                "vision_size_gb": round(float(getattr(g, "mmproj_size_gb", 0.0) or 0.0), 2) if getattr(g, "mmproj_filename", None) else 0.0}

    def _recommended_quant_idx(self, rows, g) -> int | None:
        """The starred quant: fits this Mac, not degraded (models_catalog)."""
        return recommended_quant_index(
            rows, g, self.ram_gb,
            lambda size: estimate_decode_tok_s(size, g.display_name, self.bandwidth),
            MIN_SPEED_FRACTION)

    # ---- select (download + switch), runs on its own thread ------------------
    def select(self, key: str, filename: str) -> dict:
        g = next((g for g in MODEL_GROUPS if g.key == key), None)
        if g is None:
            return {"error": f"unknown group {key}"}
        if not _safe_filename(filename):
            return {"error": "invalid filename"}
        if not self.lock.acquire(blocking=False):
            return {"error": "a model switch is already in progress"}
        alias = _alias(filename)
        queued = False
        previous = None
        try:
            # Check liveness before disk/network access: the loaded file may
            # have moved, and reselecting it must never trigger a download.
            process = getattr(self, "proc", None)
            if alias == getattr(self, "current", None) and process is not None and process.poll() is None and self.healthy():
                if self.state["state"] != "ready":
                    self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
                return {"ok": True, "model": alias, "state": "ready", "already_loaded": True,
                        "detail": "Model already loaded"}
            if process is not None and process.poll() is None and self.progress().get("phase") != "idle":
                return {"error": "A request is active or server activity cannot be verified. Cancel it or wait before switching models."}
            existing = find_model(filename, self.models_dir)
            target = self.models_dir / filename
            if existing is not None and not target.is_file():
                self.models_dir.mkdir(parents=True, exist_ok=True)
                target.symlink_to(existing.resolve())
            if not target.is_file():
                listed = {q.filename for q in self._quant_rows(g)}
                if filename not in listed:
                    return {"error": f"{filename} is not a file of {g.hf_repo}"}
            previous = (dict(self.state), dict(self.active), self.cancel.is_set())
            self.cancel.clear()
            self.active = {"group": key, "filename": filename}
            self.state = {"state": "loading" if target.is_file() else "downloading",
                          "model": alias, "detail": "", "pct": None}
            threading.Thread(target=self._select_worker, args=(g, filename, alias), daemon=True).start()
            queued = True
            return {"ok": True, "model": alias, "state": self.state["state"]}
        except Exception as exc:
            log(f"Could not begin model switch: {type(exc).__name__}")
            return {"error": "Could not begin the model switch. Check the model folder and retry."}
        finally:
            if not queued:
                if previous is not None:
                    self.state, self.active, was_cancelled = previous
                    if was_cancelled:
                        self.cancel.set()
                self.lock.release()

    def _select_worker(self, g, filename: str, alias: str) -> None:
        previous_process = getattr(self, "proc", None)
        load_started = False
        try:
            download_requested = not (self.models_dir / filename).exists()
            if download_requested:
                size = next((q.size_gb for q in self._quant_rows(g) if q.filename == filename), 0.0)
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
                    self.state = {"state": "idle", "model": self.current, "detail": "download cancelled", "pct": None}
                    return
                if not ok:
                    self.state = {"state": "error", "model": alias, "detail": res, "pct": None}
                    return
                got = Path(res)
                if got.parent != self.models_dir and not (self.models_dir / filename).exists():
                    # download_model saved under localcode's model_dir(); link it here.
                    os.symlink(got, self.models_dir / filename)
            # The drafter that ships with this model (its size was part of the
            # figure the user picked). Missing it is not fatal: the model runs
            # without speculation and the next select retries.
            drafter = getattr(g, "drafter", None)
            if drafter is not None and not drafter.local_path.is_file() and (
                download_requested or os.environ.get("LOCALCODE_DOWNLOAD_DRAFTER") == "1"
            ):
                def on_draft_progress(msg: str) -> None:
                    pct = None
                    if "(" in msg and "%)" in msg:
                        try:
                            pct = int(msg.rsplit("(", 1)[1].split("%")[0])
                        except ValueError:
                            pct = None
                    self.state = {"state": "downloading", "model": alias, "detail": f"drafter: {msg}", "pct": pct}
                ok, res = bootstrap.download_drafter(choice_for_quant(g, filename, 0.0), on_progress=on_draft_progress, cancel_event=self.cancel)
                if self.cancel.is_set():
                    self.state = {"state": "idle", "model": self.current, "detail": "download cancelled", "pct": None}
                    return
                if not ok:
                    log(f"supervisor: drafter download failed for {alias}: {res}")
                else:
                    self._drop_retired_sidecars()
            # Downloads may take minutes; another client can start a request
            # before the switch gets here. Never terminate that active reply.
            process = getattr(self, "proc", None)
            if process is not None and process.poll() is None and self.progress().get("phase") != "idle":
                self.state = {"state": "error", "model": alias,
                              "detail": "Model files are ready, but another request is active. Retry the selection when it finishes.", "pct": None}
                return
            self.state = {"state": "loading", "model": alias, "detail": "loading model…", "pct": None}
            if getattr(self, "vision_wanted", None) != alias:
                self.vision_wanted = None  # a newly loaded model starts text-only
            load_started = True
            if self.start(alias):
                self.state = {"state": "ready", "model": alias, "detail": "", "pct": None}
            else:
                if getattr(self, "proc", None) is not previous_process:
                    self.stop()
                self.state = {"state": "error", "model": alias,
                              "detail": "llama-server failed to load the model (see server.log in the localcode-agent run dir)",
                              "pct": None}
        except Exception as e:  # noqa: BLE001
            if load_started and getattr(self, "proc", None) is not previous_process:
                self.stop()
            import traceback
            log(traceback.format_exc())
            self.state = {"state": "error", "model": alias, "detail": str(e), "pct": None}
        finally:
            self.active = {}
            log(f"supervisor: switch to {alias} -> {self.state['state']}")
            self.lock.release()

    def vision(self) -> bool:
        if not self.current:
            return False
        from localcode.ui.server_cmd import mmproj_for
        return mmproj_for(str(self.models_dir / f"{self.current}.gguf")) is not None

    # ---- voice: push-to-talk STT + read-aloud, all local ---------------------
    voice_proc: subprocess.Popen | None = None
    voice_recorder = ""  # "portaudio" (bundled) or "ffmpeg" (fallback) while recording
    voice_wav = HERE / "voice.wav"
    voice_detail = ""
    voice_setup_lock = threading.Lock()
    voice_live = False          # the running recorder is the live-dictation worker
    voice_partial_text = ""     # what it has heard so far
    voice_final: dict | None = None
    voice_final_ready = threading.Event()

    @staticmethod
    def _voice_venv() -> Path:
        return HERE / "voice-venv"

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
                self.voice_detail = "installing whisper.cpp + microphone recorder into the voice runtime…"
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
        """ffmpeg avfoundation audio device index: the built-in mic, else 0."""
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                                 capture_output=True, text=True, timeout=15, check=False).stderr
            audio = out.split("audio devices:", 1)[1] if "audio devices:" in out else ""
            for line in audio.splitlines():
                if "[" in line and "]" in line and ("MacBook" in line or "Built-in" in line):
                    return line.rsplit("[", 1)[1].split("]")[0]
        except Exception:  # noqa: BLE001, S110
            pass
        return "0"

    def voice_start(self) -> dict:
        if self.voice_proc is not None and self.voice_proc.poll() is None:
            return {"error": "already recording"}
        py = self._voice_python()
        if py is None or not self._voice_model_ready():
            return {"error": "setup_needed"}
        self.voice_wav.unlink(missing_ok=True)
        try:
            subprocess.run([str(py), "-c", "import sounddevice"], check=True, capture_output=True, timeout=30)
            return self._voice_start_live(py)
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

    def _voice_start_live(self, py: Path) -> dict:
        """Live dictation (localcode/voice_live.py): one worker records and
        transcribes as you speak, and yields the GPU to the coding model."""
        from localcode.voice import DEFAULT_STT_MODEL_DIR, DEFAULT_STT_MODEL_NAME, VoiceState
        model = VoiceState().stt_model_path or (DEFAULT_STT_MODEL_DIR / DEFAULT_STT_MODEL_NAME)
        worker = Path(localcode.__file__).resolve().parent / "voice_live.py"
        env = {**os.environ, "LOCALCODE_CONTROL_URL": f"http://127.0.0.1:{self.control_port}", "LOCALCODE_CONTROL_TOKEN": self.control_token}
        self.voice_recorder, self.voice_live, self.voice_partial_text, self.voice_final = "portaudio", True, "", None
        self.voice_final_ready = threading.Event()
        opened, done = threading.Event(), self.voice_final_ready
        proc = subprocess.Popen([str(py), str(worker), str(model)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True)
        self.voice_proc = proc

        def read() -> None:
            final: dict = {"error": "the voice worker stopped without a transcript"}
            for line in proc.stdout:  # type: ignore[union-attr]
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "recording":
                    opened.set()
                elif kind == "partial" and self.voice_proc is proc:
                    self.voice_partial_text = str(msg.get("text", ""))
                elif kind == "final":
                    final = {"ok": True, "text": str(msg.get("text", ""))}
                elif kind == "error":
                    final = {"error": str(msg.get("message", "voice worker failed"))}
            if self.voice_proc is proc or self.voice_final is None:
                self.voice_final = final
            opened.set(); done.set()

        threading.Thread(target=read, daemon=True).start()
        opened.wait(timeout=8)
        if proc.poll() is not None:
            self.voice_proc, self.voice_live = None, False
            err = (self.voice_final or {}).get("error") or "unknown error"
            return {"error": f"could not open the microphone: {err} (grant microphone access to your terminal in System Settings → Privacy)"}
        return {"ok": True, "live": True}

    def voice_partial(self) -> dict:
        recording = self.voice_proc is not None and self.voice_proc.poll() is None
        return {"recording": recording, "live": self.voice_live, "text": self.voice_partial_text if self.voice_live else ""}

    def _voice_stop_live(self, p: subprocess.Popen) -> dict:
        done = self.voice_final_ready
        try:
            p.stdin.close()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            p.terminate()
        if not done.wait(timeout=120):
            p.kill()
        self.voice_proc, self.voice_live = None, False
        out = self.voice_final or {"error": "transcription timed out"}
        if "error" in out:
            return {"error": f"transcription failed: {out['error']}"}
        return {"ok": True, "text": out.get("text", "").strip()}

    def voice_stop(self) -> dict:
        p = self.voice_proc
        if p is not None and self.voice_live:
            return self._voice_stop_live(p)      # also when the worker already ended: its transcript is kept
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
            f"sys.path.insert(0, {str(Path(localcode.__file__).resolve().parent.parent)!r})\n"
            "from pathlib import Path\n"
            "from localcode.voice import VoiceState, transcribe\n"
            f"ok, text = transcribe(VoiceState(), Path({str(wav)!r}))\n"
            "print(json.dumps({'ok': ok, 'text': text}))\n"
        )
        try:
            r = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=300, check=False)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
            out = json.loads(line)
        except Exception as e:  # noqa: BLE001
            return {"error": f"transcription failed: {e}"}
        if not out.get("ok"):
            return {"error": out.get("text") or "transcription failed"}
        # whisper labels non-speech ("[wind blowing]", "(music)", "[BLANK_AUDIO]"); drop them
        import re
        text = re.sub(r"\s*[\[(][^\])]{0,40}[\])]\s*", " ", out.get("text", "")).strip()
        return {"ok": True, "text": text}

    def voice_speak(self, text: str) -> dict:
        with self.speech_lock:
            if self.speech_proc and self.speech_proc.poll() is None:
                self.speech_proc.terminate()
                try:
                    self.speech_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.speech_proc.kill()
                    self.speech_proc.wait()
                self.speech_proc = None
                return {"ok": True, "speaking": False}
            return self._voice_speak(text)

    def _voice_speak(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "nothing to say"}
        if not shutil.which("say"):
            return {"error": "macOS `say` not available"}
        try:
            muted = subprocess.run(["osascript", "-e", "output muted of (get volume settings)"],
                                   capture_output=True, text=True, timeout=2, check=False)
            if muted.returncode == 0 and muted.stdout.strip() == "true":
                return {"error": "Sound output is muted. Unmute your Mac, then run /speak again."}
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.speech_proc = subprocess.Popen(["say", text[:4000]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True, "speaking": True}

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
            log(f"supervisor: could not persist model_dir: {e}")
        return dict(self.models_dir_info(), ok=True)


def _safe_filename(name: str) -> bool:
    return bool(name) and name.endswith(".gguf") and "/" not in name and "\\" not in name \
        and ".." not in name and not name.startswith(".") and name == name.strip() and "\x00" not in name


def make_handler(sup: Supervisor):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")  # no keep-alive reuse against a closed socket
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self, why: str) -> None:
            log(f"control: refused {self.command} {self.path}: {why}")
            self._json({"error": why}, 403)

        def do_GET(self):
            why = request_allowed(self.headers, sup.control_token, sup.control_port)
            if why:
                return self._refuse(why)
            u = urlparse(self.path)
            if u.path == "/catalog":
                return self._json(sup.catalog())
            if u.path == "/quants":
                key = parse_qs(u.query).get("group", [""])[0]
                return self._json(sup.quants(key))
            if u.path == "/status":
                from localcode.decision import supported
                return self._json(dict(sup.server_status(), current=sup.current,
                                       port=sup.gateway.port if getattr(sup, "gateway", None) else sup.port,
                                       inference_port=sup.port,
                                       group=sup.active.get("group"), filename=sup.active.get("filename"),
                                       supports_systemone=supported(sup.current),
                                       **sup.context_info(), **sup.vision_info()))
            if u.path == "/server":
                from localcode.ui.ports import preferred_port
                from localcode.ui.diagnostics import machine_observations, timing_summary
                try:
                    preferred = preferred_port()
                except RuntimeError:
                    preferred = "invalid; choose a new port"
                port = sup.gateway.port if getattr(sup, "gateway", None) else sup.port
                return self._json(dict(sup.server_status(), port=port,
                                       base_url=f"http://127.0.0.1:{port}/v1",
                                       model=f"localcode/{sup.current}" if sup.current else None,
                                       preferred_port=preferred,
                                       **sup.context_info(),
                                       performance=timing_summary(sup._timing.samples if sup._timing else []),
                                       machine=machine_observations()))
            if u.path == "/models_dir":
                return self._json(sup.models_dir_info())
            if u.path == "/voice/status":
                return self._json(sup.voice_status())
            if u.path == "/voice/partial":
                return self._json(sup.voice_partial())
            if u.path == "/progress":
                return self._json(sup.progress())
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            why = request_allowed(self.headers, sup.control_token, sup.control_port, want_json=True)
            if why:
                return self._refuse(why)
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json({"error": "bad content length"}, 400)
            if n < 0 or n > 8 * 1024 * 1024:
                return self._json({"error": "request exceeds 8 MB"}, 413)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            if u.path in ("/server/start", "/server/stop"):
                if not isinstance(body, dict):
                    return self._json({"error": "Expected a JSON object"}, 400)
                force = body.get("force", False)
                if type(force) is not bool:
                    return self._json({"error": "force must be a boolean"}, 400)
                res = sup.server_action(u.path.rsplit("/", 1)[1], force=force)
                return self._json(res, 409 if "error" in res else 200)
            if u.path == "/server/port":
                from localcode.ui.gateway import PortChangeError
                if not isinstance(body, dict) or "port" not in body:
                    return self._json({"error": "port is required"}, 400)
                if not getattr(sup, "gateway", None):
                    return self._json({"error": "Restart LocalCode once to enable live port changes"}, 409)
                try:
                    return self._json(sup.gateway.change(body["port"]))
                except PortChangeError as exc:
                    return self._json({"error": str(exc)}, exc.status)
            if u.path == "/decision":
                from localcode.decision import DecisionError
                try:
                    return self._json(sup.decision(body))
                except DecisionError as exc:
                    return self._json({"error": str(exc)}, exc.status)
            if u.path == "/select":
                res = sup.select(str(body.get("group", "")), str(body.get("filename", "")))
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/warmup/cancel":
                return self._json(sup.cancel_warmup())
            if u.path == "/cancel":
                res = sup.cancel_download()
                return self._json(res, 200 if "error" not in res else 409)
            if u.path == "/models_dir":
                res = sup.set_models_dir(str(body.get("path", "")))
                return self._json(res, 200 if "error" not in res else 400)
            if u.path == "/vision/load":
                return self._json(sup.vision_load())
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
    ap.add_argument("--models-dir", default=os.environ.get("LOCALCODE_MODEL_DIR") or os.environ.get(
        "LOCALCODE_MODELS_DIR", str(Path.home() / ".local/share/localcode/models")))
    ap.add_argument("--parent-pid", type=int, default=0, help="launcher PID; exit if it disappears")
    a = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    # A per-user lock, shared across checkouts, closes the concurrent-launch race.
    # The model inherits it so a killed supervisor cannot allow a second model.
    import fcntl
    lock_path = Path.home() / ".local/share/localcode-agent/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lock_path.open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("localcode is already running; refusing a second model supervisor")
        return 1
    from localcode.ui.gateway import ModelGateway
    import socket
    # The model and every LocalCode window use this stable private endpoint.
    # Only the externally advertised API listener moves during a port change.
    try:
        for _ in range(10):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                inference_port = sock.getsockname()[1]
            if inference_port not in (a.port, a.control_port):
                break
        else:
            log("could not allocate an internal model port")
            return 1
    except OSError as exc:
        log(f"could not allocate an internal model port: {exc}")
        return 1
    sup = Supervisor(a.server, inference_port, Path(a.models_dir), 0)
    sup.lease_fd = lease.fileno()
    # Session secrets: from the launcher's environment, else fresh. Written
    # owner-only into the run dir for a second `localcode` and dev tooling.
    sup.control_port = a.control_port
    sup.control_token = control_token() or new_secret()
    if not server_key():
        os.environ[KEY_ENV] = new_secret()
    write_auth_file(HERE, sup.control_token, server_key())
    httpd = None
    try:
        sup.gateway = ModelGateway(sup.port, a.port, a.control_port)
        httpd = ThreadingHTTPServer(("127.0.0.1", a.control_port), make_handler(sup))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    except (OSError, RuntimeError) as exc:
        if httpd is not None:
            httpd.server_close()
        if getattr(sup, "gateway", None):
            sup.gateway.close()
        log(f"could not start local API services: {exc}")
        return 1

    exiting = False

    def bye(signum, frame):
        nonlocal exiting
        if exiting:
            return
        exiting = True
        import traceback
        log(f"supervisor: got signal {signum}, exiting")
        log("".join(traceback.format_stack(frame)))
        # Stop model ownership before waiting for any network connection drain.
        sup.shutdown()
        sup.gateway.close()
        with sup.speech_lock:
            if sup.speech_proc and sup.speech_proc.poll() is None:
                sup.speech_proc.terminate()
        httpd.shutdown()
        httpd.server_close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGHUP, bye)
    import atexit
    atexit.register(sup.gateway.close)
    atexit.register(sup.shutdown)
    if a.parent_pid:
        def watch_launcher() -> None:
            while os.getppid() == a.parent_pid:
                time.sleep(0.5)
            log("supervisor: launcher exited; shutting down its model")
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=watch_launcher, daemon=True).start()

    if a.model:
        sup.state = {"state": "loading", "model": a.model, "detail": "", "pct": None}
        if not sup.start(a.model):
            log(f"llama-server failed to load {a.model}")
            return 1
        sup.state = {"state": "ready", "model": a.model, "detail": "", "pct": None}
    else:
        # First-run journey: the TUI opens first and its /models picker loads a model.
        sup.state = {"state": "idle", "model": None, "detail": "no model loaded", "pct": None}
    log("ready")
    threading.Thread(target=sup.crash_watchdog, daemon=True).start()
    # NOT signal.pause(): it returns on ANY signal, including the SIGCHLD from
    # a llama-server we just stopped, which made the supervisor exit mid-switch.
    threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
