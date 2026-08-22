"""Single source of truth for the local llama-server subprocess lifecycle.

Why this exists
---------------
The codebase previously scattered `pkill -f llama-server` and `lsof -ti :8081`
across the TUI setup flow, the runtime gateway, and the /model switch
handler. Each site had its own idea of "kill the old server" and "start a
new one," leading to:
- stale servers surviving TUI exits (no atexit cleanup)
- zombie children after ungraceful crashes (no process-group kill)
- next launch picking up the zombie via healthcheck, then failing in a
  hard-to-diagnose way (wrong model, wrong binary, mid-shutdown)
- `/model` switches killing the server without actually restarting it

This module centralises that state. Every place that starts or stops
`llama-server` now goes through `ServerManager.get()`.

Lifecycle contract
------------------
- Start the server with `mgr.start(cmd, model_path)`. Blocks until the HTTP
  health endpoint returns OK or the timeout fires. Returns True on success.
- Switch models with `mgr.restart(cmd, model_path)` — shorthand for
  shutdown + start.
- Check health without side effects via `mgr.is_healthy()`.
- Exit cleanly: an atexit handler calls `shutdown()` on process exit. The
  handler runs even on sys.exit or normal TUI quit; for SIGTERM/SIGINT the
  Python default handler also fires atexit, so we're covered.
- Ungraceful kill (SIGKILL, jetsam, power loss) leaves a stale PID file;
  the next `ServerManager.get()` call reconciles it — sends SIGKILL to the
  recorded PID if still alive, otherwise just deletes the file.

Why a process group
-------------------
We launch `llama-server` with `start_new_session=True` so the child becomes
the leader of a new process group. Shutdown then sends SIGTERM to the whole
group (via `os.killpg`), not just the immediate child. This catches any
worker threads or forked helpers the server might spawn in future versions
(currently llama-server doesn't fork, but the invariant is cheap to keep).

Why a PID file
--------------
The Popen object lives in this Python process. If the Python process dies
and a new one starts, we need to find the old server to clean it up. PID
file at ~/.localcode/llama-server.pid records the group leader PID; the new
process reads it, killpg's it, and deletes the file.

Why port-based last-resort kill
-------------------------------
If the PID file is missing, we fall back to killing whatever is bound to
:8081 (covers an already-running server from a different install).
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


from ._subproc_env import clean_env
from .paths import (
    server_pid_file,
    stuck_server_marker_path,
    pressure_kill_marker_path,
    lifecycle_log_path,
)

# PID file stays GLOBAL — only one llama-server runs per machine (single port,
# single GPU budget), so it keeps serving you when you `cd` between projects.
# See paths.py for the global-vs-project split rationale.
PID_FILE = server_pid_file()
DEFAULT_PORT = 8081
# Preferred-range scan (8081-8099) before falling through to an OS-assigned
# ephemeral port. If all 19 are occupied, `find_free_port()` falls back to
# `bind(('', 0))` so localcode can never be shut out by port contention.
PORT_FALLBACK_RANGE = 19
HEALTH_TIMEOUT_S = 120  # a 10 GB model can take ~60s cold on slower disks

# Env var to force an exact port — takes precedence over all scanning.
# For users with firewall rules, SSH tunnels, or reverse proxies that
# need localcode reachable on a specific port.
PORT_ENV_VAR = "LOCALCODE_PORT"


def _port_in_use(port: int) -> bool:
    """True if a TCP socket is already listening on `port`.

    Uses bind() without SO_REUSEADDR — the pessimistic check: "if *I*
    can't bind right now, treat it as busy." Safer than connect(),
    which could hang on a D-state process that holds the fd but isn't
    accept()ing. Consequence: a port in TIME_WAIT from our own prior
    server will read as busy for ~60s — fine, we just walk to the next.
    """
    import socket as _s
    try:
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return False
    except OSError:
        return True


def _os_ephemeral_port() -> int:
    """Ask the kernel for a guaranteed-free port.

    `bind(('', 0))` tells the OS to pick any free port from the ephemeral
    range (49152-65535 on macOS). There is a tiny race between the close
    here and whoever binds it next, but in practice it's the most
    reliable way to get a port nobody else is using. We rely on this as
    the last resort when every port in our preferred range is held.
    """
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_free_port(preferred: int = DEFAULT_PORT,
                   scan_range: int = PORT_FALLBACK_RANGE) -> int:
    """Return a port we can bind to *right now*.

    Selection strategy, in order:
      1. If $LOCALCODE_PORT is set, return that exact port without
         scanning. Intended for users with firewall / reverse-proxy /
         tunnel setups that need a pinned port. If it's occupied at
         launch, the server will fail to bind — the user asked for it,
         we don't second-guess.
      2. If `preferred` is free, return it.
      3. Walk `preferred + 1 .. preferred + scan_range` and return the
         first free port.
      4. Fall back to an OS-assigned ephemeral port via bind(0). This
         is the hard guarantee that localcode can always find a port,
         even on a machine where 8081-8099 are all taken by other
         services.

    Pure helper — does not kill anything. Callers that want to reclaim
    the preferred port from a stale process should do that separately
    before calling this (see `ServerManager._pick_free_port`).
    """
    import os
    forced = os.environ.get(PORT_ENV_VAR, "").strip()
    if forced:
        try:
            return int(forced)
        except ValueError:
            # Malformed env var — fall through to normal selection
            # rather than failing. A bad env var shouldn't brick the app.
            pass
    if not _port_in_use(preferred):
        return preferred
    for offset in range(1, scan_range + 1):
        candidate = preferred + offset
        if not _port_in_use(candidate):
            return candidate
    return _os_ephemeral_port()


def _rewrite_port_arg(cmd: list[str], port: int) -> list[str]:
    """Return a copy of `cmd` with `--port <N>` set to `port`."""
    out = list(cmd)
    try:
        i = out.index("--port")
        out[i + 1] = str(port)
    except (ValueError, IndexError):
        pass
    return out


STUCK_SERVER_MARKER = stuck_server_marker_path()

# Persistent (append-only) lifecycle log for every server start/stop/kill/
# recovery event. Per-project (`<project_root>/.localcode/lifecycle.log`) so
# each project keeps its own diagnostic timeline across sessions.
LIFECYCLE_LOG = lifecycle_log_path()


def _lifecycle_log(event: str, **fields) -> None:
    """Compatibility shim — routes to the central event sink.

    All historical call sites in this module + tools/__init__.py + agent.py
    + setup.py emit through `_lifecycle_log(event, **fields)`. This shim
    forwards them to `events.emit(event, **fields)` so we get ONE
    structured JSONL log (`.localcode/events.jsonl`) instead of the
    text "key=value" lifecycle.log + JSONL turns.jsonl + marker file
    sprawl. See `src/localcode/events.py` for the full schema.

    The old `lifecycle.log` text file is no longer written. Callers
    that want to inspect events should use `events.jsonl` (parseable
    with `jq`) or call `events.read_events()` / `events.find_recent()`.
    """
    try:
        from .events import emit
        emit(event, **fields)
    except Exception:
        pass


def _system_free_memory_mb() -> int:
    """Return approximate system-wide free memory in MiB.

    Uses `vm_stat` (already in PATH on every Mac). Returns 0 on any
    parse failure — caller should treat 0 as "couldn't tell, proceed
    with caution" rather than "definitely zero free."

    Free memory = (free + inactive) * page_size. Inactive pages are
    cached file content the kernel can release on demand, so they
    behave as free for a new large allocation. Wired and active are
    not counted (they belong to running apps that need them).
    """
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
        page_size = 16384  # M-series default; vm_stat header confirms
        free_pages = 0
        inactive_pages = 0
        for line in r.stdout.splitlines():
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                try:
                    page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                except Exception:
                    pass
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        if free_pages == 0 and inactive_pages == 0:
            return 0
        total_bytes = (free_pages + inactive_pages) * page_size
        return total_bytes // (1024 * 1024)
    except Exception:
        return 0


def _wait_for_memory_release(min_free_mb: int, timeout_s: float = 8.0) -> bool:
    """Block until system free memory ≥ `min_free_mb` or `timeout_s` elapses.

    Used between killing an old server and spawning a new one, so that
    we don't double-commit ~10 GB while the kernel is still releasing
    the dead server's wired Metal pages. Returns True if the threshold
    was met, False on timeout. Polls every 250 ms.

    A False return is informational, not fatal — caller can decide to
    proceed anyway (the new server might still fit, just at higher
    pressure-kill risk). Logs the outcome so we have evidence later.
    """
    import time
    deadline = time.monotonic() + timeout_s
    last_seen = -1
    while time.monotonic() < deadline:
        free_mb = _system_free_memory_mb()
        last_seen = free_mb
        if free_mb >= min_free_mb:
            _lifecycle_log("memory_released", free_mb=free_mb, target_mb=min_free_mb)
            return True
        time.sleep(0.25)
    _lifecycle_log("memory_release_timeout", free_mb=last_seen, target_mb=min_free_mb,
                   timeout_s=timeout_s)
    return False


def _mark_stuck_server(pid: int) -> None:
    """Persist a note that PID `pid` wouldn't die under SIGKILL.

    Read by `localcode.health.find_stuck_servers()` on the next launch.
    Cleared when health check confirms PID is gone.
    """
    try:
        STUCK_SERVER_MARKER.parent.mkdir(parents=True, exist_ok=True)
        STUCK_SERVER_MARKER.write_text(f"{pid}\n{time.time():.0f}\n")
    except Exception:
        pass


class ServerManager:
    """Owns the llama-server subprocess. Use `ServerManager.get()`."""

    _instance: Optional["ServerManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ServerManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
                atexit.register(cls._instance.shutdown)
            return cls._instance

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._model_path: Optional[str] = None
        self._port: int = DEFAULT_PORT
        self._pressure_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_exit_code: Optional[int] = None  # disconnect diagnostics
        self._last_death_was_pressure: bool = False
        # Idle auto-suspend: a watchdog shuts the server down after
        # `_idle_timeout_s` of inactivity (stops the GPU cooking the laptop
        # while the user reads); the next chat call respawns via
        # _restart_server. 0 disables; default 10 min. Env: LOCALCODE_IDLE_SUSPEND_S.
        import os as _os
        try:
            self._idle_timeout_s: float = float(
                _os.environ.get("LOCALCODE_IDLE_SUSPEND_S", "600"),
            )
        except ValueError:
            self._idle_timeout_s = 600.0
        self._last_activity_ts: float = 0.0
        self._idle_thread: Optional[threading.Thread] = None
        # Post-start verification: what the SERVER says it loaded, and the
        # complaint if that disagreed with what we asked for. Never trust
        # `_model_path` (what we *wanted*) as evidence of what is serving.
        self._verified_model: Optional[str] = None
        self._verification_error: Optional[str] = None
        # On construction, clean up any stale PID file from a prior crash.
        self._reap_stale_pid_file()

    def mark_activity(self) -> None:
        """Record that the server just handled (or is about to handle)
        a request. Resets the idle-suspend countdown."""
        import time as _t
        self._last_activity_ts = _t.time()

    def request_started(self) -> None:
        """Mark a chat request in flight. The idle watchdog must NEVER
        suspend the server while a request is live: a single long agentic
        round (a 10+ minute decode is routine for a 30B+ model on local
        hardware) used to count as "idle" because activity was only marked at
        request START — observed live as `idle_suspend idle_s=612` firing
        mid-build, shutting the server down under an active stream and
        cascading into back-to-back reloads (35 GB of weights each) and a dead
        session. Pair with request_finished() in a finally block."""
        with self._lock:
            self._inflight = getattr(self, "_inflight", 0) + 1
        self.mark_activity()

    def request_finished(self) -> None:
        """Mark a chat request complete (see request_started)."""
        with self._lock:
            self._inflight = max(0, getattr(self, "_inflight", 0) - 1)
        self.mark_activity()

    def set_idle_timeout(self, seconds: float) -> None:
        """Update the idle auto-suspend window. 0 disables suspend."""
        self._idle_timeout_s = max(0.0, float(seconds))

    def _idle_watchdog(self) -> None:
        import time as _t
        while True:
            _t.sleep(30)
            try:
                if self._idle_timeout_s <= 0:
                    continue
                if self._process is None or self._process.poll() is not None:
                    continue
                if self._last_activity_ts == 0.0:
                    continue
                # A live request is never idle, no matter how long its decode
                # runs. This is the primary guard; the streaming path also
                # marks activity per chunk as a belt-and-braces backstop.
                if getattr(self, "_inflight", 0) > 0:
                    continue
                idle = _t.time() - self._last_activity_ts
                if idle >= self._idle_timeout_s:
                    _lifecycle_log(
                        "idle_suspend",
                        idle_s=round(idle, 1),
                        threshold_s=self._idle_timeout_s,
                    )
                    self.shutdown(reason="idle_suspend")
            except Exception:
                # Never let the watchdog crash the process.
                pass

    def _ensure_idle_thread(self) -> None:
        if self._idle_thread is not None and self._idle_thread.is_alive():
            return
        self._idle_thread = threading.Thread(
            target=self._idle_watchdog,
            name="lc-idle-suspend",
            daemon=True,
        )
        self._idle_thread.start()

    @property
    def port(self) -> int:
        """Port the live server is actually listening on. May differ
        from DEFAULT_PORT (8081) if the default was held by a stuck
        process at start time and we fell back to 8082 / 8083 / etc.
        Callers MUST read this AFTER start()/restart() and update
        their HTTP client base URL — without that, requests go to the
        old default port and fail with connection-refused while a
        perfectly good server runs on the fallback. (See RESUME.md
        port-isolation note.)"""
        return self._port

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def start(self, cmd: list[str], model_path: str, port: int = DEFAULT_PORT,
              timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
        """Start the server. Kills any pre-existing instance first. Blocks
        until the health endpoint returns OK or `timeout_s` elapses.
        Returns True on success, False on healthcheck timeout.

        Port fallback: if the default port is held by a D-state ("stuck
        in a Metal syscall that won't release") process that can't be
        killed, incrementally try 8082, 8083 … up to 8085. Without this,
        a single hung llama-server from a prior session blocks every
        subsequent localcode launch until reboot — which was user-
        reported on 2026-04-23.
        """
        requested_port = port
        with self._lock:
            # Nothing is verified until the post-health probe below says so.
            self._verified_model = None
            self._verification_error = None
            had_prior = self._process is not None
            _prev_model = self._model_path  # capture before shutdown clears it
            self._shutdown_locked(reason="start_or_model_swap")
            # Wait for the OLD server's memory to release before spawning the
            # new one (kernel takes 1-3 s to free wired Metal memory; spawning
            # into that window double-commits). Best-effort: ~11 GB target, 8 s.
            if had_prior:
                # Only a genuine model change is "model_switch"; a same-model
                # relaunch (crash/disconnect recovery) is "server_restart".
                _reason = ("model_switch" if (model_path and model_path != _prev_model)
                           else "server_restart")
                _lifecycle_log("memory_wait_start", reason=_reason, target_mb=11000)
                _wait_for_memory_release(min_free_mb=11000, timeout_s=8.0)

            # Find a port we can actually bind to. If the default is
            # held by an undead process, pick the next free one and
            # rewrite the --port argument in cmd.
            port = self._pick_free_port(port)
            cmd = _rewrite_port_arg(cmd, port)

            # Single source of truth for env scrubbing — see _subproc_env.py.
            # Strips MallocStackLogging* + MallocNanoZone, the libsystem-init
            # warning sources we battled all of 2026-04-26.
            env = clean_env()
            env["GGML_BACKEND_PATH"] = ""
            # Relax macOS GPU interactivity watchdog (AGX CDM context-store
            # timeout). llama.cpp upstream PR #22216 sets this in-process, but
            # AGX driver may read the env at first Metal touch — setting it in
            # the parent env before spawn is the belt-and-suspenders approach.
            env.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")
            # Disable Metal residency sets on at-risk launches → prevents the wired-cap SIGKILL.
            try:
                from .memory_guard import metal_residency_env
                _renv = metal_residency_env(str(model_path), cmd)
                if _renv:
                    env.update(_renv)
                    _lifecycle_log("metal_no_residency", model=str(model_path))
            except Exception:
                pass
            # Capture the server's OWN stdout/stderr to server.log (this runtime
            # path used to send both to /dev/null, discarding llama-server's
            # final ggml/Metal error on the recurring -9). Append per-spawn.
            log_fh = None
            try:
                from .paths import global_state_dir
                log_fh = open(global_state_dir() / "server.log", "a", buffering=1)
                log_fh.write(f"\n===== llama-server spawn (port {port}) =====\n")
            except Exception:
                log_fh = None
            try:  # close any prior spawn's handle before replacing it
                if getattr(self, "_log_fh", None) is not None:
                    self._log_fh.close()
            except Exception:
                pass
            self._log_fh = log_fh
            self._process = subprocess.Popen(
                cmd,
                stdout=(log_fh or subprocess.DEVNULL),
                stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
                start_new_session=True,  # process group leader → killpg works
                env=env,
            )
            self._model_path = model_path
            self._port = port

            # Layer 1: kernel-enforced memory ceiling via jetsam — the hard
            # backstop (sanctioned Apple API) that kills llama-server before it
            # can push the system over the edge. ONLY installed on small Macs
            # (<48 GB): there a runaway server must be killed before it freezes
            # the machine. On a large Mac a big model legitimately needs most of
            # RAM, and a FATAL highwater there just SIGKILLs (-9) the server
            # mid-decode when its footprint spikes past total−reserve — the
            # mystery long-task crash. Big machines rely on Layer 2 (graceful).
            try:
                from .memory_guard import (
                    set_jetsam_highwater, recommended_jetsam_limit_mb,
                    should_set_fatal_jetsam_limit, start_pressure_monitor,
                )
                if should_set_fatal_jetsam_limit():
                    _limit = recommended_jetsam_limit_mb()
                    set_jetsam_highwater(self._process.pid, _limit)
                    _lifecycle_log("jetsam_highwater_set", pid=self._process.pid,
                                   limit_mb=_limit)
                else:
                    _lifecycle_log("jetsam_highwater_skipped",
                                   pid=self._process.pid,
                                   reason="large_ram_uses_pressure_monitor")
            except Exception:
                pass

            # Layer 2: userspace pressure monitor — polls
            # kern.memorystatus_vm_pressure_level every 500 ms and SIGTERMs our
            # server on WARN. Complements the jetsam ceiling (acts on system-
            # wide pressure, not just our own RSS).
            try:
                self._pressure_thread = start_pressure_monitor(
                    self._process,
                    on_pressure=self._on_pressure_kill,
                )
            except Exception:
                self._pressure_thread = None
            self._write_pid_file(self._process.pid)
            # Log the FULL launch command (which flags was the server using?);
            # truncated to 4000 chars (~80 args) to keep the event small.
            _flags_str = " ".join(str(c) for c in cmd)[:4000]
            _lifecycle_log("server_started", pid=self._process.pid, port=port,
                           model=Path(model_path).name,
                           free_mb_after_spawn=_system_free_memory_mb(),
                           flags=_flags_str)
        ok = self._wait_healthy(port, timeout_s)
        _lifecycle_log("server_health_result", port=port, healthy=ok,
                       free_mb=_system_free_memory_mb())
        if ok:
            # A healthy port is NOT proof the model we asked for is loaded.
            # `_shutdown_locked()` only kills a process THIS manager owns, and
            # `_kill_port()` deliberately refuses to kill a healthy foreign
            # llama-server (it belongs to another terminal's session) — so a
            # start() can sail through the healthcheck while a prior session's
            # server answers on the port. Live on 2026-08-22: /model switched
            # to Muse Glimmer, UI said "server: ready", and pid 49152 was still
            # serving Qwen3.8-27B. Ask the server what it loaded and FAIL if it
            # isn't what we asked for. We do NOT kill the foreign server (that
            # would end someone else's session) — we detect and report.
            ok = self._verify_loaded_model(model_path, port,
                                           requested_port=requested_port)
        if ok:
            self.mark_activity()
            self._ensure_idle_thread()
            self._last_exit_code = None
            self._last_death_was_pressure = False
        return ok

    def _verify_loaded_model(self, model_path: str, port: int,
                             requested_port: int | None = None) -> bool:
        """Confirm the live server on `port` really loaded `model_path`.

        Returns False (and records `verification_error`) only on a genuine
        MISMATCH. An unknown answer — older server, `/props` unreachable —
        leaves `verified_model` None and returns True: we can't prove it's
        wrong, but callers/UI must then say "unverified" rather than name a
        model confidently.

        Port fallback is the same class of bug: if we bound 8082 because a
        foreign server held 8081, every caller still pointed at 8081 is
        talking to that foreign server. We do not fail here — the existing
        design is that callers re-read `mgr.port` after start() and rewrite
        their base_url (runtime._restart_server does exactly this), which is
        the recovery that keeps localcode usable when a port is squatted.
        We log it loudly so a caller that forgot is diagnosable.
        """
        if requested_port is not None and port != requested_port:
            _lifecycle_log("server_port_fallback", requested=requested_port,
                           actual=port,
                           note="callers must re-read mgr.port into base_url")
        reported = probe_loaded_model(port, timeout=2.0)
        if reported is None:
            self._verified_model = None
            self._verification_error = None
            _lifecycle_log("model_verify_unknown", port=port,
                           requested=Path(model_path).name)
            return True
        if model_identity_matches(model_path, reported):
            self._verified_model = reported
            self._verification_error = None
            _lifecycle_log("model_verify_ok", port=port,
                           model=Path(reported).name)
            return True
        self._verified_model = reported
        self._verification_error = (
            f"Wrong model on port {port}: asked for {model_path}, "
            f"server is serving {reported}. A llama-server from another "
            f"session is holding this port (localcode never kills a healthy "
            f"one). Quit that session, or set LOCALCODE_PORT to a free port."
        )
        _lifecycle_log("model_verify_mismatch", port=port,
                       requested=model_path, actual=reported)
        return False

    def restart(self, cmd: list[str], model_path: str, port: int = DEFAULT_PORT,
                timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
        """Convenience: shutdown + start. Callers switching models should
        use this rather than open-coding kill-then-start, to ensure we go
        through the single lifecycle path.
        """
        return self.start(cmd, model_path, port=port, timeout_s=timeout_s)

    def shutdown(self, force: bool = False, reason: str = "") -> None:
        """Stop the server. Idempotent; safe to call multiple times.

        ``force=False`` (default, used for model swaps): graceful path —
        SIGTERM → wait up to 3 s for clean shutdown → SIGKILL fallback.
        Lets llama-server release its Metal allocations cleanly so the
        replacement model can claim them without an OOM.

        ``force=True`` (used on app exit): straight to SIGKILL with a
        0.5 s reap timeout. The kernel reclaims the Metal allocations
        on process death anyway, and skipping the 5–10 s graceful
        cleanup is the difference between a snappy exit and the user
        wondering if localcode is hung.

        ``reason`` is recorded on the server_stopped lifecycle event so a
        restart storm is diagnosable from events.jsonl alone.
        """
        with self._lock:
            if force:
                self._pending_stop_reason = reason or "app_exit_force"
                self._force_kill_locked()
            else:
                self._shutdown_locked(reason=reason)

    def _force_kill_locked(self) -> None:
        """SIGKILL-immediately path for app exit. Must be called with
        self._lock held."""
        if self._process is None:
            return
        pid = self._process.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        try:
            self._process.wait(timeout=0.5)
        except Exception:
            pass
        self._process = None
        self._model_path = None
        self._port = None
        self._verified_model = None
        self._verification_error = None

    def is_healthy(self, port: int | None = None) -> bool:
        """Non-blocking probe of the port the server ACTUALLY bound (default was 8081, wrong after port fallback)."""
        return _probe_health(port if port is not None else self._port, timeout=1.0)

    def is_running(self) -> bool:
        """True if we have a tracked Popen and it hasn't exited."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def current_model(self) -> Optional[str]:
        """The model we ASKED the server to load. Not evidence of what is
        actually serving — see `verified_model`."""
        return self._model_path

    @property
    def verified_model(self) -> Optional[str]:
        """The model path the live server REPORTED after the last start(),
        or None when verification hasn't run or couldn't answer."""
        return self._verified_model

    @property
    def verification_error(self) -> Optional[str]:
        """Human-readable complaint when the server is serving a different
        model than the one requested. None when verified or unknown."""
        return self._verification_error

    def disconnect_diagnostics(self) -> dict:
        """Snapshot of WHY the server is unreachable (disconnect CLASS:
        memory-guard kill / crash / not-running / wedged) for recovery
        logging. exit_code is `-signal` on POSIX. Never raises."""
        running = False
        try:
            if self._process is not None:
                rc = self._process.poll()
                running = rc is None
                if rc is not None:
                    self._last_exit_code = rc
        except Exception:
            pass
        ec = None if running else self._last_exit_code
        cls = (
            "memory_guard_kill" if self._last_death_was_pressure
            else "sigkill_or_jetsam" if ec == -9
            else "sigterm" if ec == -15
            else "crash_exit" if isinstance(ec, int) and ec > 0
            else "wedged_listener" if running else "not_running"
        )
        return {"running": running, "exit_code": ec,
                "pressure_kill": bool(self._last_death_was_pressure),
                "disconnect_class": cls, "port": self._port,
                "free_mb": _system_free_memory_mb()}

    # ────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────

    def _shutdown_locked(self, reason: str = "") -> None:
        """Must be called with self._lock held.

        Supervisor pattern: SIGTERM → wait 3s → SIGKILL → wait 2s. If
        BOTH waits time out, the process is in D-state (uninterruptible
        kernel sleep — Metal mmap/GPU). We record that in a marker file
        so the next localcode launch's health check can detect it and
        refuse to start, instead of letting the OS OOM-kill the user.

        `reason` lands on the server_stopped lifecycle event. The restart
        storm of 2026-07-12 (7 stops in 15 min) was invisible precisely
        because stops carried no reason — never emit a bare stop again.
        """
        self._pending_stop_reason = reason or getattr(self, "_pending_stop_reason", "") or "unspecified"
        if self._process is not None:
            pid = self._process.pid
            stuck = False
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Both signals delivered but process won't
                        # exit — D-state. Mark it for the next launch.
                        stuck = True
            except ProcessLookupError:
                pass
            except Exception:
                # Last ditch: plain SIGKILL on the child
                try:
                    self._process.kill()
                except Exception:
                    pass
            if stuck:
                _mark_stuck_server(pid)
            _lifecycle_log("server_stopped", pid=pid, stuck=stuck,
                           reason=getattr(self, "_pending_stop_reason", "unspecified"),
                           free_mb_at_stop=_system_free_memory_mb())
            self._pending_stop_reason = ""
            self._process = None
            self._model_path = None

        # 2. Kill whatever the PID file points at (covers prior Python process)
        self._kill_pid_file()

        # 3. Last-resort: anything on our port (covers other-user llama-server,
        #    detached orphans, etc.)
        self._kill_port(self._port)

    def _reap_stale_pid_file(self) -> None:
        """Reconcile the PID file on manager construction.

        HARD-LEARNED (2026-07-12, live): this used to killpg(SIGKILL)
        whatever pid the file named, unverified. Consequences observed the
        same day: (a) launching a second localcode killed the first
        session's HEALTHY server mid-build, twice, ending the user's run;
        (b) under restart churn a recycled pid let the killpg land on the
        TUI's own process group — the whole app died with `zsh: killed`
        and 60 GB free. A world-shared pid file must never be treated as
        a kill list. Now: only a VERIFIED llama-server (see
        `_pid_is_our_llama_server`) that is NOT serving healthily gets
        killed; anything else is left alone and only the file is removed.
        """
        self._kill_pid_file()

    @staticmethod
    def _pid_is_our_llama_server(pid: int) -> bool:
        """True iff `pid` is alive AND its command line is a llama-server.

        The pgid==pid check exploits our own spawn signature
        (start_new_session=True makes every server we launch its own
        process-group leader). A recycled pid belonging to some other
        program fails the name check; a process in someone else's group
        (e.g. the TUI's) fails the leader check. Both must pass before we
        are willing to signal it.
        """
        try:
            if os.getpgid(pid) != pid:
                return False
        except Exception:
            return False
        try:
            out = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        except Exception:
            return False
        return "llama-server" in out or "llama-diffusion" in out

    def _kill_pid_file(self) -> None:
        if not PID_FILE.exists():
            return
        pid, _file_port = 0, 0
        try:
            parts = PID_FILE.read_text().strip().split()
            pid = int(parts[0]) if parts else 0
            _file_port = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            pid = 0
        if pid > 0 and self._pid_is_our_llama_server(pid):
            # A healthy, serving llama-server is NOT stale — it belongs to a
            # live session (possibly this one, possibly another terminal).
            # Killing it ends that user's in-flight build. Leave it; the
            # setup screen's reuse/port-fallback logic handles coexistence.
            _check_port = _file_port or self._port
            if _probe_health(_check_port, timeout=1.0):
                _lifecycle_log("pid_reap_skipped_healthy", pid=pid, port=_check_port)
                return
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                _lifecycle_log("pid_reap_killed", pid=pid)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        elif pid > 0:
            _lifecycle_log("pid_reap_skipped_not_ours", pid=pid)
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _on_pressure_kill(self, level: int) -> None:
        """Called from the pressure-monitor thread when we decide to kill
        our llama-server to protect the system. Writes a breadcrumb file
        so the next launch's health check can tell the user what
        happened, and the TUI can also poll the file to surface the
        event in-session.

        ALSO clears `self._process` so callers that check `is_running()`
        don't think the server is still alive — the prior version
        forgot this and `_request_with_retry`'s "is_conn_err →
        _restart_server" path could race with `is_running` returning
        True for a process that had already been pressure-killed.
        """
        killed_pid = self._process.pid if self._process is not None else 0
        if self._process is not None:  # breadcrumb: OOM-guard reap, not a crash
            try:
                self._last_exit_code = self._process.poll()
            except Exception:
                pass
        self._last_death_was_pressure = True
        try:
            marker = pressure_kill_marker_path()
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"level={level}\ntime={time.time():.0f}\n")
        except Exception:
            pass
        # Logged DISTINCTLY as `pressure_kill` (OOM-guard reap vs a crash).
        _lifecycle_log("pressure_kill", level=level, pid=killed_pid,
                       guard="memory_pressure", exit_code=self._last_exit_code,
                       free_mb=_system_free_memory_mb())
        # Clear internal state so callers see "no server" not a phantom Popen.
        # Safe from the pressure-monitor thread (assignment is atomic in CPython).
        self._process = None
        self._model_path = None
        self._verified_model = None

    def _pick_free_port(self, preferred: int) -> int:
        """Reclaim `preferred` if we can, otherwise delegate to the
        shared `find_free_port()` selector (scan → ephemeral fallback).

        If $LOCALCODE_PORT is set, the user has pinned a port and we
        skip the preferred-port kill dance entirely — they want exactly
        that port even if it means waiting for the holder to clear.
        """
        import os
        if os.environ.get(PORT_ENV_VAR, "").strip():
            return find_free_port(preferred)
        # Try to reclaim the preferred port from any prior holder
        # (our own stale server, a sibling install, etc.). If the
        # holder is D-state and won't die, `find_free_port` will walk
        # to a neighbour or ephemeral — localcode still launches.
        self._kill_port(preferred)
        time.sleep(0.3)
        return find_free_port(preferred)

    def _kill_port(self, port: int) -> None:
        """Kill a WEDGED llama-server bound to `port`. Best-effort; silent on
        failure (lsof not installed, permission denied, etc.).

        Two guards, both learned live (2026-07-12):
        - Only signal a process whose command line is a llama-server —
          this sweep must never SIGKILL an arbitrary process that happens
          to sit on our port (that's the user's dev server, another app…).
        - Never signal a HEALTHY llama-server: it belongs to a live session
          (another terminal). Killing it ends that session's build mid-
          flight; the caller should use port fallback (8082, 8083, …)
          instead. Only an unresponsive listener — the actual "wedged
          zombie blocks every launch until reboot" case this sweep exists
          for — gets killed.
        """
        if _probe_health(port, timeout=1.0):
            _lifecycle_log("port_kill_skipped_healthy", port=port)
            return
        try:
            r = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=3,
            )
            for pid_str in r.stdout.split():
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                try:
                    out = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "command="],
                        capture_output=True, text=True, timeout=3,
                    ).stdout
                    if "llama-server" not in out and "llama-diffusion" not in out:
                        _lifecycle_log("port_kill_skipped_not_ours", port=port, pid=pid)
                        continue
                except Exception:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
        except Exception:
            pass

    def _write_pid_file(self, pid: int) -> None:
        try:
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            # "pid port" — the port lets the next launch's reaper health-probe
            # the RIGHT server before deciding it's stale (a bare pid forced it
            # to probe the default port and mis-kill fallback-port servers).
            PID_FILE.write_text(f"{pid} {self._port}")
        except Exception:
            # PID file is a recovery aid, not a correctness requirement.
            pass

    def _wait_healthy(self, port: int, timeout_s: int) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # If the subprocess already died, fail fast
            with self._lock:
                p = self._process
            if p is not None and p.poll() is not None:
                return False
            if _probe_health(port, timeout=1.0):
                return True
            # Poll at 0.25 s (was 1 s): the server can finish loading at any
            # point in the interval, so a coarse 1 s poll adds up to ~1 s of
            # dead wait to every warm-start's time-to-ready. 0.25 s cuts that
            # to ≤0.25 s; the probe itself is a cheap local /health GET.
            time.sleep(0.25)
        return False


def _probe_health(port: int, timeout: float = 1.0) -> bool:
    """Single HTTP health probe against the llama-server `/health` endpoint.

    Returns True **only** when the server responds 200 — i.e. the model is
    fully loaded and inference endpoints (`/v1/chat/completions`) will
    actually serve requests.

    While the model is loading, llama-server's pre-routing middleware
    intercepts every endpoint (including `/health`) and returns 503
    `{"error":{"message":"Loading model","code":503}}`. Treating 503 as
    "alive" here was the root cause of the E3102 bug: model switch would
    report "Server ready" the instant the HTTP listener came up, then the
    next user message would hit the still-loading model and get its own
    503 back as a user-facing "Lost connection" error.

    A transport-level failure (connection refused, timeout) also returns
    False — the server may be mid-restart or not yet listening.
    """
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            r.read()
            return r.status == 200
    except urllib.error.HTTPError:
        # 503 during load, 5xx on internal errors — not ready either way.
        return False
    except Exception:
        # Connection refused, read timeout — process is still coming up.
        return False


def probe_loaded_model(port: int, timeout: float = 2.0) -> Optional[str]:
    """Ask the server WHICH MODEL FILE IT ACTUALLY LOADED. None if unknown.

    Why this exists (live evidence, 2026-08-22): the user ran `/model` to
    switch to Muse Glimmer. The status bar said "model: Muse Glimmer 30B
    UD-Q8_K_XL", the model claimed to be Muse Glimmer — and the only
    llama-server on the machine was an hour-old pid serving
    `Qwen3.8-27B-UD-Q8_K_XL.gguf` on 8081. `/health` says "a server is up",
    it does NOT say "the server you asked for, with the model you asked
    for". Nothing in the stack ever asked the second question, so a start()
    that silently reused a foreign session's server reported "ready".

    Prefer `/props` (`model_path`, an absolute path). Fall back to
    `/v1/models` (`data[0].id`, usually just the basename). Cheap and
    total: short timeout, never raises — an unreachable or older server
    just yields None ("unknown"), which callers treat as unverified
    rather than as a mismatch.
    """
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(f"{base}/props", timeout=timeout) as r:
            if r.status == 200:
                import json as _json
                data = _json.loads(r.read().decode("utf-8", "replace"))
                mp = data.get("model_path") or data.get("model")
                if isinstance(mp, str) and mp.strip():
                    return mp.strip()
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
            if r.status == 200:
                import json as _json
                data = _json.loads(r.read().decode("utf-8", "replace"))
                entries = data.get("data") or []
                if entries:
                    mid = entries[0].get("id")
                    if isinstance(mid, str) and mid.strip():
                        return mid.strip()
    except Exception:
        pass
    return None


def model_identity_matches(requested: str, reported: str) -> bool:
    """True if `reported` names the same GGUF as `requested`.

    Compare resolved BASENAMES, case-sensitively: the server reports the
    path it was launched with, which may be absolute where we hold a
    relative path (or vice versa), but the filename is the model's
    identity. Case-sensitive on purpose — `Qwen3.8-27B-UD-Q8_K_XL.gguf`
    and a differently-cased near-miss are different files on a
    case-sensitive volume, and silently equating them is the exact class
    of "wrong model, confident UI" bug this guards.
    """
    if not requested or not reported:
        return False
    return Path(str(requested)).name == Path(str(reported)).name
