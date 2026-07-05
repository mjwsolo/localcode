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

# PID file stays GLOBAL — only one llama-server runs on the machine
# (single port, single GPU memory budget). When you `cd` between
# projects mid-session, the running server stays serving you. The
# per-project lifecycle log records a fresh `server_started` entry
# on next launch from the new project. See paths.py for the full
# global-vs-project split rationale.
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

# Persistent (append-only) lifecycle log for every server start, stop,
# pressure-kill, and recovery event. Distinct from the per-session
# `~/.local/share/localcode/server.log` (which the setup screen
# overwrites with `open(..., "w")` on every launch and so loses
# inter-session context). Per-project (lives at
# `<project_root>/.localcode/lifecycle.log`) so each project has its
# own diagnostic timeline.
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
        # Idle auto-suspend: track wall-clock of the most recent chat
        # activity. A background watchdog thread shuts the server down
        # after `_idle_timeout_s` seconds of inactivity to stop the GPU
        # from cooking the laptop while the user is reading replies.
        # The next chat call transparently respawns via _restart_server.
        # 0 disables; default 10 min. Override with env LOCALCODE_IDLE_SUSPEND_S.
        import os as _os
        try:
            self._idle_timeout_s: float = float(
                _os.environ.get("LOCALCODE_IDLE_SUSPEND_S", "600"),
            )
        except ValueError:
            self._idle_timeout_s = 600.0
        self._last_activity_ts: float = 0.0
        self._idle_thread: Optional[threading.Thread] = None
        # On construction, clean up any stale PID file from a prior crash.
        self._reap_stale_pid_file()

    def mark_activity(self) -> None:
        """Record that the server just handled (or is about to handle)
        a request. Resets the idle-suspend countdown."""
        import time as _t
        self._last_activity_ts = _t.time()

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
                idle = _t.time() - self._last_activity_ts
                if idle >= self._idle_timeout_s:
                    _lifecycle_log(
                        "idle_suspend",
                        idle_s=round(idle, 1),
                        threshold_s=self._idle_timeout_s,
                    )
                    self.shutdown()
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
        with self._lock:
            had_prior = self._process is not None
            self._shutdown_locked()
            # Wait for the OLD server's memory to actually release
            # before spawning the new one. `process.wait()` returns
            # when the child reaps, but the kernel may take 1-3 s to
            # release wired Metal memory. Spawning the new ~10 GB
            # server in that window double-commits memory and reliably
            # triggers the pressure monitor on a 16 GB Mac (the bug
            # behind the user's "two servers running" hypothesis).
            #
            # Target: enough free memory to fit a typical model + 1 GB
            # margin (~11 GB for our standard quant). Best-effort —
            # times out at 8 s if the target isn't met, logs the
            # outcome, and proceeds anyway (the user wanted to switch
            # models; refusing entirely would be worse than retry).
            if had_prior:
                _lifecycle_log("memory_wait_start",
                               reason="model_switch", target_mb=11000)
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
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # process group leader → killpg works
                env=env,
            )
            self._model_path = model_path
            self._port = port

            # Layer 1: kernel-enforced memory ceiling via jetsam. If
            # llama-server tries to wire more than the budget allows,
            # macOS kernel kills it BEFORE the vm_fault wait-chain can
            # form. This is the hard backstop — sanctioned Apple API.
            try:
                from .memory_guard import (
                    set_jetsam_highwater, recommended_jetsam_limit_mb,
                    start_pressure_monitor,
                )
                set_jetsam_highwater(self._process.pid,
                                     recommended_jetsam_limit_mb())
            except Exception:
                pass

            # Layer 2: userspace memory-pressure monitor. Polls
            # kern.memorystatus_vm_pressure_level every 500 ms; on WARN
            # transition, SIGTERMs our server before the kernel stalls.
            # Complementary to the jetsam ceiling — acts on system-wide
            # pressure (everything else swelling) not just our own RSS.
            try:
                self._pressure_thread = start_pressure_monitor(
                    self._process,
                    on_pressure=self._on_pressure_kill,
                )
            except Exception:
                self._pressure_thread = None
            self._write_pid_file(self._process.pid)
            # Log the FULL command we launched with so future debugging
            # can answer "what flags was the server using during this
            # session?" — needed because subtle flags like
            # `--lookup-cache-dynamic` or `--spec-type ngram-mod` cause
            # repetition pathologies that look like model bugs but are
            # actually decode-time speculative-cache feedback loops.
            # Truncated to 4000 chars to keep the event small; that's
            # enough for ~80 args.
            _flags_str = " ".join(str(c) for c in cmd)[:4000]
            _lifecycle_log("server_started", pid=self._process.pid, port=port,
                           model=Path(model_path).name,
                           free_mb_after_spawn=_system_free_memory_mb(),
                           flags=_flags_str)
        ok = self._wait_healthy(port, timeout_s)
        _lifecycle_log("server_health_result", port=port, healthy=ok,
                       free_mb=_system_free_memory_mb())
        if ok:
            self.mark_activity()
            self._ensure_idle_thread()
            self._last_exit_code = None
            self._last_death_was_pressure = False
        return ok

    def restart(self, cmd: list[str], model_path: str, port: int = DEFAULT_PORT,
                timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
        """Convenience: shutdown + start. Callers switching models should
        use this rather than open-coding kill-then-start, to ensure we go
        through the single lifecycle path.
        """
        return self.start(cmd, model_path, port=port, timeout_s=timeout_s)

    def shutdown(self, force: bool = False) -> None:
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
        """
        with self._lock:
            if force:
                self._force_kill_locked()
            else:
                self._shutdown_locked()

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

    def is_healthy(self, port: int | None = None) -> bool:
        """Non-blocking probe of the port the server ACTUALLY bound (default was 8081, wrong after port fallback)."""
        return _probe_health(port if port is not None else self._port, timeout=1.0)

    def is_running(self) -> bool:
        """True if we have a tracked Popen and it hasn't exited."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def current_model(self) -> Optional[str]:
        return self._model_path

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

    def _shutdown_locked(self) -> None:
        """Must be called with self._lock held.

        Supervisor pattern: SIGTERM → wait 3s → SIGKILL → wait 2s. If
        BOTH waits time out, the process is in D-state (uninterruptible
        kernel sleep — Metal mmap/GPU). We record that in a marker file
        so the next localcode launch's health check can detect it and
        refuse to start, instead of letting the OS OOM-kill the user.
        """
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
                           free_mb_at_stop=_system_free_memory_mb())
            self._process = None
            self._model_path = None

        # 2. Kill whatever the PID file points at (covers prior Python process)
        self._kill_pid_file()

        # 3. Last-resort: anything on our port (covers other-user llama-server,
        #    detached orphans, etc.)
        self._kill_port(self._port)

    def _reap_stale_pid_file(self) -> None:
        """If the PID file exists and points to a live process, kill it. This
        runs on manager construction so that a fresh app launch starts from
        a clean slate even if the previous run was SIGKILL'd.
        """
        self._kill_pid_file()

    def _kill_pid_file(self) -> None:
        if not PID_FILE.exists():
            return
        try:
            raw = PID_FILE.read_text().strip()
            pid = int(raw) if raw else 0
        except Exception:
            pid = 0
        if pid > 0:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
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
        # Clear internal state so subsequent code paths see "no server
        # running" rather than a phantom Popen handle. Safe to do from
        # the pressure-monitor thread because `_process` reads/writes
        # are atomic in CPython for the assignment.
        self._process = None
        self._model_path = None

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
        """Kill anything bound to `port` via lsof. Best-effort; silent on
        failure (lsof not installed, permission denied, etc.).
        """
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
            PID_FILE.write_text(str(pid))
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
            time.sleep(1)
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
