"""Gem daemon — keeps the local model hot and serves background tasks."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AppConfig, ensure_home_dirs, load_config
from .models import get_runtime_model, resolve_profile
from .runtime import GemRuntimeGateway

DAEMON_DIR = ensure_home_dirs() / "daemon"
PID_FILE = DAEMON_DIR / "gem.pid"
TASK_DIR = DAEMON_DIR / "tasks"
LOG_FILE = DAEMON_DIR / "daemon.log"
KEEPALIVE_INTERVAL = 120  # seconds — ping model to stay loaded


def _ensure_dirs() -> None:
    DAEMON_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)


# ── PID management ───────────────────────────────────────────────────────

def is_running() -> tuple[bool, int | None]:
    """Check if a daemon is already running."""
    _ensure_dirs()
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = check if process exists
        return True, pid
    except (ProcessLookupError, ValueError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return False, None


def _write_pid() -> None:
    _ensure_dirs()
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


# ── Daemon log ───────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    _ensure_dirs()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


def read_daemon_log(tail: int = 40) -> str:
    if not LOG_FILE.exists():
        return "(no daemon log)"
    lines = LOG_FILE.read_text().splitlines()
    return "\n".join(lines[-tail:])


# ── Task queue (file-based for simplicity) ───────────────────────────────

@dataclass
class DaemonTask:
    task_id: str
    prompt: str
    cwd: str
    status: str = "pending"  # pending | running | done | failed
    result: str = ""
    created_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "cwd": self.cwd,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> DaemonTask:
        return cls(**{k: data.get(k, "") for k in cls.__dataclass_fields__})


def submit_task(prompt: str, cwd: str) -> str:
    """Submit a task to the daemon queue. Returns task_id."""
    _ensure_dirs()
    task_id = f"claw-{int(time.time() * 1000)}"
    task = DaemonTask(
        task_id=task_id,
        prompt=prompt,
        cwd=cwd,
        status="pending",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    task_path = TASK_DIR / f"{task_id}.json"
    task_path.write_text(json.dumps(task.to_dict(), indent=2))
    return task_id


def list_tasks(limit: int = 20) -> list[DaemonTask]:
    """List recent tasks."""
    _ensure_dirs()
    tasks = []
    for path in sorted(TASK_DIR.glob("claw-*.json"), reverse=True)[:limit]:
        try:
            tasks.append(DaemonTask.from_dict(json.loads(path.read_text())))
        except Exception:
            continue
    return tasks


def get_task(task_id: str) -> DaemonTask | None:
    path = TASK_DIR / f"{task_id}.json"
    if not path.exists():
        return None
    try:
        return DaemonTask.from_dict(json.loads(path.read_text()))
    except Exception:
        return None


def _update_task(task: DaemonTask) -> None:
    path = TASK_DIR / f"{task.task_id}.json"
    path.write_text(json.dumps(task.to_dict(), indent=2))


# ── Keepalive — ping the model periodically ──────────────────────────────

class ModelKeepAlive:
    """Periodically pings Ollama to keep the model loaded in memory."""

    def __init__(self, gateway: GemRuntimeGateway) -> None:
        self.gateway = gateway
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                # Send a minimal request to keep the model loaded
                self.gateway.chat_once(
                    [{"role": "user", "content": "ping"}],
                )
                _log("keepalive: model pinged successfully")
            except Exception as exc:
                _log(f"keepalive: ping failed — {exc}")
            self._stop.wait(KEEPALIVE_INTERVAL)


# ── Task runner ──────────────────────────────────────────────────────────

class TaskRunner:
    """Watches the task directory and runs pending tasks."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._process_pending()
            except Exception as exc:
                _log(f"task runner error: {exc}")
            self._stop.wait(2)  # poll every 2 seconds

    def _process_pending(self) -> None:
        for path in sorted(TASK_DIR.glob("claw-*.json")):
            try:
                task = DaemonTask.from_dict(json.loads(path.read_text()))
            except Exception:
                continue
            if task.status != "pending":
                continue
            self._run_task(task)

    def _run_task(self, task: DaemonTask) -> None:
        task.status = "running"
        _update_task(task)
        _log(f"task {task.task_id}: started — {task.prompt[:80]}")

        try:
            # Late import to avoid circular dependency
            from .app import GemApp
            from .agent import AgentRunner

            app = GemApp(
                config=self.config,
                cwd=Path(task.cwd),
            )
            try:
                outcome = AgentRunner(app).run(task.prompt, auto_verify=True)
                task.result = outcome.answer
                if outcome.verification_output:
                    task.result += f"\n\n--- Verification (exit {outcome.verification_code}) ---\n{outcome.verification_output}"
                task.status = "done"
            finally:
                app.close()
        except Exception as exc:
            task.result = f"Error: {exc}"
            task.status = "failed"
            _log(f"task {task.task_id}: failed — {exc}")

        task.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        _update_task(task)
        _log(f"task {task.task_id}: {task.status}")


# ── Daemon main loop ─────────────────────────────────────────────────────

def run_daemon(config: AppConfig | None = None) -> int:
    """Run the daemon in the foreground. Returns exit code."""
    running, pid = is_running()
    if running:
        print(f"Daemon already running (pid {pid})")
        return 1

    if config is None:
        config = load_config()

    profile = resolve_profile(config.runtime.profile, config.runtime.model)
    config.runtime.model = get_runtime_model(profile, config.runtime.model)
    gateway = GemRuntimeGateway(config.runtime)

    # Verify model is reachable
    ok, detail = gateway.healthcheck()
    if not ok:
        print(f"Cannot start daemon: runtime unreachable — {detail}")
        return 1

    _write_pid()
    _log(f"daemon started (pid {os.getpid()}, model {config.runtime.model})")

    keepalive = ModelKeepAlive(gateway)
    task_runner = TaskRunner(config)

    def _shutdown(signum, frame):
        _log("daemon shutting down")
        keepalive.stop()
        task_runner.stop()
        gateway.close()
        _remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Prewarm: send an initial request to load the model
    _log("prewarming model...")
    try:
        gateway.chat_once([{"role": "user", "content": "hello"}])
        _log("prewarm complete — model loaded")
    except Exception as exc:
        _log(f"prewarm failed: {exc}")

    keepalive.start()
    task_runner.start()

    print(f"Gem daemon running (pid {os.getpid()}, model {config.runtime.model})")
    print("Press Ctrl+C to stop.")

    # Block forever
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        _shutdown(None, None)

    return 0


def stop_daemon() -> bool:
    """Stop a running daemon. Returns True if stopped."""
    running, pid = is_running()
    if not running or pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait a moment for clean shutdown
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        _remove_pid()
        return True
    except Exception:
        _remove_pid()
        return False


def daemon_status() -> dict[str, Any]:
    """Get daemon status info."""
    running, pid = is_running()
    tasks = list_tasks(5)
    pending = sum(1 for t in tasks if t.status == "pending")
    running_tasks = sum(1 for t in tasks if t.status == "running")
    return {
        "running": running,
        "pid": pid,
        "pending_tasks": pending,
        "running_tasks": running_tasks,
        "recent_tasks": len(tasks),
        "log_tail": read_daemon_log(10) if running else "",
    }
