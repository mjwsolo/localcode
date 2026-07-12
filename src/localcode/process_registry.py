"""Durable registry for app processes launched by LocalCode.

The registry is intentionally generic: it records process identity and
health metadata, not stack-specific application state. Launchers and tools can
reuse it to avoid port churn and orphaned background processes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request


@dataclass
class ProcessRecord:
    pid: int
    pgid: int
    port: int
    url: str
    cwd: str
    kind: str
    command: str
    log_path: str
    verified: bool
    started_at: float
    stopped_at: float = 0.0
    process_id: str = ""
    owner: str = "localcode"
    exit_code: int | None = None

    def __post_init__(self) -> None:
        if not self.process_id:
            raw = f"{self.started_at}:{self.pid}:{self.command}".encode()
            self.process_id = "proc-" + hashlib.sha256(raw).hexdigest()[:12]


def registry_path(repo_root: Path | str) -> Path:
    root = Path(repo_root)
    return root / ".localcode" / "processes.json"


def load_records(repo_root: Path | str) -> list[ProcessRecord]:
    path = registry_path(repo_root)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    records: list[ProcessRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            records.append(ProcessRecord(**item))
        except Exception:
            continue
    return records


def save_records(repo_root: Path | str, records: list[ProcessRecord]) -> None:
    path = registry_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records[-50:]], indent=2))


def record_process(repo_root: Path | str, record: ProcessRecord) -> None:
    records = load_records(repo_root)
    now = time.time()
    for existing in records:
        if existing.stopped_at:
            continue
        if existing.pid == record.pid and existing.started_at == record.started_at:
            continue
        if existing.cwd == record.cwd and existing.kind == record.kind and record.kind != "background":
            existing.stopped_at = now
        elif existing.pid > 0 and not _pid_alive(existing.pid):
            existing.stopped_at = now
    records.append(record)
    save_records(repo_root, records)


def latest_live_record(repo_root: Path | str) -> ProcessRecord | None:
    records = load_records(repo_root)
    changed = False
    now = time.time()
    for record in reversed(records):
        if record.stopped_at:
            continue
        if record.pid <= 0:
            continue
        if not _pid_alive(record.pid):
            record.stopped_at = now
            changed = True
            continue
        if record.url and _url_healthy(record.url):
            if changed:
                save_records(repo_root, records)
            return record
        # A live PID is not enough. The user needs a reachable app URL;
        # treating any alive server process as healthy caused stale-port
        # reuse and false "app is running" reports.
        record.verified = False
        changed = True
    if changed:
        save_records(repo_root, records)
    return None


def find_record(repo_root: Path | str, process_id: str) -> ProcessRecord | None:
    return next((r for r in reversed(load_records(repo_root)) if r.process_id == process_id), None)


def refresh_record(repo_root: Path | str, process_id: str) -> ProcessRecord | None:
    records = load_records(repo_root)
    record = next((r for r in records if r.process_id == process_id), None)
    if record is None:
        return None
    if not record.stopped_at and record.pid > 0 and not _pid_alive(record.pid):
        record.stopped_at = time.time()
        save_records(repo_root, records)
    return record


def stop_record(repo_root: Path | str, record: ProcessRecord) -> bool:
    stopped = False
    target = record.pgid or record.pid
    try:
        os.killpg(target, signal.SIGTERM)
        stopped = True
    except ProcessLookupError:
        stopped = True
    except Exception:
        try:
            os.kill(record.pid, signal.SIGTERM)
            stopped = True
        except Exception:
            stopped = False
    records = load_records(repo_root)
    now = time.time()
    for existing in records:
        if existing.pid == record.pid and existing.started_at == record.started_at:
            existing.stopped_at = now
    save_records(repo_root, records)
    return stopped


def process_summary(repo_root: Path | str) -> str:
    records = load_records(repo_root)
    if not records:
        return "No LocalCode-managed processes recorded."
    lines = ["LocalCode-managed processes:"]
    changed = False
    now = time.time()
    for record in records[-20:]:
        alive = record.pid > 0 and _pid_alive(record.pid)
        healthy = bool(record.url and alive and _url_healthy(record.url))
        if not alive and not record.stopped_at:
            record.stopped_at = now
            changed = True
        status = "stopped" if record.stopped_at else ("healthy" if healthy else ("alive-unhealthy" if alive else "dead"))
        lines.append(
            f"- {status}: pid={record.pid} port={record.port} kind={record.kind} "
            f"url={record.url or '-'} cwd={record.cwd}"
        )
    if changed:
        save_records(repo_root, records)
    return "\n".join(lines)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _url_healthy(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False
