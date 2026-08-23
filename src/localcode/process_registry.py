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
    """Per-repo registry file under the USER's state dir, not the repo.

    The registry used to live at `<repo>/.localcode/processes.json` — a file
    that ships INSIDE a cloned repo, so any repo could pre-seed records with
    attacker-chosen pid/pgid values that `stop_record` would then `killpg`.
    Keyed by a hash of the resolved repo path so different checkouts never
    collide.
    """
    root = Path(repo_root).resolve()
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    try:
        from .config import get_home_dir
        home = get_home_dir()
    except Exception:
        home = Path(os.environ.get("LOCALCODE_HOME") or (Path.home() / ".localcode"))
    return home / "processes" / f"{digest}.json"


# Pids (process-group leaders) spawned by THIS LocalCode process. Signals are
# only ever sent to members of this set: a registry record — however it got
# on disk — is never sufficient authority to kill a pid this session did not
# start.
_SESSION_SPAWNED_PIDS: set[int] = set()


def mark_spawned(pid: int) -> None:
    """Record that this process spawned `pid` (call right after Popen)."""
    try:
        if int(pid) > 1:
            _SESSION_SPAWNED_PIDS.add(int(pid))
    except Exception:
        pass


def spawned_this_session(pid: int) -> bool:
    try:
        return int(pid) in _SESSION_SPAWNED_PIDS
    except Exception:
        return False


def _coerce_record(item: dict) -> ProcessRecord | None:
    """Validate one persisted registry entry. Returns None on any bad shape.

    The file is durable state parsed with json — enforce field types instead
    of trusting `ProcessRecord(**item)` with whatever was on disk. pid/pgid
    are clamped to non-negative ints so a crafted record can never smuggle a
    negative value into a kill target (killpg(-N) signals an arbitrary group).
    """
    try:
        pid = int(item.get("pid", 0))
        pgid = int(item.get("pgid", 0))
        port = int(item.get("port", 0))
        if pid < 0 or pgid < 0:
            return None
        return ProcessRecord(
            pid=pid,
            pgid=pgid,
            port=max(0, min(port, 65535)),
            url=str(item.get("url", "") or ""),
            cwd=str(item.get("cwd", "") or ""),
            kind=str(item.get("kind", "") or ""),
            command=str(item.get("command", "") or ""),
            log_path=str(item.get("log_path", "") or ""),
            verified=bool(item.get("verified", False)),
            started_at=float(item.get("started_at", 0.0)),
            stopped_at=float(item.get("stopped_at", 0.0)),
            process_id=str(item.get("process_id", "") or ""),
            owner=str(item.get("owner", "localcode") or "localcode"),
        )
    except Exception:
        return None


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
        record = _coerce_record(item)
        if record is not None:
            records.append(record)
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
    # Only signal pids THIS process spawned. The registry file is durable
    # state — a crafted or stale record must never drive a killpg at an
    # arbitrary (or reused) pid. target > 1 also blocks killpg(0), which
    # would signal LocalCode's own process group.
    if target <= 1 or not (spawned_this_session(record.pid) or spawned_this_session(target)):
        return False
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
