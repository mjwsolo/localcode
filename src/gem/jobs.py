from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

from .config import ensure_home_dirs


@dataclass(slots=True)
class JobRecord:
    job_id: str
    command: str
    cwd: str
    pid: int
    created_at: str
    log_path: str


def _jobs_dir() -> Path:
    root = ensure_home_dirs() / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _job_log_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.log"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def launch_background_job(command: str, cwd: Path) -> JobRecord:
    job_id = uuid.uuid4().hex[:10]
    log_path = _job_log_path(job_id)
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    record = JobRecord(
        job_id=job_id,
        command=command,
        cwd=str(cwd),
        pid=process.pid,
        created_at=utc_now(),
        log_path=str(log_path),
    )
    from dataclasses import asdict
    _job_path(job_id).write_text(json.dumps(asdict(record), indent=2))
    return record


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def list_jobs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(_jobs_dir().glob("*.json"), reverse=True):
        data = json.loads(path.read_text())
        rows.append(
            {
                "job_id": data["job_id"],
                "command": data["command"],
                "cwd": data["cwd"],
                "created_at": data["created_at"],
                "status": "running" if _pid_alive(int(data["pid"])) else "finished",
                "log_path": data["log_path"],
            }
        )
    return rows


def read_job_log(job_id: str, max_chars: int = 16000) -> str:
    data = json.loads(_job_path(job_id).read_text())
    content = Path(data["log_path"]).read_text(errors="replace")
    if len(content) > max_chars:
        return content[-max_chars:]
    return content


def stop_job(job_id: str) -> bool:
    data = json.loads(_job_path(job_id).read_text())
    pid = int(data["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except OSError:
        return False
