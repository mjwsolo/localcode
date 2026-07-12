"""Explicit lifecycle tool for supervised background commands."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .base import ToolContext
from .._subproc_env import clean_env
from ..process_registry import ProcessRecord, find_record, load_records, record_process, refresh_record, save_records

SCHEMA = {
    "type": "function",
    "function": {
        "name": "background_process",
        "description": "Start, poll, list, or stop a supervised background command. Returns a stable process ID and incremental output.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "poll", "list", "stop"]},
                "command": {"type": "string", "description": "Required for start."},
                "process_id": {"type": "string", "description": "Required for poll/stop."},
                "offset": {"type": "integer", "description": "Byte offset for incremental poll output.", "default": 0},
                "owner": {"type": "string", "description": "Logical owner label.", "default": "agent"},
            },
            "required": ["action"],
        },
    },
}

def _status(record: ProcessRecord) -> str:
    return "stopped" if record.stopped_at else "running"

def execute(ctx: ToolContext, args: dict) -> str:
    action = str(args.get("action", ""))
    if action == "list":
        records = load_records(ctx.repo)
        if not records:
            return "No supervised background processes."
        return "\n".join(f"{r.process_id} {_status(refresh_record(ctx.repo, r.process_id) or r)} owner={r.owner} pid={r.pid} command={r.command}" for r in records[-50:])
    process_id = str(args.get("process_id", ""))
    if action in {"poll", "stop"} and not process_id:
        return f"Error: process_id is required for {action}."
    if action == "start":
        command = str(args.get("command", "")).strip()
        if not command:
            return "Error: command is required for start."
        log_dir = ctx.repo / ".localcode" / "process-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        provisional = ProcessRecord(pid=0, pgid=0, port=0, url="", cwd=str(ctx.repo), kind="background", command=command, log_path="", verified=False, started_at=time.time(), owner=str(args.get("owner", "agent")))
        log_path = log_dir / f"{provisional.process_id}.log"
        stream = log_path.open("ab", buffering=0)
        try:
            proc = subprocess.Popen(command, cwd=ctx.repo, shell=True, executable="/bin/sh", stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=clean_env(), start_new_session=True)
        finally:
            stream.close()
        provisional.pid = proc.pid
        provisional.pgid = proc.pid
        provisional.log_path = str(log_path)
        record_process(ctx.repo, provisional)
        return f"Started {provisional.process_id}: status=running pid={proc.pid} owner={provisional.owner} output_offset=0"
    record = refresh_record(ctx.repo, process_id)
    if record is None:
        return f"Error: unknown process_id: {process_id}"
    if action == "stop":
        if not record.stopped_at:
            try:
                os.killpg(record.pgid or record.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                return f"Error: could not stop {process_id}: {exc}"
            records = load_records(ctx.repo)
            for item in records:
                if item.process_id == process_id:
                    item.stopped_at = time.time()
            save_records(ctx.repo, records)
        return f"Stopped {process_id}."
    if action == "poll":
        offset = max(0, int(args.get("offset", 0)))
        data = b""
        try:
            with Path(record.log_path).open("rb") as handle:
                handle.seek(offset)
                data = handle.read(64_000)
                next_offset = handle.tell()
        except OSError:
            next_offset = offset
        output = data.decode(errors="replace")
        return f"id={process_id} status={_status(record)} pid={record.pid} next_offset={next_offset}\n{output}".rstrip()
    return f"Error: unsupported action: {action}"

def is_concurrency_safe(args: dict) -> bool:
    return str(args.get("action")) in {"poll", "list"}
