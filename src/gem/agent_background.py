from __future__ import annotations

from pathlib import Path
import shlex
import sys

from .jobs import launch_background_job


def launch_background_agent(prompt: str, cwd: Path, profile_name: str | None = None, model_name: str | None = None) -> str:
    parts = [shlex.quote(sys.executable), "-m", "gem.cli", "agent-runner", shlex.quote(prompt)]
    if profile_name:
        parts.insert(3, shlex.quote(profile_name))
        parts.insert(3, "--profile")
    if model_name:
        insert_at = 3 if not profile_name else 5
        parts.insert(insert_at, "--model")
        parts.insert(insert_at + 1, shlex.quote(model_name))
    command = " ".join(parts)
    job = launch_background_job(command, cwd)
    return job.job_id
