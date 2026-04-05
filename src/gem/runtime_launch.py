from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from .config import RuntimeConfig
from .jobs import launch_background_job


def runtime_command(config: RuntimeConfig) -> str | None:
    if config.provider == "ollama":
        if shutil.which("ollama") is None:
            return None
        return "ollama serve"
    if config.provider == "llama_cpp":
        server = shutil.which("llama-server") or shutil.which("llama_cpp.server")
        if server is None or not config.model:
            return None
        base = [
            shlex.quote(server),
            "-m",
            shlex.quote(config.model),
            "--port",
            _port_from_base_url(config.base_url),
        ]
        if config.llama_cpp_gpu_layers:
            base.extend(["-ngl", str(config.llama_cpp_gpu_layers)])
        if config.llama_cpp_threads:
            base.extend(["-t", str(config.llama_cpp_threads)])
        if config.llama_cpp_batch_size:
            base.extend(["-b", str(config.llama_cpp_batch_size)])
        return " ".join(base)
    return None


def launch_runtime(config: RuntimeConfig, cwd: Path) -> tuple[bool, str]:
    command = runtime_command(config)
    if not command:
        return False, "No launch command available for the selected provider."
    job = launch_background_job(command, cwd)
    return True, job.job_id


def _port_from_base_url(base_url: str) -> str:
    if ":" not in base_url.rsplit("/", 1)[-1]:
        return "8080"
    return base_url.rsplit(":", 1)[-1]
