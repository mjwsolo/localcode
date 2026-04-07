from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from .config import RuntimeConfig
from .jobs import launch_background_job
from .runtime import GemRuntimeGateway


def runtime_command(config: RuntimeConfig) -> str | None:
    if config.provider == "ollama":
        if shutil.which("ollama") is None:
            return None
        return "ollama serve"
    if config.provider == "llama_cpp":
        # Use the full llama_server_command() which includes TurboQuant,
        # flash attention, mmap, and all speed optimization flags.
        gateway = GemRuntimeGateway(config)
        model_path = config.model
        if not model_path:
            return None
        # Resolve Ollama blob path if needed
        if not model_path.startswith("/") and "sha256" not in model_path:
            model_path = gateway._find_ollama_blob(model_path)
        port = int(_port_from_base_url(config.base_url))
        cmd = gateway.llama_server_command(model_path, port=port)
        # Add single-slot for memory efficiency on 16GB
        cmd.extend(["-np", "1"])
        return " ".join(shlex.quote(c) for c in cmd)
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
