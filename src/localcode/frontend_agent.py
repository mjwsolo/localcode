"""The agent-plane front end: start the model server, hand over to the agent.

Opt-in and additive. `localcode` behaves exactly as it always has unless
LOCALCODE_FRONTEND=agent is set, in which case entrypoint dispatches here.

Division of labour:
  * Python (here)  — find the binaries, start llama-server in router mode,
                     wait for health, hand over the terminal.
  * agent binary   — everything the user sees: model picker, downloads,
                     approvals, the session itself.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import paths


def _free_port(start: int = 8123, end: int = 8199) -> int:
    for port in range(start, end):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port for the model server")


def _agent_binary() -> Path | None:
    """The bun-compiled agent: env override, next to the wheel, then a dev tree."""
    override = os.environ.get("LOCALCODE_AGENT_BIN")
    if override and Path(override).is_file():
        return Path(override)
    packaged = Path(__file__).parent / "bin" / "agent" / "localcode-agent"
    if packaged.is_file():
        return packaged
    dev = Path(__file__).resolve().parents[3] / "localcode-pi" / "agent-ts" / "dist" / "localcode-agent"
    return dev if dev.is_file() else None


def _llama_server() -> Path | None:
    packaged = Path(__file__).parent / "bin" / "llama-server"
    if packaged.is_file():
        return packaged
    found = shutil.which("llama-server")
    return Path(found) if found else None


def _extensions(agent: Path) -> list[str]:
    ext_dir = agent.parent / "extensions"
    if not ext_dir.is_dir():
        ext_dir = agent.parent.parent / "extensions"
    if not ext_dir.is_dir():
        return []
    names = ["localcode.ts", "localcode-brand.ts", "localcode-safety.ts", "localcode-web.ts", "localcode-app.ts", "localcode-redact.ts", "localcode-nav.ts"]
    return [str(ext_dir / n) for n in names if (ext_dir / n).is_file()]


def run(argv: list[str] | None = None) -> int:
    agent = _agent_binary()
    if agent is None:
        print("localcode: agent binary not found. Set LOCALCODE_AGENT_BIN, or build it "
              "with agent-ts/scripts/build.sh", file=sys.stderr)
        return 2
    server = _llama_server()
    if server is None:
        print("localcode: llama-server not found in the package.", file=sys.stderr)
        return 2

    models_dir = Path(os.environ.get("LOCALCODE_MODELS_DIR", Path.home() / ".local/share/localcode/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    log = paths.global_state_dir() / "agent-server.log"

    # Router mode: the agent can list, load, unload and download models itself,
    # which is what makes the in-app model picker work on a fresh machine.
    proc = subprocess.Popen(
        [str(server), "--models-dir", str(models_dir), "--no-models-autoload", "--jinja",
         "--host", "127.0.0.1", "--port", str(port), "-ngl", "999", "-c", "32768"],
        stdout=log.open("w"), stderr=subprocess.STDOUT,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        import httpx
        for _ in range(240):
            try:
                if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print(f"localcode: model server did not start; see {log}", file=sys.stderr)
            return 1

        env = dict(os.environ, LLAMA_BASE_URL=base, LOCALCODE_MODELS_DIR=str(models_dir))
        cmd = [str(agent), "-a", "--thinking", "off"]
        # Scope model lists to our provider — but not on a true first run,
        # where zero models exist and the scope only produces a warning.
        if any(models_dir.glob("*.gguf")):
            cmd += ["--models", "localcode/*"]
        for ext in _extensions(agent):
            cmd += ["-e", ext]
        cmd += list(argv or [])
        return subprocess.call(cmd, env=env)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
