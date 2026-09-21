"""localcode's default interface: the localcode agent runtime (a fork of
opencode, branded and stripped of cloud features) driven by a small Python
supervisor that owns the bundled llama-server and serves the in-app model
picker on a localhost control port.

Layout:
  launch.py      what `localcode` runs by default (see entrypoint.py)
  supervisor.py  llama-server owner + control API (/catalog /quants /select ...)
  server_cmd.py  the llama-server command line for THIS machine (RAM tier, KV, thinking)
  ports.py       singleton guard + free port pair
  plugin/        completion discipline loaded into the runtime as a plugin
  FORK_COMMIT    the exact source commit the bundled binary was built from

State lives under ~/.local/share/localcode-agent/run (logs, voice runtime),
never inside the user's project.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

UI_BINARY_NAME = "localcode-ui"


def run_dir() -> Path:
    """Per-user scratch dir for the supervisor and launcher (logs, voice venv)."""
    base = os.environ.get("LOCALCODE_AGENT_RUN_DIR")
    path = Path(base) if base else Path.home() / ".local" / "share" / "localcode-agent" / "run"
    return path


def fork_commit() -> str:
    try:
        return (Path(__file__).parent / "FORK_COMMIT").read_text().strip()
    except OSError:
        return "unknown"


def ui_binary_path() -> Path | None:
    """The bundled UI binary, or None when this install has none.

    LOCALCODE_UI_BIN overrides for development against a fresh fork build.
    """
    override = os.environ.get("LOCALCODE_UI_BIN")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    p = Path(__file__).resolve().parent.parent / "bin" / UI_BINARY_NAME
    return p if p.is_file() else None


def platform_supported() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() == "arm64"


def plugin_path() -> Path:
    return Path(__file__).resolve().parent / "plugin" / "localcode.ts"
