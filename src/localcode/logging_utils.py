from __future__ import annotations

import logging
from pathlib import Path
import sys

from .config import ensure_home_dirs


def configure_logging(debug: bool = False) -> Path:
    home = ensure_home_dirs()
    log_path = home / "logs" / "localcode.log"
    handlers: list[logging.Handler] = []
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    except OSError:
        # Logging must never prevent the TUI from starting. Sandboxed or
        # permission-damaged installs can fail to open ~/.localcode/logs; fall
        # back to stderr and keep the app usable.
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_path
