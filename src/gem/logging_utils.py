from __future__ import annotations

import logging
from pathlib import Path

from .config import ensure_home_dirs


def configure_logging(debug: bool = False) -> Path:
    home = ensure_home_dirs()
    log_path = home / "logs" / "gem.log"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
        ],
        force=True,
    )
    return log_path
