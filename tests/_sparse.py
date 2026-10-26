"""Create sparse files of a given size portably (NTFS needs the sparse flag)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def make_sparse(path, size: int) -> None:
    path = Path(path)
    if not path.exists():
        path.write_bytes(b"")
    if sys.platform == "win32":
        subprocess.run(["fsutil", "sparse", "setflag", str(path)],
                       check=False, capture_output=True)
    os.truncate(path, size)
