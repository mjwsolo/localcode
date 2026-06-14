"""Make ``import optimize`` resolve when tests run from anywhere.

The repo root (parent of ``optimize/``) is put on sys.path so the package
imports cleanly without an install. No model/server imports happen here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
