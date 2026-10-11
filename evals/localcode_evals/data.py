"""Where the HarnessBench data must be, and how to get it. Nothing here downloads.

    python -m localcode_evals.data

The data lives in `evals/data/` (gitignored) or `LOCALCODE_EVALS_DATA`. A symlink
to a copy you already have is fine.
"""
from __future__ import annotations

from .benchmarks import DATA

PATH = DATA / "harness-bench"
SIZE = "about 35 MB"
HOW = f"git clone --depth 1 https://github.com/Qihoo360/harness-bench {PATH}"

if __name__ == "__main__":
    print(f"benchmark data directory: {DATA}\n")
    print(f"harnessbench  {'present' if (PATH / 'tasks').is_dir() else 'MISSING'}  {PATH}")
    if not (PATH / "tasks").is_dir():
        print(f"              {SIZE}\n              {HOW}")
