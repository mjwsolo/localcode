"""Where the Harness-Bench 3.0 data must be, and how to get it. Nothing here downloads.

    python -m localcode_evals.data

The data lives in `evals/data/` (gitignored) or `LOCALCODE_EVALS_DATA`. A symlink
to a copy you already have is fine.
"""
from __future__ import annotations

from .benchmarks import DATA

PATH = DATA / "harness-bench"
SIZE = "about 90 MB"
HOW = f"git clone --depth 1 https://github.com/Qihoo360/harness-bench {PATH}"

if __name__ == "__main__":
    print(f"benchmark data directory: {DATA}\n")
    current = (PATH / "src/harnessbench/grading/outcome_cases.py").is_file()
    state = "present" if current else ("OLD (before 3.0)" if (PATH / "tasks").is_dir() else "MISSING")
    print(f"harnessbench  {state}  {PATH}")
    if not current:
        print(f"              {SIZE}\n              {HOW}")
