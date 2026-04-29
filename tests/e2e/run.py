"""Entry point — `python tests/e2e/run.py`.

Spawns llama-server (via the same path normal launch uses), runs all
scenarios in `tests/e2e/scenarios/`, writes a structured report to
`~/.localcode/test-results/<timestamp>/`.

Pre-requisites:
  • The default model is downloaded (`localcode setup` already run).
  • You're not running another LocalCode instance that holds the port.

After it finishes:
  • Skim `~/.localcode/test-results/latest/report.md` for the summary.
  • If anything failed, `~/.localcode/test-results/latest/failures.md`
    contains the failed cases + event traces — the file to paste at
    terminal coding tools when asking "fix what failed."
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python tests/e2e/run.py` from repo root by adding
# both the project root and src/ to sys.path before importing anything.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))


if __name__ == "__main__":
    from tests.e2e.runner import main
    sys.exit(main())
