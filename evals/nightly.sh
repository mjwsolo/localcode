#!/usr/bin/env bash
# Nightly evals: Checks, oracle verification, regression Tasks per model, gate vs the
# previous run of the same model. Usage: nightly.sh <checkout> [model ...]
# Refuses to run while an interactive localcode session holds the supervisor lock.
set -uo pipefail
REPO="${1:?checkout with .venv and src/localcode/bin}"; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen3.8-27B-UD-Q4_K_XL gemma-4-12b-it-UD-Q4_K_XL)
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
LABEL="nightly-$(git -C "$REPO" rev-parse --short HEAD)"
if lsof -nP -iTCP:8323 -sTCP:LISTEN >/dev/null 2>&1; then echo "a localcode session is running (control port 8323); skipping tonight"; exit 0; fi
echo "== checks"; LOCALCODE_CHECK_REPO="$REPO" "$REPO/.venv/bin/python" -m pytest checks -q -p no:cacheprovider || { echo "CHECKS RED"; exit 1; }
echo "== oracles"; python3 verify_oracles.py >/dev/null || { echo "ORACLES RED"; exit 1; }
rc=0
for m in "${MODELS[@]}"; do
  ls "$HOME/.local/share/localcode/models/$m.gguf" >/dev/null 2>&1 || { echo "== $m not downloaded, skipped"; continue; }
  echo "== tasks: $m"
  # Pick the prior completed run before starting; never infer the current run
  # from a glob after a failed launch. Nanoseconds avoid same-minute collisions.
  PREV=$(python3 - "$m" <<'PYBASE'
import sys
from pathlib import Path
runs = [p for p in Path("runs").glob("*-nightly-*-" + sys.argv[1]) if (p / ".complete").is_file()]
print(max(runs, key=lambda p: p.stat().st_mtime) if runs else "")
PYBASE
  ) || { rc=1; continue; }
  NEW="runs/$(python3 -c 'import time; print(time.time_ns())')-$LABEL-$m"
  if ! python3 run_tasks.py --repo "$REPO" --model "$m" --set regression --trials 3 --label "$LABEL" --out "$NEW" | tail -25; then
    echo "TASK RUN FAILED: $m"; rc=1; continue
  fi
  # Self-comparison validates coverage, duplicate trials and nonempty results
  # even on the first nightly run, when there is no prior baseline.
  if ! python3 gate.py "$NEW/results.jsonl" "$NEW/results.jsonl"; then rc=1; continue; fi
  touch "$NEW/.complete"
  if [ -n "$PREV" ]; then echo "== gate vs $PREV"; python3 gate.py "$PREV/results.jsonl" "$NEW/results.jsonl" || rc=1; fi
done
exit $rc
