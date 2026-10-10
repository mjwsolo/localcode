#!/usr/bin/env bash
# Nightly evals: Checks, oracle verification, regression Tasks per model, gate vs the
# previous run of the same model. Usage: nightly.sh <checkout> [model ...]
# Refuses to run while an interactive localcode session holds the supervisor lock.
set -u
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
  python3 run_tasks.py --repo "$REPO" --model "$m" --set regression --trials 3 --label "$LABEL" | tail -25
  NEW=$(ls -td runs/*-"$LABEL"-"$m" | head -1); PREV=$(ls -td runs/*-nightly-*-"$m" | grep -v "$NEW" | head -1)
  if [ -n "$PREV" ]; then echo "== gate vs $PREV"; python3 gate.py "$PREV/results.jsonl" "$NEW/results.jsonl" || rc=1; fi
done
exit $rc
