#!/usr/bin/env bash
# Nightly evals: Checks, oracle verification, the Inspect regression task per model, gate
# vs the previous run of the same model. Usage: nightly.sh <checkout> [model ...]
# Refuses to run while an interactive localcode session holds the supervisor lock.
set -uo pipefail
REPO="${1:?checkout with .venv and src/localcode/bin}"; shift
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen3.8-27B-UD-Q4_K_XL gemma-4-12b-it-UD-Q4_K_XL)
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
# Inspect AI and this suite (pip install -e evals) live in their own environment.
INSPECT="${INSPECT:-inspect}"; PY="${INSPECT_PY:-python3}"
LABEL="nightly-$(git -C "$REPO" rev-parse --short HEAD)"
if lsof -nP -iTCP:8323 -sTCP:LISTEN >/dev/null 2>&1; then echo "a localcode session is running (control port 8323); skipping tonight"; exit 0; fi
echo "== checks"; LOCALCODE_CHECK_REPO="$REPO" "$REPO/.venv/bin/python" -m pytest checks -q -p no:cacheprovider || { echo "CHECKS RED"; exit 1; }
echo "== oracles"; python3 verify_oracles.py >/dev/null || { echo "ORACLES RED"; exit 1; }
rc=0
for m in "${MODELS[@]}"; do
  ls "$HOME/.local/share/localcode/models/$m.gguf" >/dev/null 2>&1 || { echo "== $m not downloaded, skipped"; continue; }
  echo "== tasks: $m"
  # Pick the prior completed run before starting; never infer the current run
  # from a glob after a failed launch.
  PREV=$(ls -1dt "runs/nightly/$m"/*/ 2>/dev/null | while read -r d; do [ -f "$d/.complete" ] && { echo "${d%/}"; break; }; done)
  NEW="runs/nightly/$m/$(date +%Y%m%d-%H%M%S)-$LABEL"
  if ! LOCALCODE_EVALS_REPO="$REPO" LOCALCODE_PY="$REPO/.venv/bin/python" "$INSPECT" eval localcode_evals/regression \
        --model "localcode/$m" --epochs 3 --log-dir "$NEW" --tags "$LABEL" --display plain --no-fail-on-error | tail -25; then
    echo "TASK RUN FAILED: $m"; rc=1; continue
  fi
  # Self-comparison validates coverage, duplicate trials and nonempty results
  # even on the first nightly run, when there is no prior baseline.
  if ! "$PY" gate.py "$NEW" "$NEW"; then rc=1; continue; fi
  touch "$NEW/.complete"
  if [ -n "$PREV" ]; then echo "== gate vs $PREV"; "$PY" gate.py "$PREV" "$NEW" || rc=1; fi
done
exit $rc
