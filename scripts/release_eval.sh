#!/usr/bin/env bash
# Release eval: the same short Harness-Bench subset with the same local model on
# the candidate and on the previous release from PyPI, each with a fresh HOME.
# Exits 1 when the candidate scores lower than the margin allows or is more than
# the allowed fraction slower, 2 when the runs cannot be compared or started.
#
#   scripts/release_eval.sh --model <gguf name or group key> [options]
#
#   --model NAME            required; an installed model, never downloaded
#   --previous VERSION      release to compare against (default: newest older final release on PyPI)
#   --previous-venv DIR     use a venv that already has the previous release (skips the PyPI install)
#   --candidate-venv DIR    test an installed candidate (e.g. an rc) instead of this checkout
#   --tasks a,b,...         Harness-Bench task ids (default: the fixed set of six)
#   --score-margin X        allowed drop in mean score, 0..1 (default 0.05)
#   --max-slowdown X        allowed extra wall time as a fraction (default 0.25)
#   --models-dir DIR        where the GGUFs are (default: LOCALCODE_MODEL_DIR or the standard directory)
#   --work DIR              keep logs and venvs here (default: a new temp directory)
#   --dry-run               print the plan and every command, run nothing
#
# Needs `inspect` (pip install -e evals) on PATH, the Harness-Bench data in
# evals/data (see evals/README.md) and an otherwise idle GPU.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HELPER="$ROOT/scripts/release_eval.py"
PY="${RELEASE_EVAL_PYTHON:-python3}"

MODEL="" PREVIOUS="" PREV_VENV="" CAND_VENV="" TASKS="" SCORE_MARGIN="0.05" MAX_SLOWDOWN="0.25" MODELS_DIR="" WORK="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --previous) PREVIOUS="$2"; shift 2 ;;
    --previous-venv) PREV_VENV="$2"; shift 2 ;;
    --candidate-venv) CAND_VENV="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --score-margin) SCORE_MARGIN="$2"; shift 2 ;;
    --max-slowdown) MAX_SLOWDOWN="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }
[ -n "$TASKS" ] || TASKS="$("$PY" "$HELPER" tasks)"
[ -n "$MODELS_DIR" ] || MODELS_DIR="${LOCALCODE_MODEL_DIR:-$HOME/.local/share/localcode/models}"

say() { printf '%s\n' "$*"; }
run() { printf '+ %s\n' "$*"; if [ "$DRY" = 0 ]; then "$@"; fi; }

# GPU contention ruins timing: another llama-server means no valid comparison.
if pgrep -x llama-server >/dev/null 2>&1; then
  if [ "$DRY" = 1 ]; then say "WARNING: a llama-server is running; a real run would refuse to start"
  else echo "refusing to start: a llama-server process is already running (it would distort timings). Stop it and rerun." >&2; exit 2; fi
fi

# Models are only ever read from an existing directory.
if [ ! -d "$MODELS_DIR" ]; then
  if [ "$DRY" = 1 ]; then say "WARNING: models directory $MODELS_DIR does not exist"
  else echo "models directory $MODELS_DIR does not exist; this script never downloads models" >&2; exit 2; fi
fi

# Plain text read: tomllib is Python 3.11+, and LocalCode supports 3.10.
CAND_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
if [ -n "$CAND_VENV" ]; then
  CAND_VERSION="$("$CAND_VENV/bin/python" -c 'from importlib.metadata import version; print(version("localcode"))' 2>/dev/null || echo "$CAND_VERSION")"
fi
if [ -z "$PREVIOUS" ] && [ -z "$PREV_VENV" ]; then
  if [ "$DRY" = 1 ]; then PREVIOUS="<newest final release older than $CAND_VERSION, looked up on PyPI>"
  else PREVIOUS="$("$PY" "$HELPER" previous --candidate "$CAND_VERSION")" || { echo "could not determine the previous release; pass --previous" >&2; exit 2; }; fi
fi

if [ -z "$WORK" ]; then WORK="$(mktemp -d "${TMPDIR:-/tmp}/localcode-release-eval.XXXXXX")"; fi
say "release eval"
say "  model        $MODEL"
say "  candidate    ${CAND_VENV:-this checkout} ($CAND_VERSION)"
say "  previous     ${PREV_VENV:-PyPI localcode==$PREVIOUS}"
say "  tasks        $TASKS"
say "  limits       score drop <= $SCORE_MARGIN, slowdown <= $MAX_SLOWDOWN"
say "  models       $MODELS_DIR (read only)"
say "  work         $WORK"

# A directory shaped like a checkout (src/localcode) for an installed package, so
# the eval agent drives the installed UI binary and Python modules.
shim() { # <venv> <dir>
  local site; site="$("$1/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$2/src"
  ln -sfn "$site/localcode" "$2/src/localcode"
}

# One run, in a fresh HOME. $1 name, $2 checkout-like dir, $3 interpreter.
one() {
  local name="$1" repo="$2" py="$3" home="$WORK/home-$1"
  mkdir -p "$home/.config" "$home/.local/share" "$home/.cache" "$home/.local/state" "$home/state"
  say ""; say "== $name =="
  run env HOME="$home" XDG_CONFIG_HOME="$home/.config" XDG_DATA_HOME="$home/.local/share" XDG_CACHE_HOME="$home/.cache" \
      XDG_STATE_HOME="$home/.local/state" LOCALCODE_HOME="$home/state" LOCALCODE_MODEL_DIR="$MODELS_DIR" \
      LOCALCODE_EVALS_REPO="$repo" LOCALCODE_PY="$py" LOCALCODE_EVALS_WORKROOT="$WORK/workspaces-$name" \
      inspect eval localcode_evals/harnessbench -T "ids=$TASKS" --model "localcode/$MODEL" --log-dir "$WORK/logs-$name"
  run "$PY" "$HELPER" summarize "$WORK/logs-$name" --out "$WORK/$name.json"
}

# Candidate first: any warm-up or thermal advantage then goes to the previous
# release, so a pass is conservative.
if [ -n "$CAND_VENV" ]; then
  if [ "$DRY" = 0 ]; then shim "$CAND_VENV" "$WORK/candidate-repo"; else say "+ link $WORK/candidate-repo/src/localcode to the candidate venv's installed package"; fi
  one candidate "$WORK/candidate-repo" "$CAND_VENV/bin/python"
else
  CAND_PY="${LOCALCODE_PY:-}"
  if [ -z "$CAND_PY" ]; then if [ -x "$ROOT/.venv/bin/python" ]; then CAND_PY="$ROOT/.venv/bin/python"; else CAND_PY="$(command -v "$PY")"; fi; fi
  one candidate "$ROOT" "$CAND_PY"
fi

if [ -z "$PREV_VENV" ]; then
  PREV_VENV="$WORK/previous-venv"
  run "$PY" -m venv "$PREV_VENV"
  run "$PREV_VENV/bin/python" -m pip install --quiet --disable-pip-version-check "localcode==$PREVIOUS"
fi
if [ "$DRY" = 0 ]; then shim "$PREV_VENV" "$WORK/previous-repo"; else say "+ link $WORK/previous-repo/src/localcode to the previous venv's installed package"; fi
one previous "$WORK/previous-repo" "$PREV_VENV/bin/python"

say ""; say "== comparison =="
if [ "$DRY" = 1 ]; then
  say "+ $PY $HELPER compare $WORK/previous.json $WORK/candidate.json --score-margin $SCORE_MARGIN --max-slowdown $MAX_SLOWDOWN"
  say "(dry run: nothing was executed)"; exit 0
fi
set +e
"$PY" "$HELPER" compare "$WORK/previous.json" "$WORK/candidate.json" --score-margin "$SCORE_MARGIN" --max-slowdown "$MAX_SLOWDOWN" \
  --previous-label "previous" --candidate-label "candidate" --json-out "$WORK/comparison.json" | tee "$WORK/comparison.txt"
code=${PIPESTATUS[0]}
say "logs and summaries: $WORK"
exit "$code"
