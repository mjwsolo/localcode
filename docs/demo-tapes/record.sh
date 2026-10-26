#!/usr/bin/env bash
# Record the tapes against the Git Wrapped demo repo and install the GIFs.
# Needs: vhs, ffmpeg, python3, and localcode installed with `uv tool install localcode`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="Qwen3.6-35B-A3B-UD-Q8_K_XL"
command -v vhs >/dev/null || { echo "brew install vhs"; exit 1; }
command -v ffmpeg >/dev/null || { echo "brew install ffmpeg"; exit 1; }
command -v localcode >/dev/null || { echo "uv tool install localcode"; exit 1; }
PROJ="/private/tmp/git-wrapped-demo"
export LC_DEMO_PROJECT="$PROJ"
export LOCALCODE_MODEL_DIR="/private/tmp/localcode-models"
mkdir -p "$LOCALCODE_MODEL_DIR"
for model in "${LOCALCODE_MODEL_SOURCE:-$HOME/.local/share/localcode/models}"/*.gguf; do
  [[ -e "$model" ]] || continue
  [[ -e "$LOCALCODE_MODEL_DIR/${model##*/}" ]] || ln -s "$model" "$LOCALCODE_MODEL_DIR/${model##*/}"
done
[[ -e "$LOCALCODE_MODEL_DIR/$MODEL.gguf" ]] || { echo "Download $MODEL first (run localcode, /models)."; exit 1; }
MODEL="$MODEL" python3 - <<'PY'
import json
import os
import urllib.request

from localcode.ui.ports import find_running

running = find_running()
if running:
    control_port, status = running
    with urllib.request.urlopen(f"http://127.0.0.1:{control_port}/models_dir") as response:
        models_dir = json.load(response)["path"]
    if models_dir != os.environ["LOCALCODE_MODEL_DIR"]:
        raise SystemExit("The running model server uses a personal model folder. Stop it or point it at /private/tmp/localcode-models before recording.")
    if status.get("current") != os.environ["MODEL"]:
        raise SystemExit(f"Load {os.environ['MODEL']} before recording so the demo matches the tapes.")
PY
REC_DIR="$(mktemp -d /private/tmp/localcode-recording.XXXXXX)"
cp "$HERE"/*.tape "$REC_DIR/"
cd "$REC_DIR"
mkdir -p out
TAPES=("${@:-step-2-choose-model step-3-ask first-change step-4-verify}")
for t in ${TAPES[@]}; do
  # Fresh 646-commit repo for every take, so earlier output never leaks in.
  python3 "$HERE/make_demo_repo.py" "$PROJ"
  vhs "$t.tape"
  test -s "$REC_DIR/out/$t.png/frame-text-00001.png"
  if [[ "$t" == "step-4-verify" || "$t" == "first-change" ]]; then
    # The run must have produced the data and the page.
    (cd "$PROJ" && test -s wrapped.py && test -s data.js && test -s wrapped.html && grep -q "WRAPPED" data.js)
  fi
  if [[ "$t" == "first-change" ]]; then
    cp "$(ls "$REC_DIR/out/$t.png"/frame-text-*.png | tail -1)" "$REC_DIR/out/first-change-final.png"
  fi
  ffmpeg -hide_banner -loglevel error -y -framerate 10 -i "$REC_DIR/out/$t.png/frame-text-%05d.png" -filter_complex 'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none' -loop 0 "$REC_DIR/out/$t.gif"
  test -s "$REC_DIR/out/$t.gif"
  if command -v gifsicle >/dev/null; then gifsicle -O3 -o "$REC_DIR/out/$t.opt.gif" "$REC_DIR/out/$t.gif" && mv "$REC_DIR/out/$t.opt.gif" "$REC_DIR/out/$t.gif"; fi
  ls -la "$REC_DIR/out/$t.gif"
done
cp "$REC_DIR"/out/*.gif "$ROOT/docs/public/demo/"
[ -f "$REC_DIR/out/first-change-final.png" ] && cp "$REC_DIR/out/first-change-final.png" "$ROOT/docs/public/demo/"
[ -f "$REC_DIR/out/first-change.gif" ] && cp "$REC_DIR/out/first-change.gif" "$ROOT/docs/assets/demo/first-change.gif"
echo "installed into docs/public/demo and docs/assets/demo"
