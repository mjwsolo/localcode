#!/usr/bin/env bash
# Record every tape in a scratch project and install the GIFs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
command -v vhs >/dev/null || { echo "brew install vhs"; exit 1; }
PROJ="/private/tmp/retry-demo"; mkdir -p "$PROJ"
printf 'def retry(times):\n    pass\n' > "$PROJ/retry.py"
cat > "$PROJ/test_retry.py" <<'PY'
from retry import retry
calls = []
@retry(times=3)
def flaky():
    calls.append(1)
    if len(calls) < 3: raise ValueError("boom")
    return "ok"
def test_retries_until_success():
    assert flaky() == "ok"
    assert len(calls) == 3
PY
export PATH="${LOCALCODE_BIN_DIR:-$ROOT/.venv/bin}:$PATH"
export LC_DEMO_PROJECT="$PROJ"
export LOCALCODE_DISABLE_TOOLS="todo_write"
export LOCALCODE_MODEL_DIR="/private/tmp/localcode-models"
mkdir -p "$LOCALCODE_MODEL_DIR"
for model in "${LOCALCODE_MODEL_SOURCE:-$HOME/.local/share/localcode/models}"/*.gguf; do
  [[ -e "$model" ]] || continue
  [[ -e "$LOCALCODE_MODEL_DIR/${model##*/}" ]] || ln -s "$model" "$LOCALCODE_MODEL_DIR/${model##*/}"
done
python3 - <<'PY'
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
    if status.get("current") != "Qwen3.6-35B-A3B-UD-Q8_K_XL":
        raise SystemExit("Load Qwen3.6-35B-A3B-UD-Q8_K_XL before recording so the demo matches the tapes.")
PY
REC_DIR="$(mktemp -d /private/tmp/localcode-recording.XXXXXX)"
cp "$HERE"/*.tape "$REC_DIR/"
cd "$REC_DIR"
mkdir -p out
TAPES=("${@:-step-2-choose-model step-3-ask first-change}")
for t in ${TAPES[@]}; do
  rm -rf "$PROJ"/.localcode-agent "$PROJ"/__pycache__ "$PROJ"/.pytest_cache; printf 'def retry(times):\n    pass\n' > "$PROJ/retry.py"
  vhs "$t.tape"
  test -s "$REC_DIR/out/$t.png/frame-text-00001.png"
  if [[ "$t" == "step-4-verify" || "$t" == "first-change" ]]; then
    (cd "$PROJ" && python3 -m pytest -q)
  fi
  if [[ "$t" == "first-change" ]]; then
    cp "$(ls "$REC_DIR/out/$t.png"/frame-text-*.png | tail -1)" "$REC_DIR/out/first-change-final.png"
    test -s "$REC_DIR/out/$t.png/frame-text-00200.png"
    frame_limit=(-frames:v 200)
  else
    frame_limit=()
  fi
  ffmpeg -hide_banner -loglevel error -y -framerate 10 -i "$REC_DIR/out/$t.png/frame-text-%05d.png" "${frame_limit[@]}" -filter_complex 'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none' -loop 0 "$REC_DIR/out/$t.gif"
  test -s "$REC_DIR/out/$t.gif"
  if command -v gifsicle >/dev/null; then gifsicle -O3 -o "$REC_DIR/out/$t.opt.gif" "$REC_DIR/out/$t.gif" && mv "$REC_DIR/out/$t.opt.gif" "$REC_DIR/out/$t.gif"; fi
  ls -la "$REC_DIR/out/$t.gif"
done
if [[ -d "$REC_DIR/out/first-change.png" ]]; then
  # Use the verified full turn for the shorter edit-and-test clip as well.
  # Recheck this frame window if the model's timing changes on a later take.
  test -s "$REC_DIR/out/first-change.png/frame-text-00185.png"
  ffmpeg -hide_banner -loglevel error -y -framerate 10 -start_number 70 -i "$REC_DIR/out/first-change.png/frame-text-%05d.png" -frames:v 116 -filter_complex 'split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none' -loop 0 "$REC_DIR/out/step-4-verify.gif"
  if command -v gifsicle >/dev/null; then gifsicle -O3 -o "$REC_DIR/out/step-4-verify.opt.gif" "$REC_DIR/out/step-4-verify.gif" && mv "$REC_DIR/out/step-4-verify.opt.gif" "$REC_DIR/out/step-4-verify.gif"; fi
fi
cp "$REC_DIR"/out/*.gif "$ROOT/website/public/demo/"
[ -f "$REC_DIR/out/first-change-final.png" ] && cp "$REC_DIR/out/first-change-final.png" "$ROOT/website/public/demo/"
[ -f "$REC_DIR/out/first-change.gif" ] && cp "$REC_DIR/out/first-change.gif" "$ROOT/docs/assets/demo/first-change.gif"
echo "installed into website/public/demo and docs/assets/demo"
