#!/usr/bin/env bash
# Try the pi front end against your local models, interactively.
#
#   ./try.sh                         # default model, current directory
#   ./try.sh gemma-4-12b-it-UD-Q4_K_XL
#   ./try.sh Qwen3.8-27B-UD-Q4_K_XL  ~/some/project
#
# Starts a single-model llama-server (direct, not router — measurably faster),
# waits for it, then hands you pi's TUI. Ctrl+C exits and stops the server.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-Qwen3.8-27B-UD-Q4_K_XL}"
PROJECT="${2:-$PWD}"
PORT="${LOCALCODE_TRY_PORT:-8123}"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../../localcode/src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"

[ -x "$HERE/dist/localcode-agent" ] || { echo "Not built yet. Run: npm install && ./scripts/build.sh"; exit 1; }
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; echo; echo "Available:"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v '^mmproj' | sed 's/\.gguf$//;s/^/  /'; exit 1; }

echo "starting llama-server on :$PORT with $MODEL ..."
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" > "$HERE/.run/try.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server failed to start; see $HERE/.run/try.log"; exit 1; }
echo "ready. launching pi in $PROJECT"
echo

cd "$PROJECT"
LLAMA_BASE_URL="http://127.0.0.1:$PORT" \
  "$HERE/dist/localcode-agent" -a -e "$HERE/extensions/localcode-provider.ts" \
  --provider localcode --model "$MODEL" --thinking off
