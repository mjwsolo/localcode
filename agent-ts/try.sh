#!/usr/bin/env bash
# Run localcode against your local models, interactively.
#
#   ./try.sh                         # one model, fastest path
#   ./try.sh gemma-4-12b-it-UD-Q4_K_XL
#   ./try.sh Qwen3.8-27B-UD-Q4_K_XL  ~/some/project
#   ./try.sh --all                   # ALL your models: /model to switch, /llama to
#                                    #   load/unload and download new ones from HF
#
# Starts a single-model llama-server (direct, not router — measurably faster),
# waits for it, then hands you pi's TUI. Ctrl+C exits and stops the server.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ALL=0
if [ "${1:-}" = "--all" ]; then ALL=1; shift; fi
MODEL="${1:-Qwen3.8-27B-UD-Q4_K_XL}"
PROJECT="${2:-$PWD}"
PORT="${LOCALCODE_TRY_PORT:-}"
if [ -z "$PORT" ]; then
  for p in $(seq 8123 8199); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }
  done
fi
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../../localcode/src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"

[ -x "$HERE/dist/localcode-agent" ] || { echo "Not built yet. Run: npm install && ./scripts/build.sh"; exit 1; }
if [ "$ALL" = 1 ]; then
  echo "starting llama-server in ROUTER mode on :$PORT (all models in $MODELS_DIR)"
  "$SERVER" --models-dir "$MODELS_DIR" --no-models-autoload --jinja \
    --host 127.0.0.1 --port "$PORT" -ngl 999 -c 32768 > "$HERE/.run/try.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  echo "loading $MODEL ..."
  curl -sf -X POST "http://127.0.0.1:$PORT/models/load" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\"}" >/dev/null || echo "  (load failed; use /llama in the TUI)"
  echo "ready. Type /model to browse, switch or download models."
  echo
  cd "$PROJECT"
  LLAMA_BASE_URL="http://127.0.0.1:$PORT" \
    exec "$HERE/dist/localcode-agent" -a -e "$HERE/extensions/localcode.ts" -e "$HERE/extensions/localcode-brand.ts" -e "$HERE/extensions/localcode-safety.ts" -e "$HERE/extensions/localcode-web.ts" -e "$HERE/extensions/localcode-app.ts" -e "$HERE/extensions/localcode-redact.ts" -e "$HERE/extensions/localcode-nav.ts" \
    --provider localcode --model "$MODEL" $( ls "$MODELS_DIR"/*.gguf >/dev/null 2>&1 && echo --models "localcode/*" ) --thinking off
fi

[ -f "$GGUF" ] || { echo "No such model: $GGUF"; echo; echo "Available:"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v '^mmproj' | sed 's/\.gguf$//;s/^/  /'; exit 1; }

echo "starting llama-server on :$PORT with $MODEL ..."
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" > "$HERE/.run/try.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server failed to start; see $HERE/.run/try.log"; exit 1; }
echo "ready. launching localcode in $PROJECT"
echo

cd "$PROJECT"
LLAMA_BASE_URL="http://127.0.0.1:$PORT" \
  "$HERE/dist/localcode-agent" -a -e "$HERE/extensions/localcode.ts" -e "$HERE/extensions/localcode-brand.ts" -e "$HERE/extensions/localcode-safety.ts" -e "$HERE/extensions/localcode-web.ts" -e "$HERE/extensions/localcode-app.ts" -e "$HERE/extensions/localcode-redact.ts" -e "$HERE/extensions/localcode-nav.ts" \
  --provider localcode --model "$MODEL" $( ls "$MODELS_DIR"/*.gguf >/dev/null 2>&1 && echo --models "localcode/*" ) --thinking off
