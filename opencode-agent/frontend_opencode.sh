#!/usr/bin/env bash
# Run opencode as localcode's front end: bundled llama-server + project-local
# opencode.json. Global ~/.config/opencode is never touched.
set -euo pipefail
export PATH="$HOME/.hermes/node/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-Qwen3.8-27B-UD-Q4_K_XL}"; shift 2>/dev/null || true
PROJECT="${1:-$PWD}"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
mkdir -p "$HERE/.run"
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" > "$HERE/.run/server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done

cd "$PROJECT"
sed "s|8123|$PORT|" "$HERE/opencode.json" > ./opencode.json
exec opencode -m "localcode/$MODEL"
