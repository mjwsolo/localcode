#!/usr/bin/env bash
# Run codex as localcode's front end: start the bundled llama-server, point an
# isolated CODEX_HOME at it, hand over the terminal. ~/.codex is never touched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-Qwen3.8-27B-UD-Q4_K_XL}"; shift 2>/dev/null || true
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
CODEX_BIN="${LOCALCODE_CODEX_BIN:-codex}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
mkdir -p "$HERE/.run"
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" > "$HERE/.run/server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done

CH="$HERE/.run/codex-home"; mkdir -p "$CH"
sed "s|http://127.0.0.1:8123|http://127.0.0.1:$PORT|; s|^model = .*|model = \"$MODEL\"|" \
  "$HERE/config.toml" > "$CH/config.toml"
CODEX_HOME="$CH" LOCALCODE_API_KEY=local exec "$CODEX_BIN" "$@"
