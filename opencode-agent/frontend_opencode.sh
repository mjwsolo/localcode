#!/usr/bin/env bash
# Run opencode as localcode's front end: bundled llama-server + project-local
# opencode.json. Global ~/.config/opencode is never touched.
set -euo pipefail
export PATH="$HOME/.hermes/node/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
MODEL="${1:-}"; shift 2>/dev/null || true
PROJECT="${1:-$PWD}"
if [ -z "" ]; then
# Two-level picker, the localcode way: model first, then quant.
# Reads MODELS_DIR; sets MODEL. Pure bash, no deps.

  PY_BIN="${LOCALCODE_PY:-$HERE/../localcodevenv/bin/python}"
  [ -x "$PY_BIN" ] || PY_BIN=python3
  HERE_PICKER="$(cd "$(dirname "$0")" && pwd)/model_picker_cli.py"
  MODEL="$($PY_BIN "$HERE_PICKER")" || exit $?
  [ -n "$MODEL" ] || { echo "no model chosen"; exit 1; }
fi
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
mkdir -p "$HERE/.run"
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" --chat-template-kwargs '{"enable_thinking":false}' > "$HERE/.run/server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done

cd "$PROJECT"
sed "s|8123|$PORT|" "$HERE/opencode.json" > ./opencode.json
exec opencode -m "localcode/$MODEL"
