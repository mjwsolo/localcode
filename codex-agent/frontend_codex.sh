#!/usr/bin/env bash
# Run codex as localcode's front end: start the bundled llama-server, point an
# isolated CODEX_HOME at it, hand over the terminal. ~/.codex is never touched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
MODEL="${1:-}"; shift 2>/dev/null || true
if [ -z "$MODEL" ]; then
# Two-level picker, the localcode way: model first, then quant.
# Reads MODELS_DIR; sets MODEL. Pure bash, no deps.

  PY_BIN="${LOCALCODE_PY:-$HERE/../localcodevenv/bin/python}"
  [ -x "$PY_BIN" ] || PY_BIN=python3
  HERE_PICKER="$(cd "$(dirname "$0")" && pwd)/model_picker_cli.py"
  MODEL="$($PY_BIN "$HERE_PICKER")" || exit $?
  [ -n "$MODEL" ] || { echo "no model chosen"; exit 1; }
fi
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
CODEX_BIN="${LOCALCODE_CODEX_BIN:-codex}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""; CTRL=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
for p in $(seq 8323 8399); do curl -sf "http://127.0.0.1:$p/status" >/dev/null 2>&1 || { CTRL=$p; break; }; done
mkdir -p "$HERE/.run"
PY_BIN="${LOCALCODE_PY:-$HERE/../localcodevenv/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3
# The supervisor owns llama-server and serves the in-TUI /model picker
# (catalog, quants, download, switch) on a localhost control port. The
# inference port never changes across a switch.
"$PY_BIN" "$HERE/localcode_supervisor.py" --model "$MODEL" --port "$PORT" --control-port "$CTRL" \
  --server "$SERVER" --models-dir "$MODELS_DIR" > "$HERE/.run/supervisor.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SRV 2>/dev/null || { echo "model failed to load (see $HERE/.run/server.log)"; exit 1; }; sleep 1; done

CH="$HERE/.run/codex-home"; mkdir -p "$CH"
# Context window comes from the supervisor, which computed it for THIS machine
# (RAM tier + KV compression) — never a hardcoded number.
CTX=$(curl -s "http://127.0.0.1:$CTRL/status" | sed -n 's/.*"ctx": \([0-9]*\).*/\1/p'); CTX="${CTX:-32768}"
sed "s|http://127.0.0.1:8123|http://127.0.0.1:$PORT|; s|^model = .*|model = \"$MODEL\"|; s|^model_context_window = .*|model_context_window = $CTX|" \
  "$HERE/config.toml" > "$CH/config.toml"
CODEX_HOME="$CH" LOCALCODE_API_KEY=local LOCALCODE_CONTROL_URL="http://127.0.0.1:$CTRL" exec "$CODEX_BIN" "$@"
