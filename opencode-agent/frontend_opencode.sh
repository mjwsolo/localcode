#!/usr/bin/env bash
# Run the localcode-branded opencode fork as localcode's front end.
#
#   pick model → pick quant (or pass an alias)  →  supervisor starts the bundled
#   llama-server on a free port and serves the in-TUI /models picker on a
#   localhost control port  →  project-local localcode.json points the fork at
#   that server  →  run the fork binary.
#
# Nothing global is read or written: the fork's XDG dirs are ~/.*/localcode-agent,
# the config is written next to the project, and no network is used except
# model downloads the user asks for.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
BIN="${LOCALCODE_OPENCODE_BIN:-$HERE/.run/localcode-opencode}"
[ -x "$BIN" ] || { echo "fork binary not found at $BIN — build it: see README.md (build section)"; exit 1; }
PY_BIN="${LOCALCODE_PY:-$HERE/../localcodevenv/bin/python}"; [ -x "$PY_BIN" ] || PY_BIN=python3

MODEL="${1:-}"; shift 2>/dev/null || true
PROJECT="${1:-$PWD}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
# One download journey: with no model argument the TUI opens first and its
# /models picker (model → quant) downloads and loads the model in-app.
if [ -n "$MODEL" ]; then
  GGUF="$MODELS_DIR/$MODEL.gguf"
  [ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }
fi

# A picker reserves its future model port even before llama-server is running.
# Probe socket availability as well: an unhealthy service still owns its port.
read -r PORT CTRL <<< "$("$PY_BIN" "$HERE/ports.py")"
[ -n "$PORT" ] && [ -n "$CTRL" ] || { echo "No free local ports"; exit 1; }
mkdir -p "$HERE/.run"
# The supervisor owns llama-server (per-machine flags from server_cmd.py: RAM-tier
# context, KV compression, per-model thinking switch) and serves the picker API.
"$PY_BIN" "$HERE/localcode_supervisor.py" --model "$MODEL" --port "$PORT" --control-port "$CTRL" \
  --server "$SERVER" --models-dir "$MODELS_DIR" >> "$HERE/.run/supervisor.log" 2>&1 &
SUP=$!
# Keep the launcher alive to reap both children, including when the terminal
# closes or the frontend is terminated before its normal shutdown hook runs.
FRONTEND=""
cleanup() {
  [ -z "$FRONTEND" ] || kill "$FRONTEND" 2>/dev/null || true
  kill "$SUP" 2>/dev/null || true
  wait "$SUP" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
for i in $(seq 1 240); do
  if [ -n "$MODEL" ]; then curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  else curl -sf "http://127.0.0.1:$CTRL/status" >/dev/null && break; fi
  kill -0 $SUP 2>/dev/null || { echo "model failed to load (see $HERE/.run/supervisor.log)"; exit 1; }
  sleep 1
done

# Context window for THIS machine (RAM tier), from localcode's own server command.
CTX=$(curl -s "http://127.0.0.1:$CTRL/status" | sed -n 's/.*"ctx": \([0-9]*\).*/\1/p')
[ -n "$CTX" ] && [ "$CTX" != "0" ] || CTX=$("$PY_BIN" "$HERE/server_cmd.py" --ctx "$MODELS_DIR/x.gguf" 2>/dev/null || echo 32768)
# With no model yet, a hidden template entry carries the wiring; the picker
# synthesizes the real alias from it once a model is loaded.
ALIAS="${MODEL:-__pending__}"; ALIAS_NAME="${MODEL:-no model loaded}"
cd "$PROJECT"
# Project-local config. enabled_providers keeps the picker and model list to
# localcode only; capabilities are explicit so no cloud default leaks in.
cat > ./localcode.json <<JSON
{ "\$schema": "https://localcode.dev/schema/config.json",
  "provider": { "localcode": { "npm": "@ai-sdk/openai-compatible", "name": "localcode",
    "options": { "baseURL": "http://127.0.0.1:$PORT/v1", "apiKey": "local" },
    "models": { "$ALIAS": { "name": "$ALIAS_NAME",
      "tool_call": true, "reasoning": false, "temperature": true, "attachment": false,
      "modalities": { "input": ["text"], "output": ["text"] },
      "limit": { "context": $CTX, "output": 8192 } } } } },
  "enabled_providers": ["localcode"],
  "tools": { "task": false },
  "agent": { "plan": { "disable": true } },
  "lsp": true,$( [ -n "$MODEL" ] && printf '\n  "model": "localcode/%s",' "$MODEL" )
  "share": "disabled", "autoupdate": false }
JSON
# localcode's completion discipline (plan gate, build gate, stub audit, bash guard)
mkdir -p ./.localcode-agent/plugins && cp "$HERE/plugins/localcode.ts" ./.localcode-agent/plugins/localcode.ts
# Reading a file must never install software. Use language servers already
# available locally; installing another server is a separate user action.
export OPENCODE_DISABLE_LSP_DOWNLOAD=1
if [ -n "$MODEL" ]; then LOCALCODE_CONTROL_URL="http://127.0.0.1:$CTRL" "$BIN" -m "localcode/$MODEL" <&0 &
else LOCALCODE_CONTROL_URL="http://127.0.0.1:$CTRL" "$BIN" <&0 & fi
FRONTEND=$!
wait "$FRONTEND"
