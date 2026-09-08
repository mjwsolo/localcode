#!/usr/bin/env bash
# Run the localcode-branded opencode fork as localcode's front end.
#
#   pick model → pick quant (or pass an alias)  →  supervisor starts the bundled
#   llama-server on a free port and serves the in-TUI /models picker on a
#   localhost control port  →  project-local localcode.json points the fork at
#   that server  →  exec the fork binary.
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
if [ -z "$MODEL" ]; then
  # Two-level picker, the localcode way: model first, then quant.
  MODEL="$("$PY_BIN" "$HERE/model_picker_cli.py")" || exit $?
  [ -n "$MODEL" ] || { echo "no model chosen"; exit 1; }
fi
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""; CTRL=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
for p in $(seq 8323 8399); do curl -sf "http://127.0.0.1:$p/status" >/dev/null 2>&1 || { CTRL=$p; break; }; done
mkdir -p "$HERE/.run"
# The supervisor owns llama-server (per-machine flags from server_cmd.py: RAM-tier
# context, KV compression, per-model thinking switch) and serves the picker API.
"$PY_BIN" "$HERE/localcode_supervisor.py" --model "$MODEL" --port "$PORT" --control-port "$CTRL" \
  --server "$SERVER" --models-dir "$MODELS_DIR" > "$HERE/.run/supervisor.log" 2>&1 &
SUP=$!; trap 'kill $SUP 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SUP 2>/dev/null || { echo "model failed to load (see $HERE/.run/supervisor.log)"; exit 1; }
  sleep 1
done

# Context window comes from the supervisor, which computed it for THIS machine.
CTX=$(curl -s "http://127.0.0.1:$CTRL/status" | sed -n 's/.*"ctx": \([0-9]*\).*/\1/p'); CTX="${CTX:-32768}"
cd "$PROJECT"
# Project-local config. enabled_providers keeps the picker and model list to
# localcode only; capabilities are explicit so no cloud default leaks in.
cat > ./localcode.json <<JSON
{ "\$schema": "https://localcode.dev/schema/config.json",
  "provider": { "localcode": { "npm": "@ai-sdk/openai-compatible", "name": "localcode",
    "options": { "baseURL": "http://127.0.0.1:$PORT/v1", "apiKey": "local" },
    "models": { "$MODEL": { "name": "$MODEL",
      "tool_call": true, "reasoning": false, "temperature": true, "attachment": false,
      "limit": { "context": $CTX, "output": 8192 } } } } },
  "enabled_providers": ["localcode"],
  "model": "localcode/$MODEL", "share": "disabled", "autoupdate": false }
JSON
# localcode's completion discipline (plan gate, build gate, stub audit, bash guard)
mkdir -p ./.localcode-agent/plugins && cp "$HERE/plugins/localcode.ts" ./.localcode-agent/plugins/localcode.ts
LOCALCODE_CONTROL_URL="http://127.0.0.1:$CTRL" exec "$BIN" -m "localcode/$MODEL"
