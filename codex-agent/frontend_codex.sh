#!/usr/bin/env bash
# Run codex as localcode's front end: start the bundled llama-server, point an
# isolated CODEX_HOME at it, hand over the terminal. ~/.codex is never touched.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
MODEL="${1:-}"; shift 2>/dev/null || true
if [ -z "$MODEL" ]; then
  # No argument: pick from a menu of what's on disk, largest first.
  echo "Which model?"
  i=0; MODELS=()
  while IFS= read -r f; do
    i=$((i+1)); MODELS+=("$f")
    printf "  %2d) %-40s %s GB
" "$i" "$f" "$(du -g "$MODELS_DIR/$f.gguf" 2>/dev/null | cut -f1)"
  done < <(ls -S "$MODELS_DIR" | grep '\.gguf$' | grep -v '^mmproj' | sed 's/\.gguf$//')
  printf "> "; read -r pick
  MODEL="${MODELS[$((pick-1))]:-}"
  [ -n "$MODEL" ] || { echo "no such choice"; exit 1; }
fi
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
