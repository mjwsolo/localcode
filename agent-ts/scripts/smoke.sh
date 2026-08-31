#!/usr/bin/env bash
# Phase-0/CI smoke: proves the whole chain still works after a pi bump or a
# llama.cpp bump. Exits non-zero with a named reason, never silently.
#
#   ./scripts/smoke.sh [model-id]
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-gemma-4-12b-it-UD-Q4_K_XL}"
PORT="${LOCALCODE_SMOKE_PORT:-8099}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../../localcode/src/localcode/bin/llama-server}"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
export LLAMA_BASE_URL="http://127.0.0.1:$PORT"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
[ -x "$SERVER" ] || fail "no llama-server at $SERVER"

# 1. router mode must exist in whatever llama.cpp vintage we vendor
"$SERVER" --models-dir "$MODELS_DIR" --no-models-autoload --jinja \
  --host 127.0.0.1 --port "$PORT" -ngl 999 -c 32768 > "$HERE/.run/smoke-router.log" 2>&1 &
ROUTER=$!
trap 'kill $ROUTER 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/models" >/dev/null || fail "router /models missing (llama.cpp lost router mode?)"

# 2. the model must load through the router
curl -sf -X POST "http://127.0.0.1:$PORT/models/load" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\"}" | grep -q '"success":true' || fail "router refused to load $MODEL"

# 3. pi must see our provider (catches extension-API drift on a pi bump)
"$HERE/dist/localcode-agent" -e "$HERE/extensions/localcode-provider.ts" --list-models \
  | grep -q "^localcode " || fail "localcode provider did not register (pi extension API drift?)"

# 4. a real agentic turn on the floor model must edit a file and pass a test
WORK="$(mktemp -d)"; trap 'kill $ROUTER 2>/dev/null || true; rm -rf "$WORK"' EXIT
printf 'def retry(times):\n    pass\n' > "$WORK/retry.py"
cat > "$WORK/test_retry.py" <<'PY'
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
( cd "$WORK" && "$HERE/dist/localcode-agent" -a -e "$HERE/extensions/localcode-provider.ts" \
    --provider localcode --model "$MODEL" --no-session \
    -p "Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: python3 -m pytest -q" >/dev/null 2>&1 )
( cd "$WORK" && python3 -m pytest -q >/dev/null 2>&1 ) || fail "agentic turn did not make the test pass on $MODEL"

echo "SMOKE OK: pi $(cat "$HERE/PINNED_PI") + router + $MODEL"
