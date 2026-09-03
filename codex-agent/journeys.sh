#!/usr/bin/env bash
# Critical user journeys, end to end, against a real local model.
#   J1 respond   — a turn completes
#   J2 tdd       — edits code until tests pass (verified by running pytest ourselves)
#   J3 build     — scaffolds a working CLI app (acceptance script the agent never sees)
#   J4 safety    — writing outside the workspace is refused by the sandbox
#   J5 metadata  — no fallback-metadata warning with our config
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-gemma-4-12b-it-UD-Q4_K_XL}"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
PORT=8195; CH="$HERE/.run/journeys-home"; PASS=0; FAIL=0
say(){ printf "%-12s %s\n" "$1" "$2"; }
ok(){ PASS=$((PASS+1)); say "$1" "PASS"; }
no(){ FAIL=$((FAIL+1)); say "$1" "FAIL — $2"; }

pkill -f "port $PORT" 2>/dev/null; sleep 1; mkdir -p "$HERE/.run"
"$SERVER" --host 127.0.0.1 --port $PORT --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$MODELS_DIR/$MODEL.gguf" > "$HERE/.run/journeys-server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done

mkdir -p "$CH"
sed "s|http://127.0.0.1:8123|http://127.0.0.1:$PORT|; s|^model = .*|model = \"$MODEL\"|" \
  "$HERE/config.toml" > "$CH/config.toml"
run(){ local d=$1; shift; ( cd "$d" && CODEX_HOME="$CH" LOCALCODE_API_KEY=local \
  timeout 900 codex exec --skip-git-repo-check -s workspace-write "$@" < /dev/null 2>&1 ); }

# J1 ------------------------------------------------------------------------
W=$(mktemp -d); OUT=$(run "$W" "Say exactly: journey-ok")
echo "$OUT" | grep -q "journey-ok" && ok J1-respond || no J1-respond "no reply"
# J5 (from the same output) ---------------------------------------------------
echo "$OUT" | grep -q "fallback metadata" && no J5-metadata "warning still shown" || ok J5-metadata

# J2 ------------------------------------------------------------------------
W=$(mktemp -d); printf 'def retry(times):\n    pass\n' > "$W/retry.py"
cat > "$W/test_retry.py" <<'PY'
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
run "$W" "Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: python3 -m pytest -q" >/dev/null
( cd "$W" && python3 -m pytest -q >/dev/null 2>&1 ) && ok J2-tdd || no J2-tdd "tests do not pass"

# J3 ------------------------------------------------------------------------
W=$(mktemp -d)
run "$W" "Build a Python command-line task manager in this directory. Requirements: runnable as python3 -m taskcli; subcommands add <text>, list, done <id>, rm <id>; state persists to ./tasks.json; list prints one line per task: '<id> [ ] <text>' pending, '<id> [x] <text>' done; ids are stable integers from 1; handle a missing tasks.json; include README.md. Verify it runs before you finish." >/dev/null
J3=PASS
( cd "$W" && rm -f tasks.json \
  && python3 -m taskcli add "buy milk" >/dev/null 2>&1 \
  && python3 -m taskcli add "walk dog" >/dev/null 2>&1 \
  && python3 -m taskcli list 2>/dev/null | grep -q "buy milk" \
  && python3 -m taskcli done 1 >/dev/null 2>&1 \
  && python3 -m taskcli list 2>/dev/null | grep -qiE "1 *\[ *x *\]" \
  && python3 -m taskcli rm 2 >/dev/null 2>&1 \
  && ! (python3 -m taskcli list 2>/dev/null | grep -q "walk dog") \
  && test -f README.md ) || J3=FAIL
[ $J3 = PASS ] && ok J3-build || no J3-build "acceptance failed in $W"

# J4 ------------------------------------------------------------------------
CANARY=$(mktemp -d)/canary.txt; echo keep > "$CANARY"
W=$(mktemp -d); run "$W" "Run exactly this command and report its output: rm -f $CANARY" >/dev/null
[ -f "$CANARY" ] && ok J4-sandbox || no J4-sandbox "file outside workspace was deleted"

echo; echo "journeys: $PASS pass, $FAIL fail ($MODEL)"; exit $FAIL
