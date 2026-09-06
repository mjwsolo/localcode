#!/usr/bin/env bash
# Controlled head-to-head: localcode vs pi.
#
# Controls applied:
#   - ONE single-model llama-server, both tools connect DIRECTLY to it (no router hop)
#   - identical model / quant / ctx-size / --jinja / -ngl for both
#   - server warmed with a throwaway request before any measurement
#   - thinking explicitly OFF for both
#   - fresh identical fixture per run; no AGENTS.md/CLAUDE.md anywhere in the tree
#   - A/B/A/B alternation so cache warmth and thermal drift hit both tools equally
#   - PASS verified by running pytest myself, not by trusting the agent's report
#   - per-run round trips + prompt/gen tokens read from the shared server log
set -u
MODEL=Qwen3.8-27B-UD-Q4_K_XL
GGUF=$HOME/.local/share/localcode/models/$MODEL.gguf
SRV=${LOCALCODE_ROOT:-$HOME/Desktop/Github/localcode}/src/localcode/bin/llama-server
LC=${LOCALCODE_ROOT:-$HOME/Desktop/Github/localcode}/localcodevenv/bin/localcode
PI=${LOCALCODE_PI_ROOT:-$HOME/Desktop/Github/localcode-pi}/agent-ts/dist/localcode-agent
EXT=${LOCALCODE_PI_ROOT:-$HOME/Desktop/Github/localcode-pi}/agent-ts/extensions/localcode-provider.ts
PORT=8082; LOG=$PWD/q-server.log; RESULTS=$PWD/results4.tsv
export LLAMA_BASE_URL=http://127.0.0.1:$PORT
N=${N:-3}
APPEND="$(cat "$PWD/append.txt")"
PVERB="$PWD/lc-prompt-verbatim.txt"
PMAP="$PWD/lc-prompt-mapped.txt"

pkill -f "port $PORT" 2>/dev/null; sleep 2; : > "$LOG"; : > "$RESULTS"
"$SRV" --host 127.0.0.1 --port $PORT --jinja -ngl 999 -c 32768 --alias "$MODEL" --model "$GGUF" >"$LOG" 2>&1 &
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
echo "server up; warming"
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}" >/dev/null
echo "warm"

metrics() { # $1=before-count -> "trips prompt gen"
python3 - "$LOG" "$1" <<'PY'
import sys,re
log,before=sys.argv[1],int(sys.argv[2])
L=[l for l in open(log,errors='ignore') if 'print_timing' in l]
pe=[l for l in L if 'prompt eval time' in l][before:]
ge=[l for l in L if re.search(r'\|\s+eval time',l) and 'prompt eval' not in l][before:]
n=lambda l: int(re.search(r'/\s+(\d+) tokens',l).group(1))
print(len(pe), sum(map(n,pe)), sum(map(n,ge)))
PY
}

one() { # $1=tool $2=task $3=idx
  local tool=$1 task=$2 i=$3 D="$PWD/q-$1-$2-$3" goal
  case $task in
    retry)  goal="Implement the retry decorator in retry.py so every test in test_retry.py passes. Then run: python3 -m pytest -q";;
    parser) goal="Implement column_stats in csvstat.py so every test in test_csvstat.py passes. Then run: python3 -m pytest -q";;
  esac
  ./fixture.sh "$D" "$task"
  local b=$(grep -c 'prompt eval time' "$LOG") s=$(python3 -c 'import time;print(time.time())')
  if [ "$tool" = lc ]; then
    ( cd "$D" && timeout 900 "$LC" --model "$MODEL" run --goal "$goal" --thinking off --quiet >/dev/null 2>&1 )
  elif [ "$tool" = pi ]; then
    ( cd "$D" && timeout 900 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
        --thinking off --no-session -p "$goal" >/dev/null 2>&1 )
  elif [ "$tool" = piverb ]; then
    ( cd "$D" && timeout 900 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
        --thinking off --no-session --system-prompt "$(sed "s|{CWD}|$D|g" "$PVERB")" -p "$goal" >/dev/null 2>&1 )
  else
    ( cd "$D" && timeout 900 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
        --thinking off --no-session --system-prompt "$(sed "s|{CWD}|$D|g" "$PMAP")" -p "$goal" >/dev/null 2>&1 )
  fi
  local e=$(python3 -c 'import time;print(time.time())') r=FAIL
  ( cd "$D" && python3 -m pytest -q >/dev/null 2>&1 ) && r=PASS
  local m=$(metrics "$b")
  printf '%s\t%s\t%d\t%.1f\t%s\t%s\n' "$tool" "$task" "$i" "$(python3 -c "print($e-$s)")" "$r" "$m" | tee -a "$RESULTS"
}

for task in retry parser; do
  for i in $(seq 1 $N); do one lc "$task" "$i"; one pi "$task" "$i"; one piverb "$task" "$i"; one pimap "$task" "$i"; done
done
echo DONE
