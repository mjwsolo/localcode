#!/usr/bin/env bash
# Real build comparison: scaffold a working CLI app from nothing.
# Same controls as before: one shared warm server, direct connection, thinking
# off both sides, A/B interleaved, fresh empty dir per run, PASS decided by an
# objective acceptance script the agent never sees.
set -u
MODEL=Qwen3.8-27B-UD-Q4_K_XL
LC=$HOME/Desktop/Github/localcode/localcodevenv/bin/localcode
PI=$HOME/Desktop/Github/localcode-pi/agent-ts/dist/localcode-agent
EXT=$HOME/Desktop/Github/localcode-pi/agent-ts/extensions/localcode-provider.ts
PORT=8082; LOG=$PWD/q-server.log; RESULTS=$PWD/results-build.tsv
export LLAMA_BASE_URL=http://127.0.0.1:$PORT
N=${N:-3}
GOAL="$(cat "$PWD/build_spec.txt")"
: > "$RESULTS"

metrics() {
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

one() {
  local tool=$1 i=$2 D="$PWD/b-$1-$2"
  rm -rf "$D"; mkdir -p "$D"
  local b=$(grep -c 'prompt eval time' "$LOG") s=$(python3 -c 'import time;print(time.time())')
  if [ "$tool" = lc ]; then
    ( cd "$D" && timeout 1200 "$LC" --model "$MODEL" run --goal "$GOAL" --thinking off --quiet >/dev/null 2>&1 )
  else
    ( cd "$D" && timeout 1200 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
        --thinking off --no-session -p "$GOAL" >/dev/null 2>&1 )
  fi
  local e=$(python3 -c 'import time;print(time.time())') r=FAIL
  ./check_build.sh "$D" && r=PASS
  local files=$(find "$D" -name '*.py' -o -name '*.md' | wc -l | tr -d ' ')
  printf '%s\t%d\t%.1f\t%s\t%s\t%s\n' "$tool" "$i" "$(python3 -c "print($e-$s)")" "$r" "$(metrics "$b")" "$files" | tee -a "$RESULTS"
}

for i in $(seq 1 $N); do one lc "$i"; one pi "$i"; done
echo DONE
