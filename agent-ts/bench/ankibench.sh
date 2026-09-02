#!/usr/bin/env bash
# Real-project comparison on a flashcard app, three variants:
#   lc     - localcode
#   pi     - pi with its own stock system prompt
#   pisame - pi running localcode's system prompt (tool names mapped) => closest
#            achievable "identical system prompt" comparison
# PASS requires npm install + npm run build to actually succeed.
set -u
MODEL=Qwen3.8-27B-UD-Q4_K_XL
LC=$HOME/Desktop/Github/localcode/localcodevenv/bin/localcode
PI=$HOME/Desktop/Github/localcode-pi/agent-ts/dist/localcode-agent
EXT=$HOME/Desktop/Github/localcode-pi/agent-ts/extensions/localcode-provider.ts
PORT=8082; LOG=$PWD/q-server.log; RESULTS=$PWD/results-anki.tsv
export LLAMA_BASE_URL=http://127.0.0.1:$PORT
N=${N:-2}
GOAL="$(cat "$PWD/anki_spec.txt")"
PMAP="$PWD/lc-prompt-mapped.txt"
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
  local tool=$1 i=$2 D="$PWD/a-$1-$2"
  rm -rf "$D"; mkdir -p "$D"
  local b=$(grep -c 'prompt eval time' "$LOG") s=$(python3 -c 'import time;print(time.time())')
  case $tool in
    lc) ( cd "$D" && timeout 2400 "$LC" --model "$MODEL" run --goal "$GOAL" --thinking off --quiet >/dev/null 2>&1 );;
    pi) ( cd "$D" && timeout 2400 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
            --thinking off --no-session -p "$GOAL" >/dev/null 2>&1 );;
    pisame) ( cd "$D" && timeout 2400 "$PI" -a -e "$EXT" --provider localcode --model "$MODEL" \
            --thinking off --no-session --system-prompt "$(sed "s|{CWD}|$D|g" "$PMAP")" -p "$GOAL" >/dev/null 2>&1 );;
  esac
  local e=$(python3 -c 'import time;print(time.time())') r=FAIL
  ./check_anki.sh "$D" && r=PASS
  local files=$(find "$D/src" -type f 2>/dev/null | wc -l | tr -d ' ')
  printf '%s\t%d\t%.1f\t%s\t%s\t%s\n' "$tool" "$i" "$(python3 -c "print($e-$s)")" "$r" "$(metrics "$b")" "$files" | tee -a "$RESULTS"
}

for i in $(seq 1 $N); do one lc "$i"; one pi "$i"; one pisame "$i"; done
echo DONE
