#!/usr/bin/env bash
# sysctl_sweep.sh — measure whether iogpu.wired_limit_mb can be lowered.
# Uses osascript for the admin prompt so it works in non-interactive shells.

set -uo pipefail

MODEL="${MODEL:-$HOME/.local/share/localcode/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}"
LLAMA="${LLAMA:-$HOME/llama-cpp-turboquant/build/bin/llama-server}"
PORT=8181
RESULTS="benchmarks/results/sysctl_sweep.tsv"
mkdir -p "$(dirname "$RESULTS")"

set_sysctl() {
  local cap="$1"
  osascript -e "do shell script \"/usr/sbin/sysctl iogpu.wired_limit_mb=${cap}\" with administrator privileges" >/dev/null 2>&1
  return $?
}

restore() {
  echo "[restore] sysctl → 14336"
  set_sysctl 14336 || true
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
}
trap restore EXIT

echo -e "sysctl_mb\tlaunch_ok\tgraph_splits\tkv_buf_mib\tdecode_tok_s\tprompt_tok_s\twired_peak_gb" > "$RESULTS"

probe_level() {
  local cap="$1"
  local label="$2"
  echo ""
  echo "==================================================="
  echo "=== sysctl iogpu.wired_limit_mb = $cap  ($label)"
  echo "==================================================="
  set_sysctl "$cap" || { echo "sysctl set failed"; return; }

  local LOG="/tmp/sysctl_sweep_${cap}.log"
  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 1

  "$LLAMA" --model "$MODEL" -ngl 999 --mmap \
    -ctk q8_0 -ctv turbo4 -fa on -c 8192 \
    --threads 10 -b 2048 -ub 512 -np 1 -fit off --cache-ram 0 \
    --jinja --port "$PORT" > "$LOG" 2>&1 &
  local PID=$!

  local ready=0
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    sleep 1
  done

  local splits
  splits=$(grep -oE "graph splits = [0-9]+" "$LOG" | tail -1 | awk '{print $NF}')
  local kvbuf
  kvbuf=$(grep -oE "KV buffer size = *[0-9.]+ MiB" "$LOG" | head -1 | grep -oE '[0-9.]+' | head -1)

  if [ "$ready" = "0" ]; then
    echo "FAILED to start (see $LOG)"
    tail -20 "$LOG"
    echo -e "${cap}\tFAIL\t${splits:--}\t${kvbuf:--}\t-\t-\t-" >> "$RESULTS"
    kill $PID 2>/dev/null || true
    wait $PID 2>/dev/null || true
    return
  fi

  curl -s -o /dev/null "http://127.0.0.1:$PORT/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"def hello():","n_predict":20,"temperature":0,"stream":false}'

  local RESP
  RESP=$(curl -s "http://127.0.0.1:$PORT/completion" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Write a Python merge_sort function.","n_predict":200,"temperature":0,"stream":false}')

  local post_wired
  post_wired=$(vm_stat | awk '/Pages wired down/ {gsub(/\./,""); printf "%.2f", $4*16384/1073741824}')

  local dec prm
  dec=$(echo "$RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"{d[\"timings\"][\"predicted_per_second\"]:.1f}")' 2>/dev/null || echo "-")
  prm=$(echo "$RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"{d[\"timings\"][\"prompt_per_second\"]:.1f}")' 2>/dev/null || echo "-")

  echo "  splits: ${splits:-?}  KV: ${kvbuf:-?} MiB"
  echo "  decode: $dec tok/s   prompt: $prm tok/s   wired peak: $post_wired GB"

  echo -e "${cap}\tOK\t${splits:--}\t${kvbuf:--}\t${dec}\t${prm}\t${post_wired}" >> "$RESULTS"

  pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
  sleep 2
}

probe_level 14336 "baseline"
probe_level 12288 "-2 GB"
probe_level 10240 "near default"
probe_level 0     "kernel default"

echo ""
echo "=== SUMMARY → $RESULTS ==="
cat "$RESULTS" | column -t -s$'\t'
