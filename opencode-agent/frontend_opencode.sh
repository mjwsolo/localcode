#!/usr/bin/env bash
# Run opencode as localcode's front end: bundled llama-server + project-local
# opencode.json. Global ~/.config/opencode is never touched.
set -euo pipefail
export PATH="$HOME/.hermes/node/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${LOCALCODE_MODELS_DIR:-$HOME/.local/share/localcode/models}"
MODEL="${1:-}"; shift 2>/dev/null || true
PROJECT="${1:-$PWD}"
if [ -z "" ]; then
# Two-level picker, the localcode way: model first, then quant.
# Reads MODELS_DIR; sets MODEL. Pure bash, no deps.
pick_model() {
  local files=() fams=() f fam q i pick
  while IFS= read -r f; do files+=("$f"); done \
    < <(ls -S "$MODELS_DIR" | grep '\.gguf$' | grep -v '^mmproj' | sed 's/\.gguf$//')
  # family = filename minus the trailing quant tag
  for f in "${files[@]}"; do
    fam=$(echo "$f" | sed -E 's/-(UD-)?(IQ|Q|BF|F)[0-9][0-9A-Za-z_]*$//')
    case " ${fams[*]-} " in (*" $fam "*) ;; (*) fams+=("$fam");; esac
  done
  while :; do
    echo "Which model?"
    i=0; for fam in "${fams[@]}"; do
      i=$((i+1)); n=$(printf '%s
' "${files[@]}" | grep -c "^${fam}-")
      printf "  %2d) %-32s %s quant(s) downloaded
" "$i" "$fam" "$n"
    done
    printf "> "; read -r pick
    case "$pick" in (*[!0-9]*|"") echo "  type a number 1-$i"; continue;; esac
    fam="${fams[$((pick-1))]:-}"; [ -n "$fam" ] || { echo "  type a number 1-$i"; continue; }
    echo; echo "$fam — which quant?"
    local quants=() 
    while IFS= read -r q; do quants+=("$q"); done < <(printf '%s
' "${files[@]}" | grep "^${fam}-")
    i=0; for q in "${quants[@]}"; do
      i=$((i+1)); tag=${q#"$fam"-}
      printf "  %2d) %-16s %s GB
" "$i" "$tag" "$(du -g "$MODELS_DIR/$q.gguf" 2>/dev/null | cut -f1)"
    done
    printf "   b) back
> "; read -r pick
    [ "$pick" = b ] && { echo; continue; }
    case "$pick" in (*[!0-9]*|"") echo "  type a number"; continue;; esac
    MODEL="${quants[$((pick-1))]:-}"
    [ -n "$MODEL" ] && return 0
    echo "  type a number 1-$i"
  done
}

  pick_model
fi
SERVER="${LOCALCODE_LLAMA_SERVER:-$HERE/../src/localcode/bin/llama-server}"
GGUF="$MODELS_DIR/$MODEL.gguf"
[ -f "$GGUF" ] || { echo "No such model: $GGUF"; ls "$MODELS_DIR" | grep '\.gguf$' | grep -v mmproj | sed 's/\.gguf$//;s/^/  /'; exit 1; }

PORT=""
for p in $(seq 8123 8199); do curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { PORT=$p; break; }; done
mkdir -p "$HERE/.run"
"$SERVER" --host 127.0.0.1 --port "$PORT" --jinja -ngl 999 -c 32768 \
  --alias "$MODEL" --model "$GGUF" --chat-template-kwargs '{"enable_thinking":false}' > "$HERE/.run/server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done

cd "$PROJECT"
sed "s|8123|$PORT|" "$HERE/opencode.json" > ./opencode.json
exec opencode -m "localcode/$MODEL"
