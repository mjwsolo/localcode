#!/usr/bin/env bash
# Record every tape in a scratch project and install the GIFs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
command -v vhs >/dev/null || { echo "brew install vhs"; exit 1; }
PROJ="$(mktemp -d)/retry-demo"; mkdir -p "$PROJ"
printf 'def retry(times):\n    pass\n' > "$PROJ/retry.py"
cat > "$PROJ/test_retry.py" <<'PY'
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
export PATH="${LOCALCODE_BIN_DIR:-$ROOT/.venv/bin}:$PATH"
export LC_DEMO_PROJECT="$PROJ"
cd "$HERE"
mkdir -p "$HERE/out"
TAPES=("${@:-step-2-choose-model step-3-ask step-4-verify first-change}")
for t in ${TAPES[@]}; do
  rm -rf "$PROJ"/.localcode-agent "$PROJ"/__pycache__ "$PROJ"/.pytest_cache; printf 'def retry(times):\n    pass\n' > "$PROJ/retry.py"
  vhs "$t.tape"
  if command -v gifsicle >/dev/null; then gifsicle -O3 --lossy=60 --colors 128 -o "$HERE/out/$t.opt.gif" "$HERE/out/$t.gif" && mv "$HERE/out/$t.opt.gif" "$HERE/out/$t.gif"; fi
  ls -la "$HERE/out/$t.gif"
done
cp "$HERE"/out/*.gif "$ROOT/website/public/demo/"
[ -f "$HERE/out/first-change-final.png" ] && cp "$HERE/out/first-change-final.png" "$ROOT/website/public/demo/"
cp "$HERE/out/first-change.gif" "$ROOT/docs/assets/demo/first-change.gif"
echo "installed into website/public/demo and docs/assets/demo"
