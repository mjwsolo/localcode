#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import json, os
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
rr=m.get("reread_tokens") or []
assert len(rr) >= 5, f"need per-turn reread tokens, got {rr}"
bad=[(i+1,n) for i,n in enumerate(rr[1:],1) if n > 400]
assert not bad, f"turns re-read more than 400 tokens: {bad}"
PY

if [ $ok = 0 ]; then pass_json true "prefix reused on every turn after the first"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
