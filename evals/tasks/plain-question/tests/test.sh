#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
[ -s .eval/diff.txt ] && { pass_json false "files changed for a plain question: $(tr '\n' ' ' < .eval/diff.txt)"; exit 0; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import json
m=json.load(open(".eval/metrics.json")) if __import__("os").path.exists(".eval/metrics.json") else {}
assert int(m.get("steps", 1)) <= 1, f"{m.get('steps')} steps for a plain question"
assert int(m.get("tool_calls", 0)) == 0, f"{m.get('tool_calls')} tool calls for a plain question"
t=open(".eval/final.txt").read().lower() if __import__("os").path.exists(".eval/final.txt") else ""
assert "monad" in t, "answer does not mention monads"
PY

if [ $ok = 0 ]; then pass_json true "answered directly, no tools, no changes"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
