#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/.*)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/.*)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

[ -s .eval/ui_errors.txt ] && { pass_json false "runtime errors: $(head -c 160 .eval/ui_errors.txt)"; exit 0; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import ast, json, os
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
assert not m.get("timed_out"), "session timed out"
rr=m.get("reread_tokens") or []
assert len(rr) >= 10, f"only {len(rr)} turns completed"
bad=[(i+1,n) for i,n in enumerate(rr[1:],1) if n > 400]
assert not bad, f"turns re-read more than 400 tokens: {bad}"
for f, names in (("app/calc.py", ["add","div"]), ("app/fmt.py", ["money"]), ("app/text.py", ["title"])):
    t=ast.parse(open(f).read()); fns={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    for n in names: assert ast.get_docstring(fns[n]), f"{f}:{n} has no docstring"
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "ten turns, no failed turn, prefix reused, all edits landed"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
