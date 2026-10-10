#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/calc.py)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/calc.py)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import ast, json, os
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
tc=m.get("tool_calls_per_turn") or []
if tc: assert tc[1] == 0, f"the unrelated question triggered {tc[1]} tool call(s)"
t=open(".eval/final_turn2.txt").read().lower() if os.path.exists(".eval/final_turn2.txt") else ""
assert "monad" in t, "turn 2 did not answer the question"
tree=ast.parse(open("app/calc.py").read()); fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
assert ast.get_docstring(fns["add"]) and ast.get_docstring(fns["div"]), "the code task was not finished after the interruption"
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "question answered without tools, code task resumed and finished"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
