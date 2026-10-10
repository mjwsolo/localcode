#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import json, os, ast
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
assert int(m.get("steps", 99)) <= 12, f"{m.get('steps')} steps for a one-line change"
t=ast.parse(open("app/calc.py").read()); fns={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
assert ast.get_docstring(fns["add"]), "docstring missing"
PY

if [ $ok = 0 ]; then pass_json true "done within the step budget"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
