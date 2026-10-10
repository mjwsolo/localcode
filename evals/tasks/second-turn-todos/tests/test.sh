#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/calc.py)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/calc.py)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

[ -s .eval/ui_errors.txt ] && { pass_json false "runtime errors: $(head -c 200 .eval/ui_errors.txt)"; exit 0; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import ast
t=ast.parse(open("app/calc.py").read()); fns={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
assert ast.get_docstring(fns["add"]) and ast.get_docstring(fns["div"]), "both docstrings expected after turn 2"
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "two turns with open todos, no failed turn"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
