#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
from inventory.models import Item
from inventory.store import Store
from inventory.report import render
s = Store(); s.add(Item("a", 2, 3.0)); s.add(Item("b", 1, 5.0))
assert abs(s.total_value() - 11.0) < 1e-9
out = render(s); assert "TOTAL" in out and out.count("\n") >= 2, out
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider >/dev/null 2>&1; then pass_json true "three modules cooperate, tests green"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
