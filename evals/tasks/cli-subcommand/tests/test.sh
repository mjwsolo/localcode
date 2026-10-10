#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
python3 -c "from app.cli import main; import io,contextlib; b=io.StringIO()
with contextlib.redirect_stdout(b): rc=main(['sum','1','2','3'])
assert rc==0 and b.getvalue().strip()=='6', b.getvalue()" 2>/dev/null || { pass_json false "sum subcommand wrong or missing"; exit 0; }
grep -rq "sum" tests/ || { pass_json false "no test for sum"; exit 0; }

if python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "sum works, tested"; else pass_json false "tests red"; fi
