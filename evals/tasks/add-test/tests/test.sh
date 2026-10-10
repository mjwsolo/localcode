#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(tests/.*)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(tests/.*)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

grep -rqE "ZeroDivisionError" tests/ || { pass_json false "no test mentions ZeroDivisionError"; exit 0; }

if python3 -m pytest -q -p no:cacheprovider tests 2>/dev/null | grep -qE '^3 passed|[3-9] passed'; then pass_json true "new test present and green"; else pass_json false "suite not green with >=3 tests"; fi
