#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/.*)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/.*)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

if python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "tests green, only app/ changed"; else pass_json false "tests still red"; fi
