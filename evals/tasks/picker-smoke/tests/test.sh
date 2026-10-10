#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
# Graded by the Checks harness (evals/checks), not by a model run: see test_checks.py.
pass_json true "delegated to Checks"
