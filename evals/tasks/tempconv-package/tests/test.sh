#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
[ -f pyproject.toml ] && [ -d src/tempconv ] || { pass_json false "no src/tempconv or pyproject"; exit 0; }
PYTHONPATH=src python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
from tempconv import c_to_f, f_to_c, c_to_k, k_to_c
assert abs(c_to_f(100) - 212) < 1e-9 and abs(f_to_c(32)) < 1e-9 and abs(c_to_k(0) - 273.15) < 1e-9 and abs(k_to_c(273.15)) < 1e-9
try:
    c_to_k(-300); raise SystemExit("no ValueError below absolute zero")
except ValueError:
    pass
PY

if [ $ok = 0 ] && PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider >/dev/null 2>&1; then pass_json true "package works, tests green"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
