#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import os
t = open("answer.txt").read() if os.path.exists("answer.txt") else ""
assert "29" in t, f"Pro price not found in answer.txt: {t[:80]!r}"
assert "ZX-4471" in t, f"reference code not found in answer.txt: {t[:80]!r}"
assert os.path.exists("site/contact.log") and "Eval Bot" in open("site/contact.log").read(), "contact form was not submitted as Eval Bot"
PY

if [ $ok = 0 ]; then pass_json true "price read, form submitted, code read"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
