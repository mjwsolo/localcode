#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
python3 - <<'PY'
p="app/calc.py"; s=open(p).read(); open(p,"w").write(s.replace("def add(a, b):\n", "def add(a, b):\n    \"\"\"Return a plus b.\"\"\"\n"))
PY
echo '{"steps": 3, "tool_calls": 2}' > .eval/metrics.json
