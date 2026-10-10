#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
python3 - <<'PY'
p="app/calc.py"; s=open(p).read()
s=s.replace("def add(a, b):\n", "def add(a, b):\n    \"\"\"Return a plus b.\"\"\"\n").replace("def div(a, b):\n", "def div(a, b):\n    \"\"\"Return a divided by b.\"\"\"\n")
open(p,"w").write(s)
PY
echo '{"steps": 6, "tool_calls": 5, "tool_calls_per_turn": [3, 0, 2]}' > .eval/metrics.json
echo "A monad is a wrapper for computations that lets you chain operations while handling context like state or failure." > .eval/final_turn2.txt
