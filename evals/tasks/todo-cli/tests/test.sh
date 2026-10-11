#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

[ -f todo.py ]; ck $? "todo.py exists"

if [ -f todo.py ]; then
  rm -f todos.json
  python3 todo.py add "buy milk" >/dev/null 2>&1
  python3 todo.py list 2>/dev/null | grep -q "0: \[ \] buy milk"; ck $? "add + list shows unchecked task"

  rm -f todos.json
  python3 todo.py add "one" >/dev/null 2>&1
  python3 todo.py add "two" >/dev/null 2>&1
  OUT=$(python3 todo.py list 2>/dev/null)
  { echo "$OUT" | grep -q "0: \[ \] one" && echo "$OUT" | grep -q "1: \[ \] two"; }; ck $? "two adds list in order"

  python3 todo.py done 0 >/dev/null 2>&1
  python3 todo.py list 2>/dev/null | grep -q "0: \[x\] one"; ck $? "done marks [x]"

  python3 todo.py rm 0 >/dev/null 2>&1
  OUT=$(python3 todo.py list 2>/dev/null)
  { echo "$OUT" | grep -q "two" && ! echo "$OUT" | grep -q "one"; }; ck $? "rm removes the task"

  rm -f todos.json
  python3 todo.py add "persisted" >/dev/null 2>&1
  python3 todo.py list 2>/dev/null | grep -q "persisted"; ck $? "state persists across invocations"

  rm -f todos.json
  python3 todo.py add "only" >/dev/null 2>&1
  python3 todo.py done 9 >/dev/null 2>&1; rc=$?
  [ "$rc" -ne 0 ]; ck $? "out-of-range index exits non-zero"
else
  total=7
fi

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
