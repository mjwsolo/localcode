#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(todo\.py|test_todo_ext\.py)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(todo\.py|test_todo_ext\.py)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

[ -f todo.py ]; ck $? "todo.py exists"
[ -f test_todo_ext.py ]; ck $? "test_todo_ext.py exists"
grep -q "def test_" test_todo_ext.py 2>/dev/null; ck $? "test_todo_ext.py has a test_ function"
[ -n "$(sha_of test_todo.py)" ] && [ "$(sha_of test_todo.py)" = "$(cat .test_todo.sha 2>/dev/null)" ]; ck $? "test_todo.py unchanged"
python3 - >/dev/null 2>&1 <<'PY'
from todo import TodoList
t = TodoList()
for x in ("a", "b", "c"):
    t.add(x)
t.complete(1)
assert t.pending() == ["a", "c"], t.pending()
t.remove(0)
assert [i["text"] for i in t.all()] == ["b", "c"], t.all()
PY
ck $? "remove/pending behaviour"
python3 -m pytest -q -p no:cacheprovider >/dev/null 2>&1; ck $? "pytest passes (both test files)"

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
