#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(shipping\.py)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(shipping\.py)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

# One check per test in test_shipping.py; a missing or edited test file scores 0.
total=$(grep -c "def test_" test_shipping.py 2>/dev/null); [ "${total:-0}" -gt 0 ] 2>/dev/null || total=6
if [ ! -f shipping.py ] || [ ! -f test_shipping.py ]; then
  first="required files missing"
elif [ -z "$(sha_of test_shipping.py)" ] || [ "$(sha_of test_shipping.py)" != "$(cat .test_shipping.sha 2>/dev/null)" ]; then
  first="test_shipping.py was modified"
else
  OUT=$(python3 -m pytest -q -p no:cacheprovider test_shipping.py --tb=no -rfE 2>&1)

  pass=$(echo "$OUT" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" | head -1); pass=${pass:-0}
  first=$(echo "$OUT" | grep -E "^(FAILED|ERROR) " | head -1 | sed -E 's/^(FAILED|ERROR) //; s/ - .*//; s/^.*:://')
  [ -n "$first" ] || first="pytest reported $pass of $total tests passed"
fi

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
