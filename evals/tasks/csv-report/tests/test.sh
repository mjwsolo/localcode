#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

# A header-only CSV for the empty-input checks.
printf 'category,amount\n' > empty.csv

{ [ -f report.py ] && [ -f test_report.py ]; }; ck $? "report.py and test_report.py exist"
pchk "from report import totals; assert totals('sales.csv')=={'food':18,'toys':27}, totals('sales.csv')"; ck $? "totals correct"
pchk "from report import top_category; assert top_category('sales.csv')=='toys'"; ck $? "top_category correct"
pchk "from report import totals; assert totals('empty.csv')=={}"; ck $? "empty CSV totals == {}"
pchk "from report import top_category; assert top_category('empty.csv') is None"; ck $? "empty CSV top_category is None"
python3 -m pytest -q -p no:cacheprovider test_report.py >/dev/null 2>&1; ck $? "pytest test_report.py passes"

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
