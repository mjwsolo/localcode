#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

pchk "import calc; assert callable(calc.evaluate)"; ck $? "calc.evaluate exists"
pchk "import calc; assert abs(calc.evaluate('2+3*4')-14)<1e-9"; ck $? "precedence 2+3*4==14"
pchk "import calc; assert abs(calc.evaluate('(2+3)*4')-20)<1e-9"; ck $? "parens (2+3)*4==20"
pchk "import calc; assert abs(calc.evaluate('10-3-2')-5)<1e-9"; ck $? "left-assoc 10-3-2==5"
pchk "import calc; assert abs(calc.evaluate('-3+5')-2)<1e-9 and abs(calc.evaluate('2*-3')+6)<1e-9"; ck $? "unary minus"
pchk "import calc; assert abs(calc.evaluate('7/2')-3.5)<1e-9"; ck $? "real division 7/2==3.5"
pchk "import calc; assert abs(calc.evaluate('((1+2)*(3+4))')-21)<1e-9"; ck $? "nested parens==21"
pchk "import calc
try:
    calc.evaluate('2+'); raise SystemExit(1)
except ValueError:
    pass"; ck $? "malformed raises ValueError"

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
