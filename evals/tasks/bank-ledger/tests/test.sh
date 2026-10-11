#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

{ [ -f account.py ] && [ -f ledger.py ] && [ -f test_ledger.py ]; }; ck $? "account.py, ledger.py, test_ledger.py exist"
grep -Eq "from[[:space:]]+account[[:space:]]+import|import[[:space:]]+account" ledger.py 2>/dev/null; ck $? "ledger.py imports Account from account"
pchk "from account import Account
a=Account('x',10); a.deposit(5); assert a.balance==15; a.withdraw(7); assert a.balance==8"; ck $? "deposit/withdraw/balance work"
pchk "from account import Account
a=Account('x',10)
try: a.withdraw(99); raise SystemExit(1)
except ValueError: pass"; ck $? "overdraw raises ValueError"
pchk "from ledger import Ledger
L=Ledger(); a=L.open('alice',100); assert a.balance==100"; ck $? "Ledger.open creates account with balance"
pchk "from ledger import Ledger
L=Ledger(); L.open('alice',100); L.open('bob',0); L.transfer('alice','bob',30)
import ledger as _; assert L.total()==100"; ck $? "transfer moves money"
pchk "from ledger import Ledger
L=Ledger(); L.open('alice',10); L.open('bob',0)
try: L.transfer('alice','bob',999); raise SystemExit(1)
except ValueError: pass"; ck $? "insufficient transfer raises ValueError"
pchk "from ledger import Ledger
L=Ledger(); L.open('a',40); L.open('b',60); assert L.total()==100"; ck $? "total() sums balances"
python3 -m pytest -q -p no:cacheprovider test_ledger.py >/dev/null 2>&1; ck $? "pytest test_ledger.py passes"

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
