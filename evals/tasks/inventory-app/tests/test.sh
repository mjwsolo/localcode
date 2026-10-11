#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
pass=0; total=0; first=""
ck() { total=$((total + 1)); if [ "$1" -eq 0 ]; then pass=$((pass + 1)); elif [ -z "$first" ]; then first="$2"; fi; }
pchk() { python3 -c "$1" >/dev/null 2>&1; }
sha_of() { python3 -c "import hashlib,sys; print(hashlib.sha1(open(sys.argv[1],'rb').read()).hexdigest())" "$1" 2>/dev/null; }

{ [ -f models.py ] && [ -f store.py ] && [ -f report.py ] && [ -f test_inventory.py ]; }; ck $? "all four modules exist"
grep -Eq "from[[:space:]]+models[[:space:]]+import|import[[:space:]]+models" store.py 2>/dev/null; ck $? "store.py imports Item from models"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore(); s.add(Item('a','Apple',10,2.0)); it=s.get('a')
assert it.sku=='a' and it.name=='Apple' and it.qty==10 and it.price==2.0"; ck $? "add + get round-trip"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore(); s.add(Item('a','Apple',10,2.0)); s.restock('a',5)
assert s.get('a').qty==15"; ck $? "restock increases qty"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore(); s.add(Item('a','Apple',10,2.0)); s.sell('a',4)
assert s.get('a').qty==6"; ck $? "sell decreases qty"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore(); s.add(Item('a','Apple',3,2.0))
try: s.sell('a',9); raise SystemExit(1)
except ValueError: pass"; ck $? "sell beyond qty raises ValueError"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore()
s.add(Item('a','A',2,1.0)); s.add(Item('b','B',9,1.0)); s.add(Item('c','C',4,1.0))
assert s.low_stock(5)==['a','c'], s.low_stock(5)"; ck $? "low_stock returns correct skus in order"
pchk "from models import Item
from store import InventoryStore
s=InventoryStore(); s.add(Item('a','A',10,2.5)); s.add(Item('b','B',2,3.0))
assert s.total_value()==31.0, s.total_value()"; ck $? "total_value correct"
pchk "from models import Item
from store import InventoryStore
from report import summary
s=InventoryStore(); s.add(Item('a','A',2,1.0)); s.add(Item('b','B',10,2.0))
r=summary(s)
assert r=={'items':2,'total_value':22.0,'low_stock':['a']}, r"; ck $? "report.summary correct dict"
python3 -m pytest -q -p no:cacheprovider test_inventory.py >/dev/null 2>&1; ck $? "pytest test_inventory.py passes"

if [ "$pass" -eq "$total" ]; then pass_json true "$pass/$total checks"; else pass_json false "$pass/$total checks; first failure: $first"; fi
