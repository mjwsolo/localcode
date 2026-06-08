#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: verify.sh <workdir>}"
cd "$WORKDIR"
[ -f orders.py ] || { echo "FAIL: orders.py missing"; exit 2; }
EXPECTED=$(cat .test_orders.sha256)
ACTUAL=$(shasum -a 256 test_orders.py | awk '{print $1}')
[ "$EXPECTED" = "$ACTUAL" ] || { echo "FAIL: test_orders.py was modified (cheating)"; exit 3; }
for fn in validate_order calculate_total format_receipt process_order; do
  grep -qE "^def[[:space:]]+$fn\b|^def[[:space:]]+$fn\(" orders.py || { echo "FAIL: orders.py missing def $fn"; exit 4; }
done
python -m pytest test_orders.py -q 2>&1 | tail -3
echo "PASS"
