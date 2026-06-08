#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: verify.sh <workdir>}"
cd "$WORKDIR"
[ -f test_parser.py ] || { echo "FAIL: test_parser.py missing"; exit 2; }
EXPECTED=$(cat .parser.sha256)
ACTUAL=$(shasum -a 256 parser.py | awk '{print $1}')
[ "$EXPECTED" = "$ACTUAL" ] || { echo "FAIL: parser.py was modified"; exit 3; }
COUNT=$(grep -cE "^def test_" test_parser.py)
[ "$COUNT" -ge 5 ] || { echo "FAIL: need ≥5 test functions, got $COUNT"; exit 4; }
python -m pytest test_parser.py -q 2>&1 | tail -3
echo "PASS"
