#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: verify.sh <workdir>}"
cd "$WORKDIR"
[ -f calculator.py ] || { echo "FAIL: calculator.py missing"; exit 2; }
RESULT=$(python -c "from calculator import divide; print(divide(10, 0))" 2>&1)
[ "$RESULT" = "None" ] || { echo "FAIL: divide(10,0) expected None, got: $RESULT"; exit 3; }
RESULT2=$(python -c "from calculator import divide; print(divide(10, 2))" 2>&1)
[ "$RESULT2" = "5.0" ] || [ "$RESULT2" = "5" ] || { echo "FAIL: divide(10,2) expected 5, got: $RESULT2"; exit 4; }
python -m pytest test_calculator.py -q 2>&1 | tail -3
grep -qE "def test_.*[Zz]ero|divide\(10, 0\) is None|divide\(10,0\) is None" test_calculator.py || { echo "FAIL: no regression test added for divide-by-zero"; exit 5; }
echo "PASS"
