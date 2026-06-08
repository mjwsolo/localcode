#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: verify.sh <workdir>}"
cd "$WORKDIR"
[ -f fizzbuzz.py ] || { echo "FAIL: fizzbuzz.py missing"; exit 2; }
[ -f test_fizzbuzz.py ] || { echo "FAIL: test_fizzbuzz.py missing"; exit 2; }
OUT=$(python fizzbuzz.py 2>&1) || { echo "FAIL: fizzbuzz.py crashed"; echo "$OUT"; exit 3; }
LINE3=$(echo "$OUT" | sed -n '3p')
LINE5=$(echo "$OUT" | sed -n '5p')
LINE15=$(echo "$OUT" | sed -n '15p')
[ "$LINE3" = "Fizz" ] || { echo "FAIL: line 3 expected 'Fizz' got '$LINE3'"; exit 4; }
[ "$LINE5" = "Buzz" ] || { echo "FAIL: line 5 expected 'Buzz' got '$LINE5'"; exit 4; }
[ "$LINE15" = "FizzBuzz" ] || { echo "FAIL: line 15 expected 'FizzBuzz' got '$LINE15'"; exit 4; }
grep -q "def test_" test_fizzbuzz.py || { echo "FAIL: no test_ function found"; exit 5; }
python -m pytest test_fizzbuzz.py -q 2>&1 | tail -3
echo "PASS"
