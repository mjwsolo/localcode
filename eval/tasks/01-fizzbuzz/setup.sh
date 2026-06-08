#!/usr/bin/env bash
# Idempotent: clears the working dir of any prior attempt, sets up nothing
# (FizzBuzz starts from scratch).
set -euo pipefail
WORKDIR="${1:?usage: setup.sh <workdir>}"
mkdir -p "$WORKDIR"
rm -f "$WORKDIR/fizzbuzz.py" "$WORKDIR/test_fizzbuzz.py"
echo "ready"
