#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
sed -i.bak 's/return a - b/return a + b/' app/calc.py && rm -f app/calc.py.bak
