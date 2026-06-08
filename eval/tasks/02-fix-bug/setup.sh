#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: setup.sh <workdir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORKDIR"
cp "$HERE/fixtures/calculator.py" "$WORKDIR/calculator.py"
cp "$HERE/fixtures/test_calculator.py" "$WORKDIR/test_calculator.py"
echo "ready"
