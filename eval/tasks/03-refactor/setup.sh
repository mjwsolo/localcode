#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: setup.sh <workdir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORKDIR"
cp "$HERE/fixtures/orders.py" "$WORKDIR/orders.py"
cp "$HERE/fixtures/test_orders.py" "$WORKDIR/test_orders.py"
# Snapshot test file hash so verify.sh can confirm it wasn't modified.
shasum -a 256 "$HERE/fixtures/test_orders.py" | awk '{print $1}' > "$WORKDIR/.test_orders.sha256"
echo "ready"
