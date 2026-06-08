#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: setup.sh <workdir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORKDIR"
cp "$HERE/fixtures/parser.py" "$WORKDIR/parser.py"
shasum -a 256 "$HERE/fixtures/parser.py" | awk '{print $1}' > "$WORKDIR/.parser.sha256"
rm -f "$WORKDIR/test_parser.py"
echo "ready"
