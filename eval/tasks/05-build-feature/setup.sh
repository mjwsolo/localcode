#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: setup.sh <workdir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORKDIR"
cp "$HERE/fixtures/cli.py" "$WORKDIR/cli.py"
cp "$HERE/fixtures/test_cli.py" "$WORKDIR/test_cli.py"
echo "ready"
