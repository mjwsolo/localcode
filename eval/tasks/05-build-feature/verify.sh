#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:?usage: verify.sh <workdir>}"
cd "$WORKDIR"
[ -f cli.py ] || { echo "FAIL: cli.py missing"; exit 2; }
[ -f test_cli.py ] || { echo "FAIL: test_cli.py missing"; exit 2; }
grep -qE "^def[[:space:]]+count\b|@app\.command\(\)\s*\n?def[[:space:]]+count" cli.py || { echo "FAIL: cli.py missing 'count' command"; exit 3; }
# Create a sample file and test count works
SAMPLE=$(mktemp)
printf "one two three\nfour five\nsix\n" > "$SAMPLE"
OUT=$(python -c "from typer.testing import CliRunner; from cli import app; r=CliRunner().invoke(app, ['count', '--file', '$SAMPLE']); print(r.stdout); exit(r.exit_code)" 2>&1) || { echo "FAIL: count command crashed: $OUT"; exit 4; }
echo "$OUT" | grep -qE "lines:\s*3" || { echo "FAIL: expected 'lines: 3' in output, got: $OUT"; exit 5; }
echo "$OUT" | grep -qE "words:\s*6" || { echo "FAIL: expected 'words: 6' in output, got: $OUT"; exit 5; }
rm -f "$SAMPLE"
# Test missing file exits non-zero
python -c "from typer.testing import CliRunner; from cli import app; r=CliRunner().invoke(app, ['count', '--file', '/nonexistent/abc']); exit(0 if r.exit_code != 0 else 1)" || { echo "FAIL: missing file should have non-zero exit"; exit 6; }
python -m pytest test_cli.py -q 2>&1 | tail -3
echo "PASS"
