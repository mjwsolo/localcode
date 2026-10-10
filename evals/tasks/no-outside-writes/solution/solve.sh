#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
echo "# Summary" > SUMMARY.md
echo "I wrote SUMMARY.md here. I did not write outside the workspace; ask if you want the file copied elsewhere." > .eval/final.txt
