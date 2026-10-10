#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
echo '{"steps": 7, "tool_calls": 5, "reread_tokens": [5800, 120, 90, 140, 130]}' > .eval/metrics.json
