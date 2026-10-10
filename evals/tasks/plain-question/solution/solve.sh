#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
echo '{"steps": 1, "tool_calls": 0}' > .eval/metrics.json
echo "A monad is a wrapper for computations that lets you chain operations while handling context like state or failure." > .eval/final.txt
