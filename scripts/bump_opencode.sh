#!/usr/bin/env bash
# Build an updated, maintained LocalCode fork; fail if it has not absorbed dev.
# This never rebases or pushes the fork and never auto-merges a candidate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:-localcode}"
WORK="${LOCALCODE_UI_BUILD_DIR:-/tmp/localcode-ui-bump}"
FORK="${LOCALCODE_UI_FORK_REPO:-https://github.com/mjwsolo/opencode.git}"
UPSTREAM="${LOCALCODE_OPENCODE_UPSTREAM_REPO:-https://github.com/anomalyco/opencode.git}"
mkdir -p "$WORK"
if [ ! -d "$WORK/opencode/.git" ]; then git clone --no-checkout "$FORK" "$WORK/opencode"; fi
git -C "$WORK/opencode" fetch origin "$REF"
CANDIDATE="$(git -C "$WORK/opencode" rev-parse FETCH_HEAD)"
git -C "$WORK/opencode" fetch "$UPSTREAM" dev
UPSTREAM_SHA="$(git -C "$WORK/opencode" rev-parse FETCH_HEAD)"
if ! git -C "$WORK/opencode" merge-base --is-ancestor "$UPSTREAM_SHA" "$CANDIDATE"; then
  echo "Fork $CANDIDATE has not absorbed upstream dev $UPSTREAM_SHA. Merge and verify the fork first." >&2
  exit 20
fi
OLD="$(tr -d '[:space:]' < "$ROOT/src/localcode/ui/FORK_COMMIT")"
if [ "$OLD" = "$CANDIDATE" ]; then echo "OpenCode fork is already current"; exit 10; fi
# Preserve the old pin if compilation fails.
printf '%s\n' "$CANDIDATE" > "$ROOT/src/localcode/ui/FORK_COMMIT"
if ! LOCALCODE_UI_BUILD_DIR="$WORK" bash "$ROOT/scripts/build_ui_binary.sh"; then
  printf '%s\n' "$OLD" > "$ROOT/src/localcode/ui/FORK_COMMIT"
  exit 50
fi
printf '%s\n' "$UPSTREAM_SHA" > "$ROOT/src/localcode/ui/UPSTREAM_COMMIT"
echo "OpenCode candidate built; run the strict local gate before promotion"
