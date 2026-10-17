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
# Upstream promoted its v2 rewrite to dev on 2026-10-10 (5caf1f9f6f), which
# removes packages/opencode that the fork and patches/ui build on. Until the
# fork is ported to v2, track the last pre-v2 commit instead of dev's head.
PRE_V2=055d95bb7e278c94baf06235a52cac79dd13ba67
if git -C "$WORK/opencode" merge-base --is-ancestor "$PRE_V2" "$UPSTREAM_SHA"; then
  UPSTREAM_SHA="$PRE_V2"
fi
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
