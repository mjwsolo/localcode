#!/usr/bin/env bash
# Rebuild the bundled UI binary (src/localcode/bin/localcode-ui) from the exact
# fork commit in src/localcode/ui/FORK_COMMIT, in a neutral directory so no
# developer path is embedded. Prints the sha256; CI compares it.
#
#   scripts/build_ui_binary.sh            # build + install into src/localcode/bin
#   scripts/build_ui_binary.sh --check    # build only, compare sha256 with the bundled one
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(tr -d '[:space:]' < "$ROOT/src/localcode/ui/FORK_COMMIT")"
REPO="${LOCALCODE_UI_FORK_REPO:-https://github.com/mjwsolo/opencode.git}"
WORK="${LOCALCODE_UI_BUILD_DIR:-/tmp/localcode-ui-build}"
command -v bun >/dev/null || { echo "bun is required (https://bun.sh)"; exit 1; }
mkdir -p "$WORK"
if [ ! -d "$WORK/opencode/.git" ]; then git clone -q "$REPO" "$WORK/opencode"; fi
git -C "$WORK/opencode" fetch -q origin
git -C "$WORK/opencode" checkout -q "$COMMIT"
( cd "$WORK/opencode" && bun install --frozen-lockfile >/dev/null )
# The runtime shows this in its footer; keep it equal to the wheel version (semver form).
PYVER="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
export OPENCODE_VERSION="$(echo "$PYVER" | sed -E 's/([0-9])(a|b|rc)([0-9]+)$/\1-\2\3/')"
( cd "$WORK/opencode/packages/opencode" && bun run script/build.ts --single --skip-install --skip-embed-web-ui >/dev/null )
OUT="$(ls "$WORK"/opencode/packages/opencode/dist/*/bin/localcode | head -1)"
file "$OUT" | grep -q 'arm64' || { echo "unexpected architecture: $(file "$OUT")"; exit 1; }
if strings "$OUT" | grep -q "^/Users/[^r]"; then echo "developer path embedded in binary; build from a neutral directory"; exit 1; fi
SHA="$(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo "built $OUT"; echo "sha256 $SHA (fork $COMMIT)"
if [ "${1:-}" = "--check" ]; then
  HAVE="$(shasum -a 256 "$ROOT/src/localcode/bin/localcode-ui" | cut -d' ' -f1)"
  [ "$SHA" = "$HAVE" ] && echo "bundled binary matches" || { echo "bundled binary sha256 $HAVE differs"; exit 1; }
  exit 0
fi
cp "$OUT" "$ROOT/src/localcode/bin/localcode-ui"; chmod 755 "$ROOT/src/localcode/bin/localcode-ui"
echo "installed into src/localcode/bin/localcode-ui"
