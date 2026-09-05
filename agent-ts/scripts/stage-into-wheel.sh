#!/usr/bin/env bash
# Stage the built agent + its runtime assets into src/localcode/bin so the
# wheel ships them next to llama-server. Run after scripts/build.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/../src/localcode/bin/agent"
[ -f "$HERE/dist/localcode-agent" ] || { echo "build first: ./scripts/build.sh"; exit 1; }
# The binary resolves theme/assets/docs as SIBLINGS of the executable, so ship
# the whole dist layout under bin/agent/.
rm -rf "$DEST"; mkdir -p "$DEST"
cp "$HERE/dist/localcode-agent" "$DEST/"
for d in theme assets export-html docs; do [ -e "$HERE/dist/$d" ] && cp -r "$HERE/dist/$d" "$DEST/"; done
cp "$HERE/dist/package.json" "$DEST/" 2>/dev/null || true
cp "$HERE"/dist/*.wasm "$DEST/" 2>/dev/null || true
cp -r "$HERE/extensions" "$DEST/extensions"   # jiti loads TS at runtime
echo "staged: $(du -sh "$DEST" | cut -f1) -> src/localcode/bin/agent/"
