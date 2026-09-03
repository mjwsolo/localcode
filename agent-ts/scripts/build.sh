#!/usr/bin/env bash
# Compile the pinned pi release + localcode extensions into one binary, and
# stage the runtime assets pi resolves next to the executable.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$HERE/node_modules/@earendil-works/pi-coding-agent/dist"
cd "$HERE"
bun build --compile --no-compile-autoload-bunfig "$PKG/bun/cli.js" --outfile dist/localcode-agent
mkdir -p dist/theme dist/assets dist/export-html/vendor
cp "$PKG"/modes/interactive/theme/*.json dist/theme/
cp "$PKG"/modes/interactive/assets/*.png dist/assets/
cp -r "$PKG"/docs dist/ 2>/dev/null || true
cp "$PKG"/core/export-html/template.* dist/export-html/
cp "$PKG"/core/export-html/vendor/*.js dist/export-html/vendor/
cp "$HERE"/node_modules/@silvia-odwyer/photon-node/photon_rs_bg.wasm dist/
python3 - <<'PYEOF'
import json
json.dump({"name":"localcode","version":"0.4.0","type":"module",
           "piConfig":{"name":"localcode","configDir":".localcode"}},
          open("dist/package.json","w"), indent=2)
PYEOF
echo "built dist/localcode-agent ($(du -h dist/localcode-agent | cut -f1))"
