#!/usr/bin/env bash
# Compile the pinned pi release + localcode extensions into one binary, and
# stage the runtime assets pi resolves next to the executable.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$HERE/node_modules/@earendil-works/pi-coding-agent/dist"
cd "$HERE"
# --- debrand: rewrite upstream product strings before compiling -------------
python3 - "$HERE/node_modules/@earendil-works" <<'PYEOF'
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
subs = [
  ("Pi can explain its own features and look up its docs. Ask it how to use or extend Pi.",
   "localcode runs a local model on your Mac. Ask it to read, edit and test your code."),
  ("PI_CODING_AGENT_DIR", "LOCALCODE_CODING_AGENT_DIR"),
  ("PI_CODING_AGENT", "LOCALCODE_CODING_AGENT"),
  ('".pi","agent"', '".localcode","agent"'),
  ('".pi", "agent"', '".localcode", "agent"'),
  ("~/.pi", "~/.localcode"),
  ("PI_OFFLINE", "LOCALCODE_OFFLINE"),
  ("PI_EXPERIMENTAL", "LOCALCODE_EXPERIMENTAL"),
  ("PI_HARDWARE_CURSOR", "LOCALCODE_HARDWARE_CURSOR"),
  ("PI_CLEAR_ON_SHRINK", "LOCALCODE_CLEAR_ON_SHRINK"),
  ("PI_BUNDLED_NODE", "LOCALCODE_BUNDLED_NODE"),
]
n = 0
for f in list(root.rglob("*.js")) + list(root.rglob("*.map")):
    try: t = f.read_text()
    except Exception: continue
    o = t
    for a, b in subs: t = t.replace(a, b)
    if t != o:
        f.write_text(t); n += 1
print(f"debranded {n} bundled files")
PYEOF

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
