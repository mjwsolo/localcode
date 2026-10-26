#!/usr/bin/env bash
# Terminal screen snapshot tests for the LocalCode UI. Runs the real
# localcode-ui in a pseudo-terminal against a deterministic mock model server
# (no model, no llama-server) and compares each journey's rendered screen with
# the text snapshots in tests/e2e/__snapshots__.
#   scripts/ui_e2e.sh                     run
#   scripts/ui_e2e.sh --update-snapshots  accept the current screens as baselines
#   scripts/ui_e2e.sh --png               also render each journey's final screen to tests/e2e/png (for humans)
#   scripts/ui_e2e.sh --installed PY      test the package installed in the venv whose interpreter is PY
#                                         (its bundled localcode-ui and Python modules, not this checkout)
# Needs: node >= 20, python3, the bundled binary (arm64 macOS).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/tests/e2e"
[ -d node_modules/@microsoft/tui-test ] || npm ci --no-audit --no-fund
while [ $# -gt 0 ]; do
  a="$1"; shift
  case "$a" in
    --update-snapshots) export E2E_UPDATE=1 ;;
    --png) export E2E_PNG=1 ;;
    --installed)
      [ $# -gt 0 ] || { echo "--installed needs the path of a python interpreter"; exit 2; }
      export E2E_INSTALLED_PY="$1"; shift ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
if [ -z "${E2E_INSTALLED_PY:-}" ]; then
  [ -x "${LOCALCODE_UI_BIN:-$ROOT/src/localcode/bin/localcode-ui}" ] || { echo "UI binary missing (scripts/build_ui_binary.sh)"; exit 1; }
fi
rm -rf snapshot-diffs
exec node --test --test-concurrency=1 journeys.test.mjs
