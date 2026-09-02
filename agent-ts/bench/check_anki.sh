#!/usr/bin/env bash
# Objective acceptance for the flashcard build. The agent never sees this.
cd "$1" || exit 1
[ -f package.json ] || exit 1
[ -f src/srs.ts ]   || exit 1
grep -q "vite" package.json || exit 1
# must actually build
npm install --silent >/dev/null 2>&1 || exit 1
npm run build >/dev/null 2>&1 || exit 1
[ -d dist ] || exit 1
# srs module must export schedule and implement SM-2 (ease factor + interval)
grep -qE "export (function|const) schedule" src/srs.ts || exit 1
grep -qiE "ease" src/srs.ts     || exit 1
grep -qiE "interval" src/srs.ts || exit 1
# core features must be present somewhere in src
grep -rqiE "indexeddb|idb" src/ || exit 1
grep -rqE "\{\{c[0-9]+::" src/ || grep -rqiE "cloze" src/ || exit 1
exit 0
