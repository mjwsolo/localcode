#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > app/text.py <<'PY'
def title(s):
    return s.title()


def initials(name):
    return ''.join(p[0] for p in name.split() if p)
PY
