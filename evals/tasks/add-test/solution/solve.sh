#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > tests/test_div_zero.py <<'PY'
import pytest
from app.calc import div


def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        div(1, 0)
PY
