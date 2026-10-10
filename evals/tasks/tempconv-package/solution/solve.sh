#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
mkdir -p src/tempconv tests
cat > pyproject.toml <<'T'
[project]
name = "tempconv"
version = "0.1.0"
[tool.pytest.ini_options]
pythonpath = ["src"]
T
cat > src/tempconv/__init__.py <<'PY'
ABS_ZERO_C = -273.15


def _check_c(c):
    if c < ABS_ZERO_C:
        raise ValueError("below absolute zero")


def c_to_f(c):
    _check_c(c); return c * 9 / 5 + 32


def f_to_c(f):
    c = (f - 32) * 5 / 9; _check_c(c); return c


def c_to_k(c):
    _check_c(c); return c + 273.15


def k_to_c(k):
    if k < 0:
        raise ValueError("below absolute zero")
    return k - 273.15
PY
cat > tests/test_tempconv.py <<'PY'
import pytest
from tempconv import c_to_f, f_to_c, c_to_k, k_to_c


def test_roundtrips():
    assert c_to_f(100) == 212 and f_to_c(212) == 100 and c_to_k(0) == 273.15 and k_to_c(273.15) == 0


def test_error():
    with pytest.raises(ValueError):
        c_to_k(-300)
PY
