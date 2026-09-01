#!/usr/bin/env bash
# $1 = dir, $2 = task name
set -e
d="$1"; rm -rf "$d"; mkdir -p "$d"
case "$2" in
retry)
  printf 'def retry(times):\n    pass\n' > "$d/retry.py"
  cat > "$d/test_retry.py" <<'PY'
from retry import retry
calls = []
@retry(times=3)
def flaky():
    calls.append(1)
    if len(calls) < 3: raise ValueError("boom")
    return "ok"
def test_retries_until_success():
    assert flaky() == "ok"
    assert len(calls) == 3
PY
  ;;
parser)
  cat > "$d/csvstat.py" <<'PY'
def column_stats(rows, column):
    """Return {'min','max','mean'} for a numeric column across rows (list of dicts).
    Rows whose column value is missing or non-numeric are skipped.
    Raise ValueError if no row has a usable value."""
    raise NotImplementedError
PY
  cat > "$d/test_csvstat.py" <<'PY'
import pytest
from csvstat import column_stats
def test_basic():
    rows = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
    assert column_stats(rows, "a") == {"min": 1.0, "max": 3.0, "mean": 2.0}
def test_skips_bad():
    rows = [{"a": "1"}, {"a": "oops"}, {"a": None}, {"a": "5"}]
    s = column_stats(rows, "a")
    assert s["min"] == 1.0 and s["max"] == 5.0 and s["mean"] == 3.0
def test_missing_column():
    rows = [{"b": "1"}]
    with pytest.raises(ValueError):
        column_stats(rows, "a")
PY
  ;;
esac
