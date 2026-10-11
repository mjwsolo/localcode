#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > report.py <<'PY'
"""Aggregate a category,amount CSV."""
import csv


def totals(path):
    """Map each category to the sum of its amounts; {} when there are no data rows."""
    result = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            category = (row.get("category") or "").strip()
            if not category:
                continue
            result[category] = result.get(category, 0) + int(row["amount"])
    return result


def top_category(path):
    """The category with the highest total, or None when there are no data rows."""
    sums = totals(path)
    if not sums:
        return None
    return max(sums, key=sums.get)
PY
cat > test_report.py <<'PY'
from report import top_category, totals


def write(tmp_path, body):
    path = tmp_path / "data.csv"
    path.write_text("category,amount\n" + body)
    return str(path)


def test_totals_sums_per_category(tmp_path):
    assert totals(write(tmp_path, "a,1\nb,5\na,2\n")) == {"a": 3, "b": 5}


def test_top_category(tmp_path):
    assert top_category(write(tmp_path, "a,1\nb,5\na,2\n")) == "b"


def test_header_only_file(tmp_path):
    path = write(tmp_path, "")
    assert totals(path) == {}
    assert top_category(path) is None
PY
