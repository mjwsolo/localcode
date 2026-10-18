#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
python3 - <<'PY'
import json
json.dump({"steps": 12, "tool_calls": 10, "compactions": 1, "timed_out": False}, open(".eval/metrics.json", "w"))
with open("app/report.py", "a") as f:
    f.write("""

def to_markdown(rows):
    keys = list(rows[0]) if rows else []
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    lines += ["| " + " | ".join(str(row.get(k, "")) for k in keys) + " |" for row in rows]
    return "\\n".join(lines)
""")
with open("tests/test_report.py", "a") as f:
    f.write("""

def test_to_markdown():
    from app.report import to_markdown
    assert to_markdown([{"a": 1}]) == "| a |\\n| --- |\\n| 1 |"
""")
PY
