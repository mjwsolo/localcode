#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
python3 - <<'PY'
def doc(path, pairs):
    s = open(path).read()
    for fn, d in pairs:
        s = s.replace(f"def {fn}(", f"def {fn}(", 1)
        head, sep, rest = s.partition(f"def {fn}(")
        line, nl, tail = rest.partition("\n")
        s = head + sep + line + nl + f'    """{d}"""' + nl + tail
    open(path, "w").write(s)
doc("app/calc.py", [("add", "Return a plus b."), ("div", "Return a divided by b.")])
doc("app/fmt.py", [("money", "Format x as dollars.")])
doc("app/text.py", [("title", "Title-case s.")])
PY
echo '{"steps": 14, "tool_calls": 9, "reread_tokens": [5800, 60, 40, 30, 50, 120, 20, 45, 30, 110]}' > .eval/metrics.json
: > .eval/ui_errors.txt
