#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
mkdir -p .eval && python3 - <<'PY'
import json
json.dump({"steps": 30, "tool_calls": 28, "compactions": 1, "timed_out": False}, open(".eval/metrics.json", "w"))
for i in range(1, 15): open(f".eval/final_turn{i}.txt", "w").write("done")
open(".eval/final_turn14.txt", "w").write("PELICAN")
def doc(path, pairs):
    s = open(path).read()
    for fn, d in pairs:
        head, sep, rest = s.partition(f"def {fn}(")
        line, nl, tail = rest.partition("\n")
        s = head + sep + line + nl + f'    """{d}"""' + nl + tail
    open(path, "w").write(s)
doc("app/calc.py", [("add", "Return a plus b."), ("div", "Return a divided by b.")])
doc("app/fmt.py", [("money", "Format x as dollars.")])
doc("app/text.py", [("title", "Title-case s.")])
doc("app/geo.py", [("area", "Return w times h."), ("perimeter", "Return the perimeter.")])
doc("app/units.py", [("km_to_miles", "Convert km to miles."), ("c_to_f", "Convert Celsius to Fahrenheit.")])
PY
