#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/.*)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/.*)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

[ -s .eval/ui_errors.txt ] && { pass_json false "runtime errors: $(head -c 160 .eval/ui_errors.txt)"; exit 0; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import ast, json, os, glob
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
assert not m.get("timed_out"), "session timed out"
assert (m.get("compactions") or 0) >= 1, "no compaction happened: run with LOCALCODE_OVERRIDE_NCTX=16384 or a smaller context"
turns=sorted(glob.glob(".eval/final_turn*.txt"), key=lambda p: int(p.split("final_turn")[1].split(".")[0]))
assert len(turns) >= 14, f"only {len(turns)} turns completed"
last=open(turns[-1]).read().upper()
assert "PELICAN" in last, f"lost the word of the day after compaction: {last[:80]!r}"
for f, fn in [("app/calc.py","add"),("app/calc.py","div"),("app/fmt.py","money"),("app/text.py","title"),("app/geo.py","area"),("app/geo.py","perimeter"),("app/units.py","km_to_miles"),("app/units.py","c_to_f")]:
    t=ast.parse(open(f).read()); fns={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    assert ast.get_docstring(fns[fn]), f"{f}:{fn} has no docstring"
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "crossed a compaction, kept the early fact, all eight edits landed, tests green"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
