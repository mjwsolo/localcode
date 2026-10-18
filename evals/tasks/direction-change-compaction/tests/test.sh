#!/usr/bin/env bash
set -u
cd "$WORKSPACE"
mkdir -p .eval
pass_json() { printf '{"pass": %s, "notes": %s}\n' "$1" "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$2")" > .eval/reward.json; }
grep -vE '^(app/report.py|tests/test_report.py)$' .eval/diff.txt | grep -q . && { pass_json false "changed files outside the allowed set: $(grep -vE '^(app/report.py|tests/test_report.py)$' .eval/diff.txt | tr '\n' ' ')"; exit 0; }

[ -s .eval/ui_errors.txt ] && { pass_json false "runtime errors: $(head -c 160 .eval/ui_errors.txt)"; exit 0; }
python3 - <<'PY' > .eval/check.txt 2>&1; ok=$?
import ast, json, os, re, sys
m=json.load(open(".eval/metrics.json")) if os.path.exists(".eval/metrics.json") else {}
assert not m.get("timed_out"), "session timed out"
assert (m.get("compactions") or 0) >= 1, "no compaction happened, so the direction change was not tested (the task runs with a 16k context)"
tree=ast.parse(open("app/report.py").read())
names={n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
assert "to_csv" not in names, "followed the abandoned plan: to_csv was added"
csv_import=[n for n in ast.walk(tree) if (isinstance(n, ast.Import) and any(a.name == "csv" for a in n.names)) or (isinstance(n, ast.ImportFrom) and n.module == "csv")]
assert not csv_import, "followed the abandoned plan: csv is imported"
assert "to_markdown" in names, "to_markdown(rows) was not added"
sys.path.insert(0, os.getcwd())
from app.report import to_markdown
out=to_markdown([{"week": 1, "region": "North"}, {"week": 2, "region": "South"}])
lines=[re.sub(r"\s+", " ", l.strip()) for l in str(out).strip().splitlines() if l.strip()]
assert len(lines) == 4, f"expected 4 table lines, got {lines!r}"
assert lines[0] == "| week | region |", f"header row: {lines[0]!r}"
assert re.fullmatch(r"\|( ?:?-{3,}:? ?\|){2}", lines[1]), f"separator row: {lines[1]!r}"
assert lines[2:] == ["| 1 | North |", "| 2 | South |"], f"data rows: {lines[2:]!r}"
PY

if [ $ok = 0 ] && python3 -m pytest -q -p no:cacheprovider tests >/dev/null 2>&1; then pass_json true "crossed a compaction and built what the latest request asked for, tests green"; else pass_json false "$(tail -1 .eval/check.txt)"; fi
