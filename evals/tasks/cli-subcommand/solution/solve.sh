#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
python3 - <<'PY'
p="app/cli.py"; s=open(p).read()
s=s.replace("    g = sub.add_parser('greet'); g.add_argument('name')\n", "    g = sub.add_parser('greet'); g.add_argument('name')\n    s = sub.add_parser('sum'); s.add_argument('nums', nargs='+', type=int)\n")
s=s.replace("        return 0\n    return 1\n", "        return 0\n    if a.cmd == 'sum':\n        print(sum(a.nums))\n        return 0\n    return 1\n")
open(p,"w").write(s)
open("tests/test_cli_sum.py","w").write("from app.cli import main\n\n\ndef test_sum(capsys):\n    assert main(['sum', '1', '2', '3']) == 0\n    assert capsys.readouterr().out.strip() == '6'\n")
PY
