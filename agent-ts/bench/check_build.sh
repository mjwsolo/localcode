#!/usr/bin/env bash
# Objective acceptance test. The agent never sees this.
# Exits 0 only if the built CLI behaves to spec. Lenient on whitespace/extra output.
cd "$1" || exit 1
rm -f tasks.json
P="python3 -m taskcli"
$P add "buy milk"  >/dev/null 2>&1 || exit 1
$P add "walk dog"  >/dev/null 2>&1 || exit 1
L1=$($P list 2>/dev/null) || exit 1
echo "$L1" | grep -q "buy milk"  || exit 1
echo "$L1" | grep -q "walk dog"  || exit 1
echo "$L1" | grep -qE "^ *1 *\[ *\] *buy milk" || exit 1
$P done 1 >/dev/null 2>&1 || exit 1
L2=$($P list 2>/dev/null) || exit 1
echo "$L2" | grep -qiE "^ *1 *\[ *x *\] *buy milk" || exit 1
$P rm 2 >/dev/null 2>&1 || exit 1
L3=$($P list 2>/dev/null) || exit 1
echo "$L3" | grep -q "walk dog" && exit 1   # must be gone
[ -f README.md ] || exit 1
[ -f tasks.json ] || exit 1
exit 0
