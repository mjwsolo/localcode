#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > app/report.py <<'PY'
def total(rows):
    return sum(r['qty'] * r['price'] for r in rows)


def format_line(row):
    return f"{row['name']}: {row['qty']} x {row['price']}"


def report(rows):
    lines = [format_line(r) for r in rows]
    lines.append(f'TOTAL {total(rows)}')
    return '\n'.join(lines)
PY
