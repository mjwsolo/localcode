def report(rows):
    total = 0
    for r in rows:
        total += r['qty'] * r['price']
    lines = []
    for r in rows:
        lines.append(f"{r['name']}: {r['qty']} x {r['price']}")
    lines.append(f'TOTAL {total}')
    return '\n'.join(lines)
