#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
mkdir -p inventory tests
cat > inventory/__init__.py <<'PY'
PY
cat > inventory/models.py <<'PY'
from dataclasses import dataclass


@dataclass
class Item:
    name: str
    qty: int
    price: float
PY
cat > inventory/store.py <<'PY'
from .models import Item


class Store:
    def __init__(self):
        self.items: list[Item] = []

    def add(self, item: Item) -> None:
        self.items.append(item)

    def total_value(self) -> float:
        return sum(i.qty * i.price for i in self.items)
PY
cat > inventory/report.py <<'PY'
from .store import Store


def render(store: Store) -> str:
    lines = [f"{i.name}: {i.qty} x {i.price}" for i in store.items]
    lines.append(f"TOTAL {store.total_value()}")
    return "\n".join(lines)
PY
cat > tests/test_inventory.py <<'PY'
from inventory.models import Item
from inventory.store import Store
from inventory.report import render


def test_total_and_render():
    s = Store(); s.add(Item("a", 2, 3.0))
    assert s.total_value() == 6.0 and render(s).endswith("TOTAL 6.0")
PY
