#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > models.py <<'PY'
class Item:
    """One stocked product."""

    def __init__(self, sku, name, qty, price):
        self.sku = sku
        self.name = name
        self.qty = qty
        self.price = price

    def __repr__(self):
        return f"Item({self.sku!r}, {self.name!r}, {self.qty!r}, {self.price!r})"
PY
cat > store.py <<'PY'
from models import Item


class InventoryStore:
    """Items keyed by sku, kept in insertion order."""

    def __init__(self):
        self._items: dict[str, Item] = {}

    def __len__(self):
        return len(self._items)

    def add(self, item: Item) -> None:
        self._items[item.sku] = item

    def get(self, sku) -> Item:
        return self._items[sku]

    def restock(self, sku, n) -> None:
        self.get(sku).qty += n

    def sell(self, sku, n) -> None:
        item = self.get(sku)
        if n > item.qty:
            raise ValueError(f"cannot sell {n} of {sku}: only {item.qty} in stock")
        item.qty -= n

    def low_stock(self, threshold) -> list:
        return [sku for sku, item in self._items.items() if item.qty < threshold]

    def total_value(self):
        return sum(item.qty * item.price for item in self._items.values())
PY
cat > report.py <<'PY'
from store import InventoryStore

LOW_STOCK_THRESHOLD = 5


def summary(store: InventoryStore) -> dict:
    return {
        "items": len(store),
        "total_value": store.total_value(),
        "low_stock": store.low_stock(LOW_STOCK_THRESHOLD),
    }
PY
cat > test_inventory.py <<'PY'
import pytest

from models import Item
from report import summary
from store import InventoryStore


@pytest.fixture
def store():
    s = InventoryStore()
    s.add(Item("a", "Apple", 2, 1.0))
    s.add(Item("b", "Banana", 10, 2.0))
    s.add(Item("c", "Cherry", 4, 0.5))
    return s


def test_add_and_get(store):
    item = store.get("b")
    assert (item.sku, item.name, item.qty, item.price) == ("b", "Banana", 10, 2.0)


def test_restock_and_sell(store):
    store.restock("a", 5)
    store.sell("b", 4)
    assert store.get("a").qty == 7 and store.get("b").qty == 6


def test_sell_more_than_stock_raises(store):
    with pytest.raises(ValueError):
        store.sell("a", 3)
    assert store.get("a").qty == 2


def test_low_stock_is_strict_and_ordered(store):
    assert store.low_stock(5) == ["a", "c"]
    assert store.low_stock(2) == []


def test_total_value(store):
    assert store.total_value() == 24.0


def test_summary(store):
    assert summary(store) == {"items": 3, "total_value": 24.0, "low_stock": ["a", "c"]}
PY
