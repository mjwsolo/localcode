import pytest
from orders import process_order


SAMPLE = {
    "customer": "Alice",
    "items": [
        {"sku": "A1", "qty": 2, "price": 10.0},
        {"sku": "B2", "qty": 1, "price": 5.0},
    ],
}


def test_happy_path():
    out = process_order(SAMPLE)
    assert "Alice" in out
    assert "TOTAL: 25.00" in out


def test_discount():
    order = {**SAMPLE, "discount": 0.10}
    out = process_order(order)
    assert "TOTAL: 22.50" in out


def test_missing_items():
    with pytest.raises(ValueError):
        process_order({"customer": "Bob", "items": []})


def test_missing_customer():
    with pytest.raises(ValueError):
        process_order({"items": SAMPLE["items"]})


def test_bad_qty():
    bad = {"customer": "Bob", "items": [{"sku": "X", "qty": 0, "price": 5}]}
    with pytest.raises(ValueError):
        process_order(bad)
