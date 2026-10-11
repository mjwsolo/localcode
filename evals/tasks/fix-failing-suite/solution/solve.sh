#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > shipping.py <<'PY'
"""Order/shipping math."""

BASE_FEE = 5


def shipping_cost(weight_kg, distance_km):
    return BASE_FEE + 2 * weight_kg + 0.5 * distance_km


def apply_discount(price, percent):
    return price - price * percent / 100


def cart_total(prices):
    total = 0
    for p in prices:
        total += p
    return total
PY
