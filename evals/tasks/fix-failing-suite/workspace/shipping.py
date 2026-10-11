"""Order/shipping math. Several bugs lurk here - the test suite is failing."""


def shipping_cost(weight_kg, distance_km):
    # BUG: the flat base fee of 5 is missing.
    return 2 * weight_kg + 0.5 * distance_km


def apply_discount(price, percent):
    # BUG: treats percent as an absolute amount instead of a percentage.
    return price - percent


def cart_total(prices):
    total = 0
    # BUG: skips the first item.
    for p in prices[1:]:
        total += p
    return total
