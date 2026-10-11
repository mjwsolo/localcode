from shipping import apply_discount, cart_total, shipping_cost


def test_shipping_cost():
    assert shipping_cost(2, 10) == 14  # 5 base + 2*2 + 0.5*10


def test_shipping_cost_zero():
    assert shipping_cost(0, 0) == 5


def test_apply_discount():
    assert apply_discount(200, 10) == 180


def test_apply_discount_quarter():
    assert apply_discount(80, 25) == 60


def test_cart_total():
    assert cart_total([10, 20, 30]) == 60


def test_cart_total_single():
    assert cart_total([42]) == 42
