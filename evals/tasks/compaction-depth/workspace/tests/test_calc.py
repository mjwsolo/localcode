from app.calc import add, div


def test_add():
    assert add(1, 2) == 3


def test_div():
    assert div(6, 3) == 2
