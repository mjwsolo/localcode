from app.report import report


def test_report():
    out = report([{'name': 'a', 'qty': 2, 'price': 3}, {'name': 'b', 'qty': 1, 'price': 5}])
    assert out.splitlines() == ['a: 2 x 3', 'b: 1 x 5', 'TOTAL 11']
