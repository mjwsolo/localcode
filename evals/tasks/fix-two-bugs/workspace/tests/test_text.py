from app.text import title, initials


def test_title():
    assert title('hello world') == 'Hello World'


def test_initials_ignores_double_spaces():
    assert initials('Ada  Lovelace') == 'AL'
