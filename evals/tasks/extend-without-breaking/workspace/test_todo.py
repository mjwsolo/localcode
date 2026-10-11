from todo import TodoList


def test_add():
    t = TodoList()
    t.add("a")
    assert len(t.all()) == 1
    assert t.all()[0]["text"] == "a"
    assert t.all()[0]["done"] is False


def test_complete():
    t = TodoList()
    t.add("a")
    t.complete(0)
    assert t.all()[0]["done"] is True
