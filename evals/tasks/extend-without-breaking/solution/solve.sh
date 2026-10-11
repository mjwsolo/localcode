#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > todo.py <<'PY'
class TodoList:
    """A tiny in-memory todo list. Items are dicts: {"text": str, "done": bool}."""

    def __init__(self):
        self._items = []

    def add(self, text):
        self._items.append({"text": text, "done": False})

    def complete(self, index):
        self._items[index]["done"] = True

    def remove(self, index):
        del self._items[index]

    def pending(self):
        return [item["text"] for item in self._items if not item["done"]]

    def all(self):
        return list(self._items)
PY
cat > test_todo_ext.py <<'PY'
import pytest

from todo import TodoList


def make(*texts):
    t = TodoList()
    for text in texts:
        t.add(text)
    return t


def test_remove_deletes_the_item_at_index():
    t = make("a", "b", "c")
    t.remove(1)
    assert [i["text"] for i in t.all()] == ["a", "c"]


def test_remove_out_of_range_raises():
    with pytest.raises(IndexError):
        make("a").remove(5)


def test_pending_lists_undone_texts_in_order():
    t = make("a", "b", "c")
    t.complete(1)
    assert t.pending() == ["a", "c"]


def test_pending_empty_list():
    assert TodoList().pending() == []
PY
