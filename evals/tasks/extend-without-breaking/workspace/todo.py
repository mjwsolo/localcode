class TodoList:
    """A tiny in-memory todo list. Items are dicts: {"text": str, "done": bool}."""

    def __init__(self):
        self._items = []

    def add(self, text):
        self._items.append({"text": text, "done": False})

    def complete(self, index):
        self._items[index]["done"] = True

    def all(self):
        return list(self._items)
