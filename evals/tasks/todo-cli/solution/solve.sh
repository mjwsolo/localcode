#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > todo.py <<'PY'
"""A small persistent todo list. State lives in todos.json in the current directory."""
import argparse
import json
import os
import sys

STATE = "todos.json"


def load():
    if not os.path.exists(STATE):
        return []
    with open(STATE) as fh:
        return json.load(fh)


def save(tasks):
    with open(STATE, "w") as fh:
        json.dump(tasks, fh, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="todo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("text")
    sub.add_parser("list")
    sub.add_parser("done").add_argument("index", type=int)
    sub.add_parser("rm").add_argument("index", type=int)
    args = parser.parse_args(argv)

    tasks = load()
    if args.cmd == "add":
        tasks.append({"text": args.text, "done": False})
        save(tasks)
        return 0
    if args.cmd == "list":
        for i, t in enumerate(tasks):
            print(f"{i}: [{'x' if t['done'] else ' '}] {t['text']}")
        return 0
    if not 0 <= args.index < len(tasks):
        print(f"error: no task with index {args.index}", file=sys.stderr)
        return 1
    if args.cmd == "done":
        tasks[args.index]["done"] = True
    else:
        del tasks[args.index]
    save(tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
PY
