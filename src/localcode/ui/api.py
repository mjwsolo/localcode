"""Discover the authenticated local model API for use by other tools."""
from __future__ import annotations

import argparse
import json
import sys

from . import run_dir
from .auth import read_auth_file
from .ports import find_running


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localcode api",
        description="Show the OpenAI compatible API of a running LocalCode session.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine readable connection details")
    parser.add_argument("--show-key", action="store_true", help="Include the server API key in the output")
    args = parser.parse_args(argv)

    running = find_running()
    if running is None:
        print("localcode: no running model server; start `localcode` first", file=sys.stderr)
        return 1
    _, status = running
    auth = read_auth_file(run_dir())
    if auth is None or not auth[1]:
        print("localcode: the running server's API key is unavailable", file=sys.stderr)
        return 1
    port = int(status["port"])
    alias = status.get("current") or None
    info = {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "model": f"localcode/{alias}" if alias else None,
        "state": status.get("state"),
    }
    if args.show_key:
        info["api_key"] = auth[1]
    if args.json:
        print(json.dumps(info))
    else:
        print(f"Base URL: {info['base_url']}")
        print(f"Model: {info['model'] or '(none loaded)'}")
        print(f"State: {info['state'] or 'unknown'}")
        if args.show_key:
            print(f"API key: {info['api_key']}")
        else:
            print("Use --show-key to include the API key.")
    return 0
