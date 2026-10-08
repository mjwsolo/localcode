"""Discover the authenticated local model API for use by other tools."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error

from . import run_dir
from .auth import read_auth_file, control_headers
from .ports import find_running, save_port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="localcode api",
        description="Show the OpenAI compatible API of a running LocalCode session.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine readable connection details")
    parser.add_argument("--show-key", action="store_true", help="Include the server API key in the output")
    parser.add_argument("--port", help="Change the live API port, or save it for startup (1024–65535 or auto)")
    args = parser.parse_args(argv)
    if args.port is not None:
        try:
            port = None if args.port == "auto" else int(args.port)
            if port is not None and not 1024 <= port <= 65535:
                raise ValueError("Port must be between 1024 and 65535, or auto")
            running = find_running()
            if running is None:
                save_port(port)
                print("API port saved for the next launch; no running LocalCode service was found.")
                return 0
            auth = read_auth_file(run_dir())
            if not auth:
                raise ValueError("Cannot authenticate to the running service; port unchanged")
            request = urllib.request.Request(
                f"http://127.0.0.1:{running[0]}/server/port",
                data=json.dumps({"port": port}).encode(),
                headers={**control_headers(auth[0]), "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.load(response)
            print(f"API moved to {result['base_url']}. Update external clients to this address; LocalCode stays connected.")
            return 0
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get("error", str(exc))
            except (ValueError, OSError):
                message = str(exc)
            print(f"localcode: {message}", file=sys.stderr)
            return 1
        except (ValueError, OSError, KeyError) as exc:
            print(f"localcode: {exc}. Run `localcode api` to check the current address.", file=sys.stderr)
            return 1

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
