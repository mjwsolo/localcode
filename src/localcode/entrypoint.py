"""LocalCode launcher: `localcode [project]` starts the agent runtime."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localcode",
        description="LocalCode — AI coding assistant running entirely on your machine",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit")
    parser.add_argument("--model", help="Model alias to load (otherwise the picker asks)")
    parser.add_argument("-c", "--cwd", type=str, default=None,
                        help="Working directory for the project (defaults to the current directory)")
    parser.add_argument("project", nargs="?", default=None,
                        help="Project directory (same as --cwd)")
    return parser


def _unsupported_platform_message() -> str:
    import platform
    return (
        f"localcode: this release runs on Apple silicon Macs (macOS 13 or newer); "
        f"this machine reports {platform.system()} {platform.machine()}.\n"
        "Earlier releases installed on other platforms but bundled an Apple silicon "
        "model server, so they did not work there either."
    )


def main(argv: list[str] | None = None) -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # macOS malloc-warning silencer: these variables make every subprocess
    # print "can't turn off malloc stack logging" into stderr.
    for var in ("MallocStackLogging", "MallocStackLoggingNoCompact", "MallocNanoZone"):
        os.environ.pop(var, None)

    args = build_parser().parse_args(argv)

    if args.version:
        from . import __version__
        print(__version__)
        return

    project = args.cwd or args.project
    if project:
        project_dir = Path(project).resolve()
        if not project_dir.is_dir():
            print(f"localcode: directory not found: {project}", file=sys.stderr)
            sys.exit(1)
        os.chdir(project_dir)

    from .ui import platform_supported, ui_binary_path
    if not platform_supported():
        print(_unsupported_platform_message(), file=sys.stderr)
        sys.exit(1)
    if ui_binary_path() is None:
        print("localcode: the UI binary is missing from this install. "
              "Reinstall with `pip install -U localcode`.", file=sys.stderr)
        sys.exit(1)
    from .ui.launch import main as ui_main
    sys.exit(ui_main(args.model))


if __name__ == "__main__":
    main()
