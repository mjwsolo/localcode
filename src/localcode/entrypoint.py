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
        epilog="Other commands: localcode doctor diagnoses installation; localcode api shows connection details; localcode decide asks a typed question.",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit")
    parser.add_argument("--mode", choices=["chat", "decisions"], default="chat", help="Start in normal chat or typed decision mode")
    parser.add_argument("--trust-project", action="store_true", help="Allow this project's configuration, plugins and MCP startup commands to load")
    parser.add_argument("--model", help="Model alias to load (otherwise the picker asks)")
    parser.add_argument("-s", "--session", "--resume", dest="resume", metavar="SESSION_ID",
                        help="Resume a session by ID; uses its saved project directory when no project is given")
    parser.add_argument("project", nargs="?", default=None,
                        help="Project directory (defaults to the current directory)")
    return parser


def _unsupported_platform_message() -> str:
    import platform
    from .diagnostics import platform_problem
    detail = platform_problem() or ""
    return (
        f"localcode: this release runs on Apple silicon Macs (macOS 13 or newer); "
        f"this machine reports {platform.system()} {platform.machine()}.\n"
        f"{detail}\n"
        "Earlier releases installed on other platforms but bundled an Apple silicon "
        "model server, so they did not work there either."
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["doctor"]:
        from .diagnostics import main as doctor_main
        sys.exit(doctor_main())
    if argv and argv[0] == "decide":
        from .decision import main as decide_main
        sys.exit(decide_main(argv[1:]))
    if argv and argv[0] == "api":
        from .ui.api import main as api_main
        sys.exit(api_main(argv[1:]))
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # macOS malloc-warning silencer: these variables make every subprocess
    # print "can't turn off malloc stack logging" into stderr.
    for var in ("MallocStackLogging", "MallocStackLoggingNoCompact", "MallocNanoZone"):
        os.environ.pop(var, None)

    args = build_parser().parse_args(argv)
    os.environ["LOCALCODE_MODE"] = args.mode

    if args.version:
        from . import __version__
        print(__version__)
        return

    project = args.project
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
    sys.exit(ui_main(args.model, project=project, resume=args.resume, trust_project=args.trust_project))


if __name__ == "__main__":
    main()
