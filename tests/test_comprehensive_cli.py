"""CLI surface coverage — invoke every non-interactive subcommand as a real
subprocess and assert exit code + output.

These run the actual installed entrypoint (`python -m localcode ...`) in an
isolated LOCALCODE_HOME, so they catch import errors, argparse regressions,
and broken commands that unit tests miss. The interactive UI (the no-arg
default) needs the bundled binary and is covered by the plugin tests.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_SRC = str(Path(__file__).resolve().parent.parent / "src")


def _run_cli(*args: str, home: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    """Invoke `python -m localcode <args>` with an isolated home + src path."""
    env = {
        "LOCALCODE_HOME": str(home),
        "PYTHONPATH": _SRC,
        "PATH": __import__("os").environ.get("PATH", ""),
        "HOME": str(home),  # keep any ~ writes inside the sandbox
        "NO_COLOR": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "localcode", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def test_unknown_command_is_rejected(tmp_path):
    r = _run_cli("definitely-not-a-command", home=tmp_path)
    assert r.returncode != 0


def test_help_has_no_subcommands(tmp_path):
    """`localcode` is one command: a project directory and a model. The old
    `run` and `unstick` subcommands went with the previous interface."""
    r = _run_cli("--help", home=tmp_path)
    assert r.returncode == 0
    assert "{run,unstick}" not in r.stdout and "--classic" not in r.stdout
    assert "[project]" in r.stdout


def test_version_prints_package_version(tmp_path):
    from localcode import __version__
    r = _run_cli("--version", home=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == __version__
