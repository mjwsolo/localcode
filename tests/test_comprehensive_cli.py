"""CLI surface coverage — invoke every non-interactive subcommand as a real
subprocess and assert exit code + output.

These run the actual installed entrypoint (`python -m localcode ...`) in an
isolated LOCALCODE_HOME, so they catch import errors, argparse regressions,
and broken commands that unit tests miss. The interactive TUI (the no-arg
default) is covered separately by the Textual driver tests.
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


def test_help_lists_subcommands(tmp_path):
    # The TUI (bare `localcode`) is the product; only `run` (headless) and
    # `unstick` (recovery) remain as non-TUI subcommands. setup/config-init/
    # benchmark/models were removed — the TUI does all of those.
    r = _run_cli("--help", home=tmp_path)
    assert r.returncode == 0, r.stderr
    # The subcommand group renders as `{run,unstick}` — the ONLY two non-TUI
    # entry points. setup/config-init/benchmark/models were removed (the TUI
    # does all of those). Asserting the exact group is precise; the removed
    # words still appear in prose (e.g. "benchmark harness", a "setup" screen).
    assert "{run,unstick}" in r.stdout.replace(" ", ""), r.stdout


def test_preview_screen_choices_advertised(tmp_path):
    """`--help` should advertise every preview screen. (We don't *launch*
    them here — that opens a live TUI that never exits headlessly; the
    screens themselves are exercised by the Textual driver tests.)"""
    r = _run_cli("--help", home=tmp_path)
    assert r.returncode == 0, r.stderr
    for screen in ("setup", "mode-picker", "model-picker", "chat"):
        assert screen in r.stdout, f"preview screen {screen} not in --help"


def test_preview_screen_rejects_invalid_choice(tmp_path):
    """The flag validates its argument (proves it's wired into argparse)."""
    r = _run_cli("--preview-screen", "not-a-real-screen", home=tmp_path, timeout=30)
    assert r.returncode != 0
    assert "invalid choice" in (r.stderr + r.stdout)


def test_unknown_command_is_rejected(tmp_path):
    r = _run_cli("definitely-not-a-command", home=tmp_path)
    assert r.returncode != 0
