"""The TUI must tear down its llama-server on every catchable exit signal.

A leaked llama-server holds multiple GB of RAM until the machine reboots or
it's killed by hand - a real problem on the 16 GB Macs localcode targets. The
app installs a cleanup handler for the signals that would otherwise kill it
without running atexit:

  - SIGINT  (Ctrl+C)
  - SIGTERM (kill, supervisor stop)
  - SIGHUP  (terminal window closed, SSH session dropped)  <- the leak the
            PTY QA pass caught: default SIGHUP action terminates WITHOUT
            running the cleanup, orphaning the server.

SIGKILL is uncatchable by design and is deliberately NOT covered here.
"""

from __future__ import annotations

import signal

import pytest

from localcode.tui.app import LocalCodeTUI

# SIGHUP only exists on Unix; localcode ships for macOS, but keep the guard so
# the suite is honest about scope on other platforms.
_SIGNALS = [signal.SIGINT, signal.SIGTERM]
if hasattr(signal, "SIGHUP"):
    _SIGNALS.append(signal.SIGHUP)


@pytest.fixture
def _preserve_handlers():
    """Snapshot and restore the process-wide handlers around the test."""
    saved = {s: signal.getsignal(s) for s in _SIGNALS}
    try:
        yield
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


@pytest.mark.parametrize("sig", _SIGNALS, ids=lambda s: signal.Signals(s).name)
def test_exit_signal_installs_cleanup_handler(sig, _preserve_handlers):
    LocalCodeTUI()
    handler = signal.getsignal(sig)
    assert callable(handler), f"{signal.Signals(sig).name} left at default/ignored"
    assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
    # All three route through the one cleanup closure that shuts the server down.
    assert getattr(handler, "__name__", "") == "_sig_cleanup"


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is Unix-only")
def test_sighup_specifically_is_not_left_at_default(_preserve_handlers):
    # Regression guard for the terminal-close / SSH-drop orphan: before this
    # fix SIGHUP was never installed, so closing the window leaked the server.
    LocalCodeTUI()
    assert signal.getsignal(signal.SIGHUP) not in (signal.SIG_DFL, signal.SIG_IGN)
