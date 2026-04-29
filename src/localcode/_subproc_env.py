"""Single source of truth for the env dict passed to subprocesses.

Background: macOS's libsystem prints
  "MallocStackLogging: can't turn off malloc stack logging
   because it was not enabled."
to stderr at libsystem-init of every spawned child whenever certain
env vars are set in the child's environment. The vars are commonly
set by Xcode CLI tools, IDE shell integrations, and Conda env-activate
hooks. The warning fires before any user code runs, so Python can't
suppress it from inside the child — the only fix is to strip the vars
from the env dict we hand to subprocess.Popen / subprocess.run.

the console-script entrypoint pops them from os.environ on entry, but every site that builds
an explicit env dict must also strip them or risk re-introducing the
var. Real failure 2026-04-26: terminal flooded with ~60 of these
warnings because spawn sites filtered MallocStackLogging* but missed
MallocNanoZone, which produces the same warning class.

This module is the SINGLE place that knows the full ban list. Every
subprocess spawn that needs to construct an env dict should use
clean_env() rather than rolling its own filter.
"""
from __future__ import annotations

import os
from typing import Mapping


# Env vars whose presence triggers the libsystem malloc-stack-logging
# warning at child-process startup on macOS. Sources:
#   • MallocStackLogging[NoCompact]: enabled by Xcode Instruments and
#     a few legacy debug profiles; setting it to "0" doesn't disable —
#     the var must be UNSET.
#   • MallocNanoZone: typically "0" set by Xcode CLI tools / IDE
#     terminal integrations to opt-out of the nano malloc zone.
#     Triggers the same class of warning at libsystem init.
_MALLOC_NOISE_VARS = {
    "MallocStackLogging",
    "MallocStackLoggingNoCompact",
    "MallocNanoZone",
}


def clean_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of `base` (default: os.environ) with malloc-noise
    vars stripped. Safe to pass to subprocess.Popen / subprocess.run as
    `env=`.

    Always returns a NEW dict — never mutates the input. Callers that
    need to add or override keys (`env["GGML_BACKEND_PATH"] = ""`,
    etc.) can do so on the returned dict without affecting os.environ.
    """
    src = os.environ if base is None else base
    return {k: v for k, v in src.items() if k not in _MALLOC_NOISE_VARS}
