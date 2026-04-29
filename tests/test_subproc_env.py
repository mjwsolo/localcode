"""Verifies the malloc-noise vars never leak into spawned children.

Background: macOS libsystem fires a warning on every child whose env
contains MallocStackLogging[NoCompact] or MallocNanoZone — the exact
flood the user saw 2026-04-26. clean_env() is the single source of
truth for stripping these. This test asserts the helper actually
strips them AND that the three remaining spawn sites in the codebase
all route through it.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from localcode._subproc_env import _MALLOC_NOISE_VARS, clean_env


# ── Pure-function correctness ───────────────────────────────────────


def test_clean_env_strips_all_three_malloc_vars() -> None:
    base = {
        "PATH": "/usr/bin",
        "MallocStackLogging": "1",
        "MallocStackLoggingNoCompact": "1",
        "MallocNanoZone": "0",
        "HOME": "/Users/alice",
    }
    out = clean_env(base)
    assert "PATH" in out
    assert "HOME" in out
    assert "MallocStackLogging" not in out
    assert "MallocStackLoggingNoCompact" not in out
    assert "MallocNanoZone" not in out


def test_clean_env_does_not_mutate_input() -> None:
    base = {"MallocNanoZone": "0", "PATH": "/usr/bin"}
    _ = clean_env(base)
    assert base["MallocNanoZone"] == "0", "input dict was mutated"


def test_clean_env_with_no_arg_uses_os_environ() -> None:
    # Even with the noise vars set in os.environ for this test, the
    # output of clean_env() should not contain them.
    os.environ["MallocStackLogging"] = "1"
    os.environ["MallocNanoZone"] = "0"
    try:
        out = clean_env()
        assert "MallocStackLogging" not in out
        assert "MallocNanoZone" not in out
    finally:
        os.environ.pop("MallocStackLogging", None)
        os.environ.pop("MallocNanoZone", None)


def test_ban_list_covers_all_three_known_sources() -> None:
    """Lock the ban list so a future change can't silently drop a var
    and re-introduce the warning regression."""
    assert _MALLOC_NOISE_VARS == {
        "MallocStackLogging",
        "MallocStackLoggingNoCompact",
        "MallocNanoZone",
    }


# ── End-to-end: spawned child sees a clean env ──────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific behaviour")
def test_spawned_child_does_not_inherit_malloc_vars() -> None:
    """The whole point of clean_env(): hand it to subprocess and the
    child has no malloc-noise vars in its environment."""
    # Set vars in our process so we can verify they're stripped from
    # the child even though they're present here.
    os.environ["MallocStackLogging"] = "1"
    os.environ["MallocNanoZone"] = "0"
    try:
        r = subprocess.run(
            [
                sys.executable, "-c",
                "import os, json, sys;"
                "sys.stdout.write(json.dumps({"
                "'MallocStackLogging': os.environ.get('MallocStackLogging', '<unset>'),"
                "'MallocStackLoggingNoCompact': os.environ.get('MallocStackLoggingNoCompact', '<unset>'),"
                "'MallocNanoZone': os.environ.get('MallocNanoZone', '<unset>'),"
                "}))",
            ],
            env=clean_env(),
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0, f"child failed: {r.stderr!r}"
        import json
        seen = json.loads(r.stdout)
        assert seen["MallocStackLogging"] == "<unset>"
        assert seen["MallocStackLoggingNoCompact"] == "<unset>"
        assert seen["MallocNanoZone"] == "<unset>"
    finally:
        os.environ.pop("MallocStackLogging", None)
        os.environ.pop("MallocNanoZone", None)


# ── Codebase audit: every spawn site uses the helper ─────────────────


def test_no_ad_hoc_malloc_filters_remain() -> None:
    """Lock against drift: nobody should be writing their own
    `env.pop('MallocStackLogging', None)` anywhere; that must always
    go through clean_env()."""
    from pathlib import Path
    src_root = Path(__file__).resolve().parent.parent / "src" / "localcode"
    offenders = []
    for py in src_root.rglob("*.py"):
        # The helper itself is allowed to mention the names.
        if py.name == "_subproc_env.py":
            continue
        # The console-script entrypoint legitimately pops from os.environ on entry — that's
        # the parent-process scrub, not a per-spawn filter. Allow.
        if py.name == "entrypoint.py":
            continue
        text = py.read_text(errors="replace")
        for var in ("MallocStackLogging", "MallocStackLoggingNoCompact", "MallocNanoZone"):
            for line in text.splitlines():
                if f'.pop("{var}"' in line or f"startswith('{var}')" in line or f'startswith("{var}")' in line:
                    offenders.append(f"{py.relative_to(src_root)}: {line.strip()}")
    assert not offenders, (
        "Ad-hoc malloc-var filtering found — must use clean_env() instead:\n  "
        + "\n  ".join(offenders)
    )
