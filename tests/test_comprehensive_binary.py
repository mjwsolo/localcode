"""Guard the bundled `llama-server` binary actually works when shipped.

Regression context: a rebuild once shipped a *dynamically linked* binary whose
`@rpath/libmtmd.0.dylib` etc. pointed at a local build dir that isn't in the
wheel — so the model server died with `dyld: Library not loaded` (E1001). The
`otool -L` self-containment check below catches exactly that class of bug and is
environment-independent, so it's the real guard.

macOS-only: the bundled binary is an Apple-Silicon static build; there is no
Linux build to exec on the Linux CI legs.
"""
from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path

import pytest

import localcode

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="bundled llama-server is an Apple-Silicon binary; no Linux build to exec",
)

BINARY = Path(localcode.__file__).parent / "bin" / "llama-server"


def _binary_minos(path: Path) -> tuple[int, ...] | None:
    """The binary's minimum macOS (LC_BUILD_VERSION `minos`), e.g. (26, 0)."""
    out = subprocess.run(
        ["otool", "-l", str(path)], capture_output=True, text=True, timeout=30
    ).stdout
    m = re.search(r"minos (\d+(?:\.\d+)*)", out)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


@pytest.fixture(scope="module")
def binary() -> Path:
    if not BINARY.is_file():
        pytest.skip(f"no bundled binary at {BINARY} (dev checkout without it)")
    return BINARY


def test_bundled_binary_is_self_contained(binary: Path):
    """No `@rpath` dylibs and no Homebrew/local paths — the binary must depend only
    on system frameworks so it runs on any Apple-Silicon Mac, not just the build
    host. This is the check that would have caught the E1001 shared-libs regression."""
    out = subprocess.run(
        ["otool", "-L", str(binary)], capture_output=True, text=True, timeout=30
    ).stdout
    offenders = [
        ln.strip()
        for ln in out.splitlines()[1:]
        if "@rpath/" in ln or "/opt/homebrew" in ln or "/usr/local/" in ln
    ]
    assert not offenders, "bundled binary has non-system dependencies:\n" + "\n".join(offenders)


def test_bundled_binary_loads_and_runs(binary: Path):
    """`--version` exits 0 — i.e. dyld resolves every linked symbol. Skipped when the
    host macOS is older than the binary's deployment target (e.g. CI's macOS-14
    runner can't load a macOS-26-built binary's newer Metal symbols)."""
    minos = _binary_minos(binary)
    host = tuple(int(x) for x in platform.mac_ver()[0].split(".") if x.isdigit())
    if minos and host and host[0] < minos[0]:
        pytest.skip(f"host macOS {host[0]} < binary minos {minos[0]}; cannot exec here")
    proc = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, (
        f"bundled llama-server failed to run (rc={proc.returncode}).\n{proc.stderr[-800:]}"
    )
