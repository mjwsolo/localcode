"""Guard the bundled binaries (`llama-server`, `localcode-ui`) actually work when shipped.

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

BIN_DIR = Path(localcode.__file__).parent / "bin"
BINARIES = ["llama-server", "localcode-ui"]
# Every shipped binary must load on macOS 13 (Ventura) or newer: the oldest
# macOS that runs on every Apple-Silicon Mac we support. A binary built on a
# newer SDK without a deployment target silently ships a higher `minos` and
# fails with a dyld error on older machines (0.3.71 shipped llama-server with
# minos 26.0). This is a hard failure, not a skip.
MAX_MINOS = (13, 0)


def _binary_minos(path: Path) -> tuple[int, ...] | None:
    """The binary's minimum macOS (LC_BUILD_VERSION `minos`), e.g. (13, 0)."""
    out = subprocess.run(
        ["otool", "-l", str(path)], capture_output=True, text=True, timeout=30, check=False,
    ).stdout
    m = re.search(r"minos (\d+(?:\.\d+)*)", out)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


@pytest.fixture(scope="module", params=BINARIES)
def binary(request) -> Path:
    path = BIN_DIR / request.param
    if not path.is_file():
        pytest.skip(f"no bundled binary at {path} (dev checkout without it)")
    return path


def test_bundled_binary_is_arm64(binary: Path):
    out = subprocess.run(["file", str(binary)], capture_output=True, text=True, timeout=30, check=False).stdout
    assert "Mach-O 64-bit executable arm64" in out, out


def test_bundled_binary_is_self_contained(binary: Path):
    """No `@rpath` dylibs and no Homebrew/local paths — the binary must depend only
    on system frameworks so it runs on any Apple-Silicon Mac, not just the build
    host. This is the check that would have caught the E1001 shared-libs regression."""
    out = subprocess.run(
        ["otool", "-L", str(binary)], capture_output=True, text=True, timeout=30, check=False,
    ).stdout
    offenders = [
        ln.strip()
        for ln in out.splitlines()[1:]
        if "@rpath/" in ln or "/opt/homebrew" in ln or "/usr/local/" in ln
    ]
    assert not offenders, "bundled binary has non-system dependencies:\n" + "\n".join(offenders)


def test_bundled_binary_targets_macos_13(binary: Path):
    minos = _binary_minos(binary)
    assert minos is not None, "no LC_BUILD_VERSION minos found"
    assert minos <= MAX_MINOS, (
        f"{binary.name} requires macOS {'.'.join(map(str, minos))}; ship with "
        f"CMAKE_OSX_DEPLOYMENT_TARGET / a deployment target of {MAX_MINOS[0]}.{MAX_MINOS[1]}"
    )


TOOLCHAIN_PATH_MARKERS = (
    "/.cargo/", "/.rustup/", "/_work/", "/work/_temp/", "buildkite", "/webkit-release/",
    "/opentui/", "/bun/bun/", "/target/aarch64-apple-darwin/",
)


def test_bundled_binary_embeds_no_developer_path(binary: Path):
    """A build from a developer checkout leaks the developer's home path into
    the binary (__FILE__ in asserts; bundled JS keeps __dirname). Build with
    -ffile-prefix-map (llama-server) and from a neutral path
    (scripts/build_ui_binary.sh). Paths from upstream toolchains (bun's own
    CI, cargo registries) are not ours and are ignored: on a GitHub runner
    the home dir is /Users/runner, which is also what bun's CI used."""
    out = subprocess.run(["strings", "-n", "12", str(binary)], capture_output=True, text=True, timeout=120, check=False).stdout
    home = str(Path.home())
    leaks = sorted({
        ln.strip()[:160] for ln in out.splitlines()
        if (home in ln or "/Desktop/" in ln or "/Github/" in ln or "/Documents/" in ln)
        and not any(m in ln for m in TOOLCHAIN_PATH_MARKERS)
    })
    assert not leaks, "developer paths embedded:\n" + "\n".join(leaks[:5])


def test_bundled_binary_loads_and_runs(binary: Path):
    """`--version` exits 0 — i.e. dyld resolves every linked symbol."""
    proc = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, (
        f"bundled {binary.name} failed to run (rc={proc.returncode}).\n{proc.stderr[-800:]}"
    )


def test_ui_binary_version_matches_package():
    path = BIN_DIR / "localcode-ui"
    if not path.is_file():
        pytest.skip("no bundled UI binary")
    from localcode import __version__
    want = re.sub(r"(\d)(a|b|rc)(\d+)$", r"\1-\2\3", __version__)
    out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60, check=False).stdout.strip()
    assert out == want, f"UI binary reports {out!r}, package is {want!r}; rebuild with scripts/build_ui_binary.sh"
