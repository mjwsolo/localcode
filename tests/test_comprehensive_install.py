"""Real isolated-install smoke test — `pip install` into a clean venv, then
run the CLI from that venv.

This is the only test that exercises packaging end-to-end: it catches
missing package data, broken entry points, bad metadata, and import-time
failures that source-tree tests can't see (they always have `src/` on the
path). It builds nothing native — Cython is opt-in (see setup.py), so this
is a pure-Python install plus the PyPI dependency tree.

It's SLOW (creates a venv and downloads deps) and needs network, so it's
opt-in. Run it explicitly:

    LOCALCODE_RUN_INSTALL_TEST=1 pytest tests/test_isolated_install.py -v

It is skipped in the default `pytest` run.
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("LOCALCODE_RUN_INSTALL_TEST") != "1",
        reason="set LOCALCODE_RUN_INSTALL_TEST=1 to run the real venv-install test",
    ),
]


def _venv_bin(venv_dir: Path, name: str) -> Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return venv_dir / sub / name


@pytest.fixture(scope="module")
def installed_venv(tmp_path_factory):
    """A throwaway venv with `localcode` pip-installed from this repo."""
    venv_dir = tmp_path_factory.mktemp("lc_install") / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    pip = _venv_bin(venv_dir, "pip")

    # Upgrade pip first (old pip can choke on modern metadata), then install.
    subprocess.run([str(pip), "install", "-q", "--upgrade", "pip"],
                   check=True, timeout=300)
    proc = subprocess.run(
        [str(pip), "install", "-q", str(REPO_ROOT)],
        capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, f"pip install failed:\n{proc.stdout}\n{proc.stderr}"
    return venv_dir


def _run_installed(venv_dir: Path, *args: str, home: Path) -> subprocess.CompletedProcess:
    lc = _venv_bin(venv_dir, "localcode")
    env = dict(os.environ)
    env.update({"LOCALCODE_HOME": str(home), "HOME": str(home), "NO_COLOR": "1"})
    return subprocess.run([str(lc), *args], capture_output=True, text=True,
                          timeout=120, env=env)


def test_entrypoint_is_installed(installed_venv):
    """Both console scripts (`localcode` and `lc`) should exist."""
    assert _venv_bin(installed_venv, "localcode").exists()
    assert _venv_bin(installed_venv, "lc").exists()


def test_installed_help_runs(installed_venv, tmp_path):
    r = _run_installed(installed_venv, "--help", home=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "LocalCode" in r.stdout


def test_installed_package_imports_clean(installed_venv):
    """Import the top-level package from the installed location (NOT the
    source tree) to catch missing modules / packaging gaps."""
    py = _venv_bin(installed_venv, "python")
    r = subprocess.run(
        [str(py), "-c", "import localcode, localcode.entrypoint; print('ok')"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
