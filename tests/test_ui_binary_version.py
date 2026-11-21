"""The bundled UI binary shows its version in the footer, and that version is
fixed when the binary is built (scripts/build_ui_binary.sh bakes the base of
pyproject's version). Bumping pyproject without rebuilding left 0.5.43rc2
showing 0.5.42, so a version bump must come with a rebuild."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "src" / "localcode" / "bin" / "localcode-ui"


def _base_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    return re.sub(r"(a|b|rc)\d+$", "", version)


@pytest.mark.skipif(not BINARY.exists(), reason="bundled UI binary not present")
def test_bundled_ui_footer_matches_package_version():
    base = _base_version()
    data = BINARY.read_bytes()
    assert f"'{base}'".encode() in data or f'"{base}"'.encode() in data, (
        f"src/localcode/bin/localcode-ui was built for another version, so its footer "
        f"would not show {base}. Rebuild it with scripts/build_ui_binary.sh."
    )
