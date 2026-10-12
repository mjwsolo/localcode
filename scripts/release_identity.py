#!/usr/bin/env python3
"""Resolve one release version and refuse mismatched or retargeted tags."""
from __future__ import annotations

import argparse
import re
import subprocess
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path


def release_identity(root: Path, ref: str, sha: str) -> tuple[str, str]:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.-]+)?", version):
        raise ValueError("Invalid package release version")
    fallback = re.search(r'^\s+__version__ = "([^"\n]+)"',
                         (root / "src/localcode/__init__.py").read_text(), re.MULTILINE)
    if fallback is None or fallback[1] != version:
        raise ValueError("Fallback version does not match pyproject.toml")
    tag = f"v{version}"
    if ref.startswith("refs/tags/") and ref != f"refs/tags/{tag}":
        raise ValueError("Release tag does not match package version")
    head = subprocess.check_output(["git", "rev-parse", f"{sha}^{{commit}}"], cwd=root, text=True).strip()
    existing = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
                              cwd=root, text=True, capture_output=True)
    if existing.returncode == 0 and existing.stdout.strip() != head:
        raise ValueError(f"{tag} already points to a different commit; bump the version")
    return version, tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        version, tag = release_identity(Path.cwd(), args.ref, args.sha)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release refused: {exc}\n")
    print(f"version={version}\ntag={tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
