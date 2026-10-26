#!/usr/bin/env python3
"""Rewrite the package version in a build tree (publish.yml, build job only; nothing is committed).

    release_promote.py <final version> [--root DIR]

A final tag v0.5.41 that points at the commit tested as 0.5.41rc1 publishes that
exact commit: the build first sets 0.5.41 in pyproject.toml and
src/localcode/__init__.py of its throwaway checkout. Main keeps the rc string.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def set_version(root: Path, version: str) -> None:
    edits = []
    for rel, pattern, repl in (("pyproject.toml", r'^version = ".*"$', f'version = "{version}"'),
                               ("src/localcode/__init__.py", r'^(\s+__version__ = )".*"$', rf'\1"{version}"')):
        path = root / rel
        new, n = re.subn(pattern, repl, path.read_text(), count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"no version to set in {rel}")
        edits.append((path, new))
    for path, new in edits:  # nothing is half-edited when one file fails
        path.write_text(new)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("version")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args(argv)
    if not re.fullmatch(r"\d+\.\d+\.\d+", a.version):
        p.error("give the final version, e.g. 0.5.41")
    try:
        set_version(a.root, a.version)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"version set to {a.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
