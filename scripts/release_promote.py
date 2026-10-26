#!/usr/bin/env python3
"""Turn a release candidate checkout into the final release (used by the release workflow).

    release_promote.py <final version> [--root DIR] [--date YYYY-MM-DD]

Sets the final version in pyproject.toml and src/localcode/__init__.py, and in
CHANGELOG.md renames the heading that holds the notes to "## <version> - <date>":
a "## <version>rcN" heading if there is one, else "## Unreleased". A heading for
the final version that already exists is left as it is. Fails if the changelog
would have no notes for the final version.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path


def promote_changelog(text: str, version: str, date: str) -> str:
    if re.search(rf"^## {re.escape(version)}(\s|$)", text, re.M):
        return text
    for pattern in (rf"^## {re.escape(version)}rc\d+(.*)$", r"^## Unreleased(.*)$"):
        new, n = re.subn(pattern, f"## {version} - {date}", text, count=1, flags=re.M)
        if n:
            return new
    raise ValueError(f"CHANGELOG.md has no '## {version}', '## {version}rcN' or '## Unreleased' heading")


def set_version(root: Path, version: str) -> None:
    for rel, pattern in (("pyproject.toml", r'^version = ".*"$'), ("src/localcode/__init__.py", r'^(\s+__version__ = )".*"$')):
        path = root / rel
        new, n = re.subn(pattern, (f'version = "{version}"' if rel == "pyproject.toml" else rf'\1"{version}"'), path.read_text(), count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"no version to set in {rel}")
        path.write_text(new)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("version")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--date", default=datetime.date.today().isoformat())
    a = p.parse_args(argv)
    if not re.fullmatch(r"\d+\.\d+\.\d+", a.version):
        p.error("give the final version, e.g. 0.5.41")
    try:
        log = a.root / "CHANGELOG.md"
        updated = promote_changelog(log.read_text(), a.version, a.date)
        set_version(a.root, a.version)
        log.write_text(updated)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(f"promoted to {a.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
