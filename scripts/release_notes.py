#!/usr/bin/env python3
"""Print the CHANGELOG.md section for one version, for the GitHub release body.

    scripts/release_notes.py 0.4.8            -> the "## 0.4.8 — ..." section body
    scripts/release_notes.py 0.4.8 --check    -> exit 1 when the section is missing or empty

A pre-release such as 0.5.41rc1 needs no heading of its own: it uses the
"## 0.5.41rc1" section if there is one, else "## 0.5.41", else "## Unreleased",
else a one-line candidate notice, and --check always passes for it.

Headings are "## <version> — <date>" (or "## <version> — unreleased"); the body
runs until the next "## " heading. The publish workflow calls this on every tag,
so a release always carries the notes that were written for it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str | None:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^## {re.escape(version)}(\s|$)", line):
            start = i + 1
            break
    if start is None:
        return None
    body = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip() + "\n"


PRERELEASE = re.compile(r"^(\d+\.\d+\.\d+)(?:a|b|rc|\.dev|\.post)\d*$|^(\d+\.\d+\.\d+)[-.]?(?:alpha|beta|rc|dev|pre)", re.I)


def prerelease_base(version: str) -> str | None:
    m = PRERELEASE.match(version)
    return (m.group(1) or m.group(2)) if m else None


def notes_for(text: str, version: str) -> str | None:
    """The section for this version; a pre-release falls back as described above."""
    body = section(text, version)
    if body is not None and body.strip():
        return body
    base = prerelease_base(version)
    if base is None:
        return None
    for heading in (base, "Unreleased"):
        body = section(text, heading)
        if body is not None and body.strip():
            return f"Release candidate {version}; not the default install. Notes for {base} so far:\n\n{body}"
    return f"Release candidate {version}; not the default install. See CHANGELOG.md for {base} once it is released.\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[0].lstrip("v")
    check = "--check" in argv[1:]
    text = Path(__file__).resolve().parents[1].joinpath("CHANGELOG.md").read_text()
    body = notes_for(text, version)
    if body is None or not body.strip():
        if check:
            print(f"CHANGELOG.md has no notes for {version}", file=sys.stderr)
            return 1
        body = f"See CHANGELOG.md for {version}.\n"
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
