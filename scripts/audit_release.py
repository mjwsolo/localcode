#!/usr/bin/env python3
"""Audit dirty worktree contents before releasing LocalCode.

This is deliberately simple and local-only. It prevents the recurring
failure mode where generated benchmark apps/logs get mixed with core
harness changes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


GENERATED_PREFIXES = (
    "sample_learning_app",
    "learn-",
    "logo",
)
GENERATED_SUFFIXES = (".log",)
CORE_PREFIXES = ("src/localcode/", "tests/", "scripts/", "pyproject.toml", "uv.lock")


def audit(repo: Path) -> str:
    rows = _git_status(repo)
    core: list[str] = []
    generated: list[str] = []
    other: list[str] = []
    deleted: list[str] = []
    for status, path in rows:
        item = f"{status} {path}"
        if status.strip() == "D":
            deleted.append(item)
        if _is_generated(path):
            generated.append(item)
        elif path.startswith(CORE_PREFIXES):
            core.append(item)
        else:
            other.append(item)
    lines = [
        "release audit",
        f"core_changes: {len(core)}",
        f"generated_or_logs: {len(generated)}",
        f"other_changes: {len(other)}",
        f"deleted_files: {len(deleted)}",
    ]
    if generated:
        lines.extend(["", "generated/log artifacts:", *generated[:80]])
    if other:
        lines.extend(["", "other dirty files:", *other[:80]])
    if deleted:
        lines.extend(["", "deleted files:", *deleted[:80]])
    if generated or other:
        lines.append("\nBefore release: commit/stage core harness changes separately from generated artifacts.")
    return "\n".join(lines)


def _git_status(repo: Path) -> list[tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append((status, path))
    return rows


def _is_generated(path: str) -> bool:
    p = path.lstrip("/")
    if p.endswith(GENERATED_SUFFIXES):
        return True
    return any(p == prefix or p.startswith(prefix + "/") for prefix in GENERATED_PREFIXES)


def main() -> int:
    print(audit(Path.cwd()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
