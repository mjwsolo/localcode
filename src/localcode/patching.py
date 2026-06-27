from __future__ import annotations

import re
from dataclasses import dataclass


DIFF_BLOCK_RE = re.compile(r"```diff\s*(.*?)```", re.DOTALL)


@dataclass
class DiffHunk:
    header: str
    lines: list[str]


@dataclass
class DiffFile:
    old_path: str
    new_path: str
    headers: list[str]
    hunks: list[DiffHunk]


def parse_diff(diff_text: str) -> list[DiffFile]:
    files: list[DiffFile] = []
    current: DiffFile | None = None
    current_hunk: DiffHunk | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if current is not None:
                if current_hunk is not None:
                    current.hunks.append(current_hunk)
                    current_hunk = None
                files.append(current)
            parts = line.split()
            old_path = parts[2][2:] if len(parts) > 2 else ""
            new_path = parts[3][2:] if len(parts) > 3 else old_path
            current = DiffFile(old_path=old_path, new_path=new_path, headers=[line], hunks=[])
            continue
        if current is None:
            continue
        if line.startswith("@@"):
            if current_hunk is not None:
                current.hunks.append(current_hunk)
            current_hunk = DiffHunk(header=line, lines=[])
            continue
        if current_hunk is not None:
            current_hunk.lines.append(line)
        else:
            current.headers.append(line)
    if current is not None:
        if current_hunk is not None:
            current.hunks.append(current_hunk)
        files.append(current)
    return files


def build_diff(selected_files: list[DiffFile]) -> str:
    blocks: list[str] = []
    for file in selected_files:
        blocks.extend(file.headers)
        for hunk in file.hunks:
            blocks.append(hunk.header)
            blocks.extend(hunk.lines)
    return "\n".join(blocks) + ("\n" if blocks else "")
