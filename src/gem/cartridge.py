from __future__ import annotations

from pathlib import Path

from .context import git_status, list_repo_files
from .indexer import load_index, search_index


def build_repo_cartridge(repo_root: Path, query: str, limit: int = 6) -> str:
    sections: list[str] = [
        f"repo: {repo_root}",
        "git status:",
        git_status(repo_root),
    ]
    files = list_repo_files(repo_root, limit=12)
    if files:
        sections.append("recent file sample:\n" + "\n".join(files[:12]))
    if load_index(repo_root) is not None and query.strip():
        hits = search_index(repo_root, query, limit=limit)
        if hits:
            sections.append(
                "indexed matches:\n" + "\n\n".join(
                    f"{item['path']}#chunk{item['chunk_id']}: {item['preview']}"
                    for item in hits
                )
            )
    cartridge = "\n\n".join(sections)
    return cartridge[:6000]
