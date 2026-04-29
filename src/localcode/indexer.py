from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .config import ensure_home_dirs
from .context import IGNORE_DIRS, list_repo_files, read_file


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@dataclass(slots=True)
class Chunk:
    path: str
    chunk_id: int
    text: str
    tokens: list[str]
    path_tokens: list[str]


def _index_dir() -> Path:
    root = ensure_home_dirs() / "indexes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_key(repo_root: Path) -> str:
    return hashlib.sha1(str(repo_root.resolve()).encode()).hexdigest()[:12]


def index_path(repo_root: Path) -> Path:
    return _index_dir() / f"{_repo_key(repo_root)}.json"


def build_index(repo_root: Path, chunk_chars: int = 1200) -> tuple[int, Path]:
    chunks: list[dict[str, object]] = []
    file_count = 0
    for relative_path in list_repo_files(repo_root, limit=5000):
        path = repo_root / relative_path
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        try:
            content = read_file(repo_root, relative_path, max_chars=200000)
        except Exception:
            continue
        file_count += 1
        for idx, start in enumerate(range(0, len(content), chunk_chars)):
            text = content[start:start + chunk_chars]
            tokens = sorted(set(token.lower() for token in TOKEN_RE.findall(text)))[:300]
            chunks.append(
                {
                    "path": relative_path,
                    "chunk_id": idx,
                    "text": text,
                    "tokens": tokens,
                    "path_tokens": sorted(set(token.lower() for token in TOKEN_RE.findall(relative_path))),
                }
            )
    payload = {"repo_root": str(repo_root), "files": file_count, "chunks": chunks}
    path = index_path(repo_root)
    path.write_text(json.dumps(payload))
    return file_count, path


def load_index(repo_root: Path) -> dict | None:
    path = index_path(repo_root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def search_index(repo_root: Path, query: str, limit: int = 8) -> list[dict[str, str]]:
    data = load_index(repo_root)
    if data is None:
        return []
    query_tokens = [token.lower() for token in TOKEN_RE.findall(query)]
    results: list[tuple[int, dict[str, str]]] = []
    for chunk in data.get("chunks", []):
        tokens = set(chunk.get("tokens", []))
        path_tokens = set(chunk.get("path_tokens", []))
        score = sum(2 for token in query_tokens if token in path_tokens)
        score += sum(1 for token in query_tokens if token in tokens)
        if score == 0:
            text = str(chunk.get("text", "")).lower()
            path_text = str(chunk.get("path", "")).lower()
            lowered_query = query.lower()
            if lowered_query not in text and lowered_query not in path_text:
                continue
            score = 1
        preview = str(chunk.get("text", "")).strip().replace("\n", " ")
        results.append(
            (
                score,
                {
                    "path": str(chunk.get("path", "")),
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "preview": preview[:240],
                },
            )
        )
    results.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in results[:limit]]
