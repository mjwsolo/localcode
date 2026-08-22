from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .agent.helpers import _is_blocked_write_path
from .config import ensure_home_dirs
from .context import IGNORE_DIRS, list_repo_files, read_file
from .paths import chmod_quiet


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

# Credential material that must never be copied verbatim into the index.
#
# The index at ~/.localcode/indexes/<sha1>.json stores full 1200-char chunks of
# every file it walks, so anything indexed is duplicated outside the repo, into
# a file the repo's own .gitignore does not cover. A `.env` with live API keys
# landing there is a real, observed outcome — hence this filter.
#
# Matching reuses `_is_blocked_write_path` (exact basename or exact path
# segment: `.ssh/`, `.aws/`, `id_rsa`, `.netrc`, `credentials.json`, …), never a
# naive substring, so the project's own `tokenizer.py` / `api_keys.py` still get
# indexed. On top of that: dotenv files and private-key / keystore extensions.
#
# NB: a denylist, not the extension ALLOWLIST that `embeddings.py:build_index`
# uses (and which is why the embedding index never had this bug). An allowlist
# would be strictly safer, but the lexical index is the one that answers
# "where is the Dockerfile / the Makefile / that YAML", so restricting it to a
# fixed set of source extensions would be a user-visible retrieval regression.
# The denylist buys the security fix without narrowing what the index can find.
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


def _is_secret_file(relative_path: str) -> bool:
    """True if this path looks like credential / key material."""
    if _is_blocked_write_path(relative_path):
        return True
    name = Path(relative_path).name.lower()
    # `.env`, `.env.local`, `.env.production` — but not `environment.py`.
    if name == ".env" or name.startswith(".env."):
        return True
    return name.endswith(_SECRET_SUFFIXES)


@dataclass
class Chunk:
    path: str
    chunk_id: int
    text: str
    tokens: list[str]
    path_tokens: list[str]


def _index_dir() -> Path:
    root = ensure_home_dirs() / "indexes"
    root.mkdir(parents=True, exist_ok=True)
    chmod_quiet(root, 0o700)
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
        if _is_secret_file(relative_path):
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
    # The index is a verbatim copy of the repo's text. Owner-read only.
    chmod_quiet(path, 0o600)
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
