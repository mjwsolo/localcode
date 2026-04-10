"""Semantic code search using local embeddings.

Uses a small embedding model (nomic-embed-text or all-MiniLM) to enable
"find code that does X" queries across the codebase. ~300MB RAM overhead.

Index is built once and cached to .gem/embeddings/. Incremental updates
on file changes via the ProjectWatcher.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CodeChunk:
    """A searchable chunk of code."""
    file: str
    start_line: int
    end_line: int
    text: str
    language: str = ""


@dataclass
class SearchResult:
    """A semantic search result."""
    file: str
    start_line: int
    end_line: int
    score: float
    preview: str


class EmbeddingSearch:
    """Semantic code search backed by a local embedding model.

    Workflow:
    1. build_index() — chunk all code files, compute embeddings, save to disk
    2. search(query) — embed query, cosine similarity, return top-K
    3. update_files(paths) — incrementally re-embed changed files

    Model options (in order of preference):
    - sentence-transformers with nomic-ai/nomic-embed-text-v1.5 (~137M params)
    - sentence-transformers with all-MiniLM-L6-v2 (~22M params, faster)
    - Fallback: TF-IDF with scikit-learn (no GPU, always available)
    """

    CACHE_DIR = ".gem/embeddings"
    EMBEDDINGS_FILE = "vectors.npz"
    CHUNKS_FILE = "chunks.json"
    META_FILE = "meta.json"

    def __init__(self, project_root: str) -> None:
        self.root = Path(project_root)
        self.cache_path = self.root / self.CACHE_DIR
        self.model = None
        self.model_name = ""
        self.chunks: list[CodeChunk] = []
        self.embeddings: np.ndarray | None = None
        self._dimension = 0

    def setup(self) -> bool:
        """Load embedding model. Returns True if successful."""
        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            # Try nomic first (better quality)
            try:
                self.model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True,
                )
                self.model_name = "nomic-embed-text-v1.5"
            except Exception:
                # Fallback to MiniLM (smaller, always works)
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"

            # Use half precision to save memory
            try:
                self.model.half()
            except Exception:
                pass
            return True
        except ImportError:
            pass

        # Fallback: TF-IDF (no extra deps needed beyond scikit-learn)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = "tfidf"
            self.model_name = "tfidf"
            return True
        except ImportError:
            pass

        return False

    def build_index(self, extensions: set[str] | None = None) -> int:
        """Chunk all code files and compute embeddings. Returns chunk count.

        Takes ~10-30s for a medium project (~500 files).
        Results cached to disk for fast reload.
        """
        if not self.model:
            if not self.setup():
                return 0

        if extensions is None:
            extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                          ".rb", ".java", ".c", ".cpp", ".h", ".sh"}

        self.chunks = []
        skip_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv",
                     "dist", "build", ".egg-info", ".gem"}

        for fpath in self.root.rglob("*"):
            if not fpath.is_file() or fpath.suffix not in extensions:
                continue
            if any(s in fpath.parts for s in skip_dirs):
                continue
            try:
                content = fpath.read_text(errors="replace")
                rel = str(fpath.relative_to(self.root))
                lang = fpath.suffix.lstrip(".")
                file_chunks = self._chunk_code(rel, content, lang)
                self.chunks.extend(file_chunks)
            except Exception:
                continue

        if not self.chunks:
            return 0

        # Compute embeddings
        self.embeddings = self._embed_chunks(self.chunks)

        # Save to disk
        self._save()

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic search across the codebase."""
        if self.embeddings is None:
            if not self._load():
                return []

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Embed the query
        query_vec = self._embed_query(query)
        if query_vec is None:
            return []

        # Cosine similarity (embeddings are L2-normalized)
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            score = float(scores[idx])
            if score < 0.1:  # skip very low scores
                continue
            results.append(SearchResult(
                file=chunk.file,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=score,
                preview=chunk.text[:200],
            ))

        return results

    def update_files(self, paths: list[str]) -> int:
        """Incrementally update embeddings for changed files."""
        if self.embeddings is None:
            return self.build_index()

        updated = 0
        for rel_path in paths:
            fpath = self.root / rel_path
            if not fpath.is_file():
                # File deleted — remove its chunks
                self.chunks = [c for c in self.chunks if c.file != rel_path]
                updated += 1
                continue

            try:
                content = fpath.read_text(errors="replace")
                lang = fpath.suffix.lstrip(".")

                # Remove old chunks for this file
                self.chunks = [c for c in self.chunks if c.file != rel_path]

                # Add new chunks
                new_chunks = self._chunk_code(rel_path, content, lang)
                self.chunks.extend(new_chunks)
                updated += 1
            except Exception:
                continue

        if updated > 0:
            # Recompute all embeddings (simpler than partial update)
            self.embeddings = self._embed_chunks(self.chunks)
            self._save()

        return updated

    # ── Chunking ────────────────────────────────────────────────────

    def _chunk_code(self, file_path: str, content: str, language: str,
                    chunk_size: int = 30, overlap: int = 5) -> list[CodeChunk]:
        """Split code into overlapping chunks at function/class boundaries."""
        lines = content.splitlines()
        if not lines:
            return []

        # Find function/class boundaries
        boundaries = [0]
        boundary_keywords = {
            "py": ("def ", "class ", "async def "),
            "js": ("function ", "class ", "const ", "export function", "export class", "export default"),
            "ts": ("function ", "class ", "const ", "export function", "export class", "interface "),
            "jsx": ("function ", "class ", "const ", "export "),
            "tsx": ("function ", "class ", "const ", "export ", "interface "),
            "go": ("func ", "type "),
            "rs": ("fn ", "struct ", "impl ", "enum ", "trait "),
            "rb": ("def ", "class ", "module "),
            "java": ("public ", "private ", "protected ", "class ", "interface "),
            "c": ("int ", "void ", "char ", "static ", "struct "),
            "cpp": ("int ", "void ", "class ", "struct ", "namespace "),
            "h": ("int ", "void ", "class ", "struct "),
            "sh": ("function ", ),
        }

        kws = boundary_keywords.get(language, ("def ", "class ", "function "))
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if any(stripped.startswith(kw) for kw in kws):
                if i > 0:
                    boundaries.append(i)
        boundaries.append(len(lines))

        chunks: list[CodeChunk] = []

        # Create chunks from boundaries
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = min(boundaries[i] + chunk_size, len(lines))
            if i + 1 < len(boundaries):
                end = min(end, boundaries[i + 1])

            if end - start < 3:
                continue

            chunk_text = "\n".join(lines[start:end])
            chunks.append(CodeChunk(
                file=file_path,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
                language=language,
            ))

        # If no boundaries found or chunks are too few, use sliding window
        if len(chunks) <= 1 and len(lines) > chunk_size:
            chunks = []
            for start in range(0, len(lines), chunk_size - overlap):
                end = min(start + chunk_size, len(lines))
                chunk_text = "\n".join(lines[start:end])
                chunks.append(CodeChunk(
                    file=file_path,
                    start_line=start + 1,
                    end_line=end,
                    text=chunk_text,
                    language=language,
                ))

        return chunks

    # ── Embedding ───────────────────────────────────────────────────

    def _embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """Compute embeddings for all chunks."""
        texts = [self._prepare_text(c.text, mode="document") for c in chunks]

        if self.model_name == "tfidf":
            return self._tfidf_embed(texts)
        else:
            return self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

    def _embed_query(self, query: str) -> np.ndarray | None:
        """Compute embedding for a search query."""
        text = self._prepare_text(query, mode="query")

        if self.model_name == "tfidf":
            return self._tfidf_query(text)
        else:
            return self.model.encode(
                [text],
                normalize_embeddings=True,
            )

    def _prepare_text(self, text: str, mode: str = "document") -> str:
        """Format text for the embedding model."""
        if "nomic" in self.model_name:
            prefix = "search_document: " if mode == "document" else "search_query: "
            return prefix + text
        return text

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF fallback embedding."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]+",
        )
        matrix = self._tfidf_vectorizer.fit_transform(texts)
        return normalize(matrix).toarray().astype(np.float32)

    def _tfidf_query(self, text: str) -> np.ndarray | None:
        """TF-IDF query embedding."""
        from sklearn.preprocessing import normalize
        if not hasattr(self, "_tfidf_vectorizer"):
            return None
        matrix = self._tfidf_vectorizer.transform([text])
        return normalize(matrix).toarray().astype(np.float32)

    # ── Persistence ─────────────────────────────────────────────────

    def _save(self) -> None:
        """Save embeddings and chunks to disk."""
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.savez_compressed(
            str(self.cache_path / self.EMBEDDINGS_FILE),
            embeddings=self.embeddings,
        )

        # Save chunks
        chunks_data = [
            {
                "file": c.file,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "text": c.text,
                "language": c.language,
            }
            for c in self.chunks
        ]
        (self.cache_path / self.CHUNKS_FILE).write_text(json.dumps(chunks_data))

        # Save metadata
        meta = {
            "model": self.model_name,
            "chunk_count": len(self.chunks),
            "built_at": time.time(),
            "project_root": str(self.root),
        }
        (self.cache_path / self.META_FILE).write_text(json.dumps(meta))

    def _load(self) -> bool:
        """Load cached embeddings from disk."""
        emb_path = self.cache_path / self.EMBEDDINGS_FILE
        chunks_path = self.cache_path / self.CHUNKS_FILE

        if not emb_path.exists() or not chunks_path.exists():
            return False

        try:
            data = np.load(str(emb_path))
            self.embeddings = data["embeddings"]

            chunks_data = json.loads(chunks_path.read_text())
            self.chunks = [
                CodeChunk(
                    file=c["file"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    text=c["text"],
                    language=c.get("language", ""),
                )
                for c in chunks_data
            ]

            # Load metadata
            meta_path = self.cache_path / self.META_FILE
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                self.model_name = meta.get("model", "")

            # Setup model for queries
            if not self.model:
                self.setup()

            return True
        except Exception:
            return False

    def is_indexed(self) -> bool:
        """Check if an index exists on disk."""
        return (self.cache_path / self.EMBEDDINGS_FILE).exists()

    def index_stats(self) -> dict:
        """Return stats about the current index."""
        meta_path = self.cache_path / self.META_FILE
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                age = time.time() - meta.get("built_at", 0)
                return {
                    "model": meta.get("model", "unknown"),
                    "chunks": meta.get("chunk_count", 0),
                    "age_hours": round(age / 3600, 1),
                }
            except Exception:
                pass
        return {"model": "none", "chunks": 0, "age_hours": 0}
