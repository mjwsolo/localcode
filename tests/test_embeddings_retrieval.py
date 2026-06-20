"""Regression coverage for the embedding (semantic) retrieval leg.

F6: on the TF-IDF fallback path (the default when `sentence-transformers`
isn't installed) the fitted vectorizer lives only in memory and is never
persisted. After a process restart, `_load` restored the saved doc matrix but
not the vectorizer, so `_tfidf_query` returned None and the whole semantic leg
silently went dead — `search()` returned `[]` forever. `_ensure_tfidf_vectorizer`
now rebuilds the vectorizer from the loaded chunk texts on first query.
"""

from __future__ import annotations

from localcode.embeddings import EmbeddingSearch


def _force_tfidf(self) -> bool:
    """Pin setup() to the TF-IDF backend so the test is deterministic
    regardless of whether sentence-transformers is installed in the venv."""
    self.model = "tfidf"
    self.model_name = "tfidf"
    return True


def test_tfidf_semantic_search_survives_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(EmbeddingSearch, "setup", _force_tfidf)

    (tmp_path / "alpha.py").write_text(
        "def authenticate_user(token):\n"
        "    # validate the session token\n"
        "    return verify_signature(token)\n"
    )
    (tmp_path / "beta.py").write_text(
        "def render_dashboard(widgets):\n"
        "    return layout_widgets(widgets)\n"
    )

    builder = EmbeddingSearch(str(tmp_path))
    assert builder.build_index() > 0
    # Same-process search works — the vectorizer is still in memory.
    assert builder.search("authenticate_user token session", top_k=3)

    # Simulate a process restart: a brand-new instance over the same cache,
    # with no in-memory vectorizer. Before the fix this returned [].
    restarted = EmbeddingSearch(str(tmp_path))
    assert restarted.is_indexed()
    assert not hasattr(restarted, "_tfidf_vectorizer")

    results = restarted.search("authenticate_user token session", top_k=3)
    assert results, "semantic leg went dead after restart (F6 regression)"
    assert any("alpha.py" in r.file for r in results)
    # The vectorizer was rebuilt lazily on the query path.
    assert hasattr(restarted, "_tfidf_vectorizer")
