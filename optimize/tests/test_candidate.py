from __future__ import annotations

from optimize.candidate import Candidate


def test_unevaluated_then_evaluated():
    c = Candidate(prompt="p")
    assert c.evaluated is False
    assert c.objective_vector() == {"score": 0.0}
    c.score = 0.5
    assert c.evaluated is True
    assert c.objective_vector() == {"score": 0.5}


def test_unique_ids_and_stable_hash():
    a = Candidate(prompt="same")
    b = Candidate(prompt="same")
    assert a.cid != b.cid  # distinct lineage even for identical text
    assert a.prompt_hash == b.prompt_hash  # content hash is stable


def test_multi_objective_vector_prefers_scores_dict():
    c = Candidate(prompt="p", score=0.5, scores={"codegen": 0.8, "agentic": 0.2})
    assert c.objective_vector() == {"codegen": 0.8, "agentic": 0.2}


def test_short_renders_without_score():
    assert "n/a" in Candidate(prompt="p").short()
