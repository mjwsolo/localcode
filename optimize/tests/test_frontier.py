from __future__ import annotations

import random

import pytest

from optimize.candidate import Candidate
from optimize.frontier import Frontier, dominates


def _c(score=None, **scores):
    return Candidate(prompt="p", score=score, scores=scores)


def test_add_rejects_unevaluated():
    f = Frontier()
    with pytest.raises(ValueError):
        f.add(Candidate(prompt="p"))


def test_best_picks_highest_score():
    f = Frontier()
    f.add(_c(0.2))
    top = _c(0.9)
    f.add(top)
    f.add(_c(0.5))
    assert f.best() is top


def test_best_tie_keeps_earliest_no_regression():
    f = Frontier()
    first = _c(0.7)
    f.add(first)
    second = _c(0.7)  # ties incumbent -> must NOT displace it
    f.add(second)
    assert f.best() is first


def test_dominance_single_objective():
    hi, lo = _c(0.8), _c(0.4)
    assert dominates(hi, lo)
    assert not dominates(lo, hi)


def test_pareto_keeps_tradeoffs():
    f = Frontier()
    a = _c(0.6, codegen=0.9, agentic=0.3)  # strong codegen
    b = _c(0.6, codegen=0.3, agentic=0.9)  # strong agentic
    c = _c(0.2, codegen=0.2, agentic=0.2)  # dominated by both
    f.add_all([a, b, c])
    front = f.pareto()
    assert a in front and b in front
    assert c not in front  # strictly dominated


def test_pareto_keeps_equal_duplicates():
    f = Frontier()
    a = _c(0.5, x=0.5)
    b = _c(0.5, x=0.5)  # equal, neither dominates -> both stay
    f.add_all([a, b])
    assert {c.cid for c in f.pareto()} == {a.cid, b.cid}


def test_select_parents_from_frontier_deterministic():
    rng = random.Random(123)
    f = Frontier(rng=rng)
    a = _c(0.6, codegen=0.9, agentic=0.3)
    b = _c(0.6, codegen=0.3, agentic=0.9)
    f.add_all([a, b])
    picks = f.select_parents(k=1)
    assert len(picks) == 1 and picks[0] in (a, b)
    # k >= frontier size returns the whole frontier
    assert {c.cid for c in f.select_parents(k=5)} == {a.cid, b.cid}


def test_select_parents_empty():
    assert Frontier().select_parents() == []
