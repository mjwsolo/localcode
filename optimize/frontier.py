"""Pareto/best frontier of candidates + parent selection.

GEPA keeps a *frontier* of candidates rather than a single best, because a
prompt that wins on one objective (raw code-gen correctness) may lose on
another (agentic tool-use), and the best next mutation can come from either.
This is pure logic — no model calls — so it is unit-tested directly.

A candidate A *dominates* B when A is >= B on every objective and strictly >
on at least one. The Pareto frontier is the set of non-dominated candidates.
``best`` is still exposed for the single-objective "report the winner" path
and ties are broken deterministically (highest score, then lowest id).
"""
from __future__ import annotations

import random
from collections.abc import Iterable

from .candidate import Candidate


def dominates(a: Candidate, b: Candidate) -> bool:
    """True iff ``a`` Pareto-dominates ``b`` across all shared objectives."""
    va, vb = a.objective_vector(), b.objective_vector()
    keys = set(va) | set(vb)
    at_least_one_strictly_greater = False
    for k in keys:
        av, bv = va.get(k, 0.0), vb.get(k, 0.0)
        if av < bv:
            return False
        if av > bv:
            at_least_one_strictly_greater = True
    return at_least_one_strictly_greater


class Frontier:
    """Holds evaluated candidates and exposes the non-dominated frontier."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._all: list[Candidate] = []
        self._rng = rng or random.Random(0)

    def __len__(self) -> int:
        return len(self._all)

    @property
    def all(self) -> list[Candidate]:
        return list(self._all)

    def add(self, candidate: Candidate) -> None:
        """Record an evaluated candidate. Unevaluated ones are rejected so the
        frontier never selects a parent with no score/feedback to learn from.
        """
        if not candidate.evaluated:
            raise ValueError("cannot add an unevaluated candidate to the frontier")
        self._all.append(candidate)

    def add_all(self, candidates: Iterable[Candidate]) -> None:
        for c in candidates:
            self.add(c)

    def pareto(self) -> list[Candidate]:
        """The non-dominated set. A candidate stays unless some *other*
        candidate strictly dominates it. Equal-scoring duplicates all stay
        (none dominates another), which keeps lineage diversity.
        """
        front: list[Candidate] = []
        for c in self._all:
            if not any(dominates(o, c) for o in self._all if o is not c):
                front.append(c)
        return front

    def best(self) -> Candidate | None:
        """Single best by scalar score; deterministic tie-break (lowest id).

        Ties on score resolve to the *earliest* candidate, so a later mutation
        that merely matches the incumbent does not displace it — "keep best,
        don't regress on a tie".
        """
        if not self._all:
            return None
        return max(
            self._all,
            key=lambda c: (c.score if c.score is not None else float("-inf"), -c.cid),
        )

    def select_parents(self, k: int = 1) -> list[Candidate]:
        """Pick ``k`` parents to mutate from the Pareto frontier.

        Sampling from the frontier (not just the single best) is what keeps
        the search exploring multiple trade-off regions instead of collapsing
        onto one local optimum. Sampling is seeded for reproducibility.
        """
        front = self.pareto()
        if not front:
            return []
        if k >= len(front):
            return list(front)
        return self._rng.sample(front, k)
