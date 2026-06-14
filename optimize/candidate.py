"""The unit of optimization: a system-prompt variant plus its eval result.

A ``Candidate`` is one system-prompt string and (once evaluated) its scalar
score and the natural-language feedback the bench harness produced. The
feedback is GEPA's key input — the reflector reads it to propose a better
variant, so we keep it on the candidate rather than throwing it away.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

# Monotonic id source so candidates have a stable, human-readable lineage
# even when two prompts hash-collide on text (e.g. the seed re-evaluated).
_COUNTER = itertools.count()


@dataclass
class Candidate:
    """A system-prompt variant under optimization.

    Attributes:
        prompt: the full system-prompt template string (with ``{cwd}`` etc.
            placeholders still present — the evaluator fills them).
        score: scalar objective in ``[0, 1]`` (pass-rate). ``None`` until
            evaluated.
        feedback: natural-language failure/trace text from the bench run.
            What the reflector reads to propose an improvement.
        scores: per-objective breakdown (e.g. codegen vs agentic rate) used
            by the Pareto frontier. Falls back to ``{"score": score}``.
        iteration: which loop iteration produced this candidate (0 = seed).
        parent_id: id of the candidate this was mutated from (None for seed).
        cid: stable unique id for lineage/printing.
    """

    prompt: str
    score: float | None = None
    feedback: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    iteration: int = 0
    parent_id: int | None = None
    cid: int = field(default_factory=lambda: next(_COUNTER))

    @property
    def evaluated(self) -> bool:
        return self.score is not None

    @property
    def prompt_hash(self) -> str:
        """Short content hash — used to dedup re-proposed identical prompts."""
        return hashlib.sha1(self.prompt.encode("utf-8")).hexdigest()[:12]

    def objective_vector(self) -> dict[str, float]:
        """Per-objective scores for Pareto comparison.

        Prefer the explicit multi-objective ``scores`` dict; fall back to a
        single ``{"score": ...}`` so single-objective evals still work.
        """
        if self.scores:
            return dict(self.scores)
        return {"score": self.score if self.score is not None else 0.0}

    def short(self) -> str:
        s = "n/a" if self.score is None else f"{self.score:.3f}"
        return f"cand#{self.cid}(it{self.iteration} score={s} {self.prompt_hash})"
