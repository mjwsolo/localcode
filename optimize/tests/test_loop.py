from __future__ import annotations

from optimize.candidate import Candidate
from optimize.loop import GepaConfig, run_gepa
from optimize.tests.mocks import ScriptedEvaluator, ScriptedReflector


def test_loop_improves_over_iterations():
    # Score = length proxy: longer prompt scores higher (deterministic signal).
    scorer = lambda p: min(1.0, len(p) / 100.0)  # noqa: E731
    ev = ScriptedEvaluator(scorer)
    # Each proposal is strictly longer -> strictly better.
    refl = ScriptedReflector(fn=lambda parent, ctx: parent.prompt + "X" * 20)
    res = run_gepa("start", ev, refl, GepaConfig(iterations=3, patience=5))
    assert res.best.score > scorer("start")
    # best is monotonic non-decreasing across history
    scores = [h["best_score"] for h in res.history]
    assert scores == sorted(scores)


def test_loop_keeps_best_when_proposals_regress():
    # Seed scores high; every proposal scores low -> best must stay the seed.
    scorer = lambda p: 0.9 if p == "seed" else 0.1  # noqa: E731
    ev = ScriptedEvaluator(scorer)
    refl = ScriptedReflector(fn=lambda parent, ctx: "worse-" + str(len(parent.prompt)))
    res = run_gepa("seed", ev, refl, GepaConfig(iterations=4, patience=10))
    assert res.best.prompt == "seed"
    assert res.best.score == 0.9


def test_early_stop_on_patience():
    # All proposals equal the seed score -> no improvement -> early stop.
    ev = ScriptedEvaluator(lambda p: 0.5)
    refl = ScriptedReflector(fn=lambda parent, ctx: parent.prompt + "!")
    res = run_gepa("seed", ev, refl, GepaConfig(iterations=10, patience=2))
    # seed iter (0) + exactly `patience` non-improving iters, then stop.
    iters_run = max(h["iteration"] for h in res.history)
    assert iters_run == 2


def test_handles_ties_without_regression():
    # Tie on score must not replace the incumbent best (lowest-id wins).
    ev = ScriptedEvaluator(lambda p: 0.6)
    refl = ScriptedReflector(fn=lambda parent, ctx: parent.prompt + "#")
    res = run_gepa("seed", ev, refl, GepaConfig(iterations=3, patience=10))
    assert res.best.iteration == 0  # original seed retained on ties


def test_seed_always_evaluated_and_on_frontier():
    ev = ScriptedEvaluator(lambda p: 0.5)
    refl = ScriptedReflector(fn=lambda parent, ctx: parent.prompt)
    res = run_gepa("seed", ev, refl, GepaConfig(iterations=1, patience=10))
    assert ev.calls[0] == "seed"  # seed evaluated first
    assert any(isinstance(c, Candidate) for c in res.frontier)


def test_reflector_receives_feedback_bearing_parent():
    seen = {}

    def fn(parent: Candidate, ctx: str) -> str:
        seen["feedback"] = parent.feedback
        return parent.prompt + "+"

    ev = ScriptedEvaluator(lambda p: min(1.0, len(p) / 10.0))
    run_gepa("seed", ev, ScriptedReflector(fn=fn),
             GepaConfig(iterations=1, patience=10))
    # The reflector saw the parent's textual feedback (the GEPA signal).
    assert "mock feedback" in seen["feedback"]
