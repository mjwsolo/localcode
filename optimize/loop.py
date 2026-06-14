"""The GEPA optimization loop: seed -> evaluate -> reflect -> select -> repeat.

This module wires the pure pieces together and is itself dependency-injected:
it takes an ``Evaluator`` and a ``Reflector`` (both protocols), so the whole
loop is unit-testable with mocks and never needs a model or the network.

Algorithm (one iteration):
  1. select parent(s) from the Pareto frontier;
  2. for each parent, ask the reflector for an improved prompt (it reads the
     parent's textual feedback — the GEPA signal);
  3. evaluate each fresh proposal on the bench objective;
  4. add evaluated proposals to the frontier.

The best candidate is tracked across all iterations, so the loop is
guaranteed never to *return* a regression even if a given iteration proposes
a worse prompt. Early-stop triggers after ``patience`` iterations with no
improvement to the global best.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .candidate import Candidate
from .evaluator import Evaluator, evaluate_candidate
from .frontier import Frontier
from .reflector import Reflector


@dataclass
class GepaConfig:
    iterations: int = 6
    proposals_per_iter: int = 1
    patience: int = 3  # stop after this many iters with no global improvement
    min_improvement: float = 1e-9  # score delta that counts as "better"
    seed: int = 0


@dataclass
class GepaResult:
    best: Candidate
    frontier: list[Candidate]
    history: list[dict] = field(default_factory=list)


def run_gepa(
    seed_prompt: str,
    evaluator: Evaluator,
    reflector: Reflector,
    config: GepaConfig | None = None,
    on_event: Callable[[str], None] | None = None,
) -> GepaResult:
    """Run the reflective optimization loop and return the best candidate.

    Args:
        seed_prompt: starting system prompt (typically the live SYSTEM_PROMPT).
        evaluator: maps a prompt -> EvalResult (real bench, or a mock).
        reflector: proposes an improved prompt from a parent + feedback.
        config: loop knobs (iterations, proposals/iter, patience).
        on_event: optional sink for human-readable progress lines.
    """
    cfg = config or GepaConfig()
    log = on_event or (lambda _msg: None)
    rng = random.Random(cfg.seed)
    frontier = Frontier(rng=rng)
    history: list[dict] = []

    # Seed: evaluate the starting prompt so we always have a baseline + a
    # populated frontier to select the first parent from.
    seed = evaluate_candidate(
        Candidate(prompt=seed_prompt, iteration=0), evaluator
    )
    frontier.add(seed)
    best = seed
    log(f"seed {seed.short()}")
    history.append({"iteration": 0, "best_score": best.score,
                    "evaluated": [seed.cid]})

    no_improve = 0
    for it in range(1, cfg.iterations + 1):
        parents = frontier.select_parents(k=cfg.proposals_per_iter)
        if not parents:
            break
        evaluated_ids: list[int] = []
        for parent in parents:
            # Show the reflector the other frontier prompts as context so it
            # can borrow what's working elsewhere, not just patch one parent.
            others = [c for c in frontier.pareto() if c is not parent]
            context = "\n---\n".join(c.prompt for c in others[:2])
            proposed_prompt = reflector.propose(parent, context=context)
            child = Candidate(
                prompt=proposed_prompt,
                iteration=it,
                parent_id=parent.cid,
            )
            evaluate_candidate(child, evaluator)
            frontier.add(child)
            evaluated_ids.append(child.cid)
            log(f"it{it}: {parent.short()} -> {child.short()}")

        current_best = frontier.best()
        improved = (
            current_best is not None
            and best is not None
            and current_best.score is not None
            and best.score is not None
            and current_best.score - best.score > cfg.min_improvement
        )
        if improved:
            best = current_best
            no_improve = 0
            log(f"it{it}: new best {best.short()}")
        else:
            no_improve += 1
            # Keep the incumbent best (no regression) even if this iter was worse.
            best = current_best if current_best is not None else best

        history.append({"iteration": it, "best_score": best.score,
                        "evaluated": evaluated_ids, "improved": improved})

        if no_improve >= cfg.patience:
            log(f"early stop: no improvement in {cfg.patience} iterations")
            break

    return GepaResult(best=best, frontier=frontier.pareto(), history=history)
