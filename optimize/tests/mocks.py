"""Mock evaluator + reflector so the loop is testable without models/network."""
from __future__ import annotations

from collections.abc import Callable

from optimize.candidate import Candidate
from optimize.evaluator import EvalResult


class ScriptedEvaluator:
    """Scores a prompt by a caller-supplied scoring function.

    The scorer maps prompt-string -> float; feedback is synthesized so the
    reflector path has something to read. Counts calls for assertions.
    """

    def __init__(self, scorer: Callable[[str], float]) -> None:
        self._scorer = scorer
        self.calls: list[str] = []

    def evaluate(self, prompt: str) -> EvalResult:
        self.calls.append(prompt)
        score = self._scorer(prompt)
        return EvalResult(
            score=score,
            feedback=f"mock feedback for score {score:.3f}",
            scores={"pass_rate": score},
        )


class ScriptedReflector:
    """Proposes prompts from a fixed list, or via a function of the parent.

    With a list it pops the next proposal each call (handy for "improve, then
    plateau" scenarios). With a function it computes one from the parent.
    """

    def __init__(
        self,
        proposals: list[str] | None = None,
        fn: Callable[[Candidate, str], str] | None = None,
    ) -> None:
        self._proposals = list(proposals or [])
        self._fn = fn
        self.calls: list[Candidate] = []

    def propose(self, parent: Candidate, context: str = "") -> str:
        self.calls.append(parent)
        if self._fn is not None:
            return self._fn(parent, context)
        if self._proposals:
            return self._proposals.pop(0)
        return parent.prompt  # nothing left -> propose the parent unchanged
