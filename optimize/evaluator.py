"""Evaluate a candidate prompt -> (scalar score, natural-language feedback).

The objective is the EXISTING bench harness — the same task set and hidden
asserts ``scripts/bench_prompt_hard.py`` uses — so optimization is driven by
the project's real eval signal, not a toy metric. Crucially we capture BOTH:

  * a scalar (pass-rate) for the frontier, and
  * textual feedback: which tasks failed, the generated code, and the
    assertion/exception that sank it. That feedback is GEPA's input to the
    reflector.

Running the harness needs a live llama-server + a downloaded model, so the
real evaluator is import-guarded and only imported when actually run. Tests
use a mock evaluator (see optimize/tests/) and never touch this path.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .candidate import Candidate


@dataclass
class EvalResult:
    """What an evaluator returns: the scalar + the feedback text + the
    per-objective breakdown the Pareto frontier consumes."""

    score: float
    feedback: str
    scores: dict[str, float]


class Evaluator(Protocol):
    """The contract the loop depends on. Anything that maps a prompt string
    to an :class:`EvalResult` works — the real bench, or a test mock."""

    def evaluate(self, prompt: str) -> EvalResult: ...


# --- feedback rendering -------------------------------------------------

def render_feedback(failures: list[dict[str, Any]], summary: dict[str, float]) -> str:
    """Turn structured per-task failures into the prose the reflector reads.

    Pure function so it is unit-testable without any model/server. Each
    failure entry is ``{id, prompt, code, error}``; we cap the code/error so
    one runaway generation can't blow the reflector's context.
    """
    lines: list[str] = []
    lines.append("BENCHMARK FEEDBACK")
    lines.append(
        "Scores: " + ", ".join(f"{k}={v:.1%}" for k, v in sorted(summary.items()))
    )
    if not failures:
        lines.append("No failing tasks captured.")
        return "\n".join(lines)
    lines.append(f"\n{len(failures)} failing sample(s) (truncated):")
    for f in failures:
        lines.append(f"\n--- task: {f.get('id', '?')} ---")
        lines.append(f"asked: {f.get('prompt', '').strip()[:300]}")
        code = (f.get("code") or "").strip()
        lines.append("generated code:\n" + (code[:600] if code else "<none/empty>"))
        err = (f.get("error") or "").strip()
        if err:
            lines.append("failure: " + err[:300])
    return "\n".join(lines)


# --- real bench-harness evaluator --------------------------------------

class BenchEvaluator:
    """Real objective: spin up the model server once, score each prompt by
    re-running the bench tasks. Reuses the harness's TASKS, extractor, scorer,
    and ``_fill`` so the optimizer optimizes for exactly what the bench
    measures.

    Constructed lazily (server started in ``__enter__``) so importing this
    module is cheap and side-effect-free.
    """

    def __init__(
        self,
        model_file: str = "gemma-4-12b-it-UD-Q4_K_XL.gguf",
        k: int = 2,
        port: int = 8196,
        max_failures: int = 6,
    ) -> None:
        self.model_file = model_file
        self.k = k
        self.port = port
        self.max_failures = max_failures
        self._gw = None
        self._proc = None
        self._fill = None
        self._score = None
        self._extract = None
        self._tasks: list[dict[str, Any]] = []

    # Lazy import of the harness so unit tests never import the server stack.
    def _load_harness(self):
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root / "scripts"))
        from bench_diffusion_quants import TASKS, extract_python_code
        from bench_prompt_hard import HARD_TASKS
        from bench_prompt_variants import CODE_ONLY, _fill, _score, _wait_health

        self._fill = _fill
        self._score = _score
        self._extract = extract_python_code
        self._code_only = CODE_ONLY
        self._wait_health = _wait_health
        # Mix the easy + hard single-turn code-gen tasks: easy gives signal on
        # weak prompts, hard gives headroom so improvements remain visible.
        self._tasks = list(TASKS) + list(HARD_TASKS)

    def __enter__(self) -> "BenchEvaluator":
        import subprocess

        from localcode import models_catalog as catalog
        from localcode.config import RuntimeConfig
        from localcode.runtime import LocalCodeRuntimeGateway

        self._load_harness()
        model_path = Path(catalog.model_dir()) / self.model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"model not downloaded: {model_path}")
        cfg = RuntimeConfig()
        cfg.provider = "llama_cpp"
        cfg.model = str(model_path)
        gw = LocalCodeRuntimeGateway(cfg)
        cmd = gw.llama_server_command(str(model_path))
        if "--port" in cmd:
            cmd[cmd.index("--port") + 1] = str(self.port)
        gw.config.base_url = f"http://localhost:{self.port}"
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_health(self._proc, f"http://localhost:{self.port}/health")
        self._gw = gw
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()

    def evaluate(self, prompt: str) -> EvalResult:
        if self._gw is None:
            raise RuntimeError("use BenchEvaluator as a context manager")
        sys_msg = {"role": "system", "content": self._fill(prompt)}
        passes = 0
        total = 0
        failures: list[dict[str, Any]] = []
        for task in self._tasks:
            for _ in range(self.k):
                total += 1
                content: list[str] = []
                err = ""
                try:
                    for ev in self._gw.stream_chat_events(
                        [sys_msg, {"role": "user", "content": task["prompt"] + self._code_only}],
                        tools=None,
                        num_predict=512,
                    ):
                        if ev.get("type") == "content":
                            content.append(ev.get("content", ""))
                except Exception:  # capture model/transport errors as feedback
                    err = traceback.format_exc(limit=2)
                code = self._extract("".join(content))
                if not err and self._score(code, task["asserts"]):
                    passes += 1
                elif len(failures) < self.max_failures:
                    failures.append(
                        {
                            "id": task["id"],
                            "prompt": task["prompt"],
                            "code": code,
                            "error": err or "asserts failed or wrong output",
                        }
                    )
        rate = passes / total if total else 0.0
        summary = {"pass_rate": rate}
        return EvalResult(
            score=rate,
            feedback=render_feedback(failures, summary),
            scores=summary,
        )


def evaluate_candidate(candidate: Candidate, evaluator: Evaluator) -> Candidate:  # noqa: F811
    """Run ``evaluator`` on a candidate and write the result back onto it.

    Returns the same candidate (mutated) for convenient chaining. Kept thin
    and side-effect-local so the loop's wiring is easy to mock-test.
    """
    result = evaluator.evaluate(candidate.prompt)
    candidate.score = result.score
    candidate.feedback = result.feedback
    candidate.scores = dict(result.scores)
    return candidate
