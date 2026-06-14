"""CLI entrypoint: ``python -m optimize.gepa --model <family> --iterations N``.

Wires the REAL bench evaluator + an LLM reflector and runs the loop, then
writes the best prompt and the score history to an output file. This is the
only place that actually spins up model servers, so importing the rest of the
package stays cheap and the unit tests stay model-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluator import BenchEvaluator
from .loop import GepaConfig, run_gepa
from .reflector import LLMReflector, default_system_prompt


def _build_reflector(model_file: str, port: int):
    """Construct an LLMReflector backed by its own llama-server.

    Returns ``(reflector, cleanup)``. The reflector model defaults to the same
    file as the bench model for a turnkey PoC, but a STRONGER model is
    strongly recommended (pass --reflector-model) — it has to reason about
    prompt design, which small models do poorly.
    """
    import subprocess

    from localcode import models_catalog as catalog
    from localcode.config import RuntimeConfig
    from localcode.runtime import LocalCodeRuntimeGateway

    model_path = Path(catalog.model_dir()) / model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"reflector model not downloaded: {model_path}")
    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(model_path)
    gw = LocalCodeRuntimeGateway(cfg)
    cmd = gw.llama_server_command(str(model_path))
    if "--port" in cmd:
        cmd[cmd.index("--port") + 1] = str(port)
    gw.config.base_url = f"http://localhost:{port}"
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Reuse the harness health-wait so we don't reimplement it.
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    from bench_prompt_variants import _wait_health

    _wait_health(proc, f"http://localhost:{port}/health")

    def cleanup() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    return LLMReflector(gw), cleanup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m optimize.gepa",
        description="GEPA-style reflective system-prompt optimizer (offline dev tool).",
    )
    ap.add_argument("--model", default="gemma-4-12b-it-UD-Q4_K_XL.gguf",
                    help="GGUF filename of the model the PROMPT is optimized for "
                         "(the bench objective).")
    ap.add_argument("--reflector-model", default=None,
                    help="GGUF filename of the model that PROPOSES rewrites "
                         "(defaults to --model; a stronger model is better).")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--proposals-per-iter", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--k", type=int, default=2, help="samples per task in the bench.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="optimize_result.json")
    args = ap.parse_args(argv)

    reflector_model = args.reflector_model or args.model
    cfg = GepaConfig(
        iterations=args.iterations,
        proposals_per_iter=args.proposals_per_iter,
        patience=args.patience,
        seed=args.seed,
    )

    reflector, cleanup_reflector = _build_reflector(reflector_model, port=8197)
    try:
        with BenchEvaluator(model_file=args.model, k=args.k, port=8196) as evaluator:
            result = run_gepa(
                seed_prompt=default_system_prompt(),
                evaluator=evaluator,
                reflector=reflector,
                config=cfg,
                on_event=lambda m: print(m, flush=True),
            )
    finally:
        cleanup_reflector()

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "reflector_model": reflector_model,
                "best_score": result.best.score,
                "best_prompt": result.best.prompt,
                "history": result.history,
                "frontier": [
                    {"cid": c.cid, "score": c.score, "iteration": c.iteration}
                    for c in result.frontier
                ],
            },
            indent=2,
        )
    )
    print(f"\nbest score: {result.best.score:.3f}")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
