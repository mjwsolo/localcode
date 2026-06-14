"""GEPA-style reflective prompt-optimization loop (offline dev-time tooling).

This package is a SEPARATE, isolated dev tool. It MAY import from
``localcode`` (to read the live system prompt and reuse the bench harness),
but nothing under ``src/localcode`` imports from here — it is never part of
the shipped runtime.

See ``optimize/README.md`` for what it does and how to run it.
"""
from __future__ import annotations

__all__ = ["Candidate", "Frontier", "GepaConfig", "run_gepa"]

from .candidate import Candidate
from .frontier import Frontier
from .loop import GepaConfig, run_gepa
