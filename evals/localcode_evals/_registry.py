"""Loaded by Inspect AI through the `inspect_ai` entry point."""
from .agent import _localcode_provider, localcode_ui  # noqa: F401
from .suite import hidden_tests, localcode, measurements, regression, tracking  # noqa: F401
from .benchmarks import (harnessbench, harnessbench_oracle, harnessbench_prepare, livecodebench, livecodebench_cases,  # noqa: F401
                         polyglot, polyglot_tests, reference_solution, swebench, swebench_tests)
