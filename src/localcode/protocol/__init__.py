"""LocalCode public automation protocol.

The single, versioned, dependency-light definition of the `localcode run --json`
event stream. Both the native `eval/` harness and the Harbor plugin consume
LocalCode through this package — neither maintains its own JSONL parser or
failure taxonomy (integration plan §7.2.1). Standard library only, so any
consumer can import it without pulling in LocalCode's full dependency tree.

Typical use:

    from localcode.protocol import parse_stream, outcome_from_parse

    parsed = parse_stream(process.stdout)      # incremental, memory-bounded
    outcome = outcome_from_parse(parsed)       # normalized RunOutcome
    if outcome.clean_finish:                    # LocalCode's loop finished...
        ...                                     # ...verifier still decides pass/fail
"""
from __future__ import annotations

from .events import (
    SCHEMA_MAJOR,
    SCHEMA_MINOR,
    SCHEMA_VERSION,
    Event,
    SchemaVersion,
    UnsupportedSchemaError,
)
from .jsonl import ParseProblem, ParseResult, iter_events, parse_stream
from .outcomes import (
    FailureCategory,
    RunOutcome,
    normalize_reason,
    outcome_from_parse,
    redact,
)

__all__ = [
    # schema
    "SCHEMA_MAJOR",
    "SCHEMA_MINOR",
    "SCHEMA_VERSION",
    "SchemaVersion",
    "UnsupportedSchemaError",
    # events + parsing
    "Event",
    "ParseProblem",
    "ParseResult",
    "parse_stream",
    "iter_events",
    # outcomes
    "FailureCategory",
    "RunOutcome",
    "normalize_reason",
    "outcome_from_parse",
    "redact",
]
