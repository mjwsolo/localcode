"""Incremental JSONL parser for the `localcode run --json` stream.

Consumes the event stream one line at a time from a text or byte iterable
without buffering the whole transcript, so a long agent run or a live process
pipe never has to be held in memory. It is deliberately tolerant of the messy
realities of a subprocess stream (partial final line after a kill, an
occasional non-JSON line, non-UTF-8 bytes) up to a small threshold, while still
surfacing genuine protocol violations (a missing or duplicated terminal event,
an unreadable major schema) as structured problems rather than swallowing them.

The parser NEVER decides whether a benchmark task passed. It reports what the
stream said; grading lives in the caller's verifier (integration plan §9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from .events import Event, SchemaVersion, UnsupportedSchemaError

__all__ = ["ParseProblem", "ParseResult", "parse_stream", "iter_events"]

# Stop tolerating junk once this many lines fail to decode — a stream that is
# mostly noise is a real problem, not a stray line, and should be reported.
_MAX_MALFORMED = 50


@dataclass
class ParseProblem:
    """A non-fatal issue found while parsing, kept for diagnostics."""

    kind: str          # "malformed_line" | "missing_terminal" | "duplicate_terminal" | "decode_error"
    detail: str
    line_no: int = 0


@dataclass
class ParseResult:
    """Everything a consumer needs from one parsed run stream."""

    events: list[Event] = field(default_factory=list)
    schema: SchemaVersion | None = None
    terminal: Event | None = None          # the mandatory `result` event, if present
    problems: list[ParseProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the stream was well-formed: a terminal event, no protocol problems."""
        return self.terminal is not None and not any(
            p.kind in {"missing_terminal", "duplicate_terminal"} for p in self.problems
        )

    def events_of(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.type == event_type]


def _to_text_lines(source: Iterable[Any]) -> Iterator[str]:
    """Yield decoded text lines from a text- or byte-iterable.

    Accepts an open file, a list of str/bytes, or any iterable of lines. Bytes
    are decoded as UTF-8 with replacement so a stray non-UTF-8 byte degrades one
    character instead of killing the stream. Handles the case of a single string
    blob (splitlines) as well as a pre-split iterable.
    """
    if isinstance(source, (str, bytes)):
        blob = source.decode("utf-8", "replace") if isinstance(source, bytes) else source
        yield from blob.splitlines()
        return
    for line in source:
        if isinstance(line, bytes):
            yield line.decode("utf-8", "replace")
        else:
            yield str(line)


def iter_events(source: Iterable[Any]) -> Iterator[Event]:
    """Stream `Event`s lazily, skipping blank/malformed lines silently.

    Use this for memory-bounded live consumption where you handle terminal /
    schema logic yourself. Use `parse_stream` for a fully-checked result.
    """
    import json

    for raw_line in _to_text_lines(source):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            yield Event.from_obj(obj)


def parse_stream(source: Iterable[Any]) -> ParseResult:
    """Fully parse a run stream into a checked `ParseResult`.

    - Decodes each line; a small number of malformed lines are recorded and
      skipped, but crossing `_MAX_MALFORMED` records a `decode_error` problem.
    - Reads `schema_version` from any record that carries it (typically the
      leading `run_start`); a MAJOR mismatch raises `UnsupportedSchemaError`
      rather than risk misreading reshaped fields.
    - Requires exactly one terminal (`result`) event: zero → `missing_terminal`,
      more than one → `duplicate_terminal`.
    """
    import json

    result = ParseResult()
    malformed = 0

    for line_no, raw_line in enumerate(_to_text_lines(source), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError) as exc:
            malformed += 1
            if malformed <= _MAX_MALFORMED:
                result.problems.append(ParseProblem("malformed_line", str(exc), line_no))
            elif malformed == _MAX_MALFORMED + 1:
                result.problems.append(ParseProblem(
                    "decode_error",
                    f"more than {_MAX_MALFORMED} malformed lines; stream is likely not JSONL",
                    line_no,
                ))
            continue
        if not isinstance(obj, dict):
            result.problems.append(ParseProblem("malformed_line", "line is not a JSON object", line_no))
            continue

        # Schema version can ride on any record; the first one we see wins and
        # gates the whole stream. A major we can't read is a hard stop.
        if result.schema is None and "schema_version" in obj:
            sv = SchemaVersion.parse(obj.get("schema_version"))
            if not sv.readable_by():
                raise UnsupportedSchemaError(sv.major)
            result.schema = sv

        event = Event.from_obj(obj)
        result.events.append(event)
        if event.is_terminal:
            if result.terminal is not None:
                result.problems.append(ParseProblem(
                    "duplicate_terminal", "more than one terminal 'result' event", line_no))
            else:
                result.terminal = event

    if result.terminal is None:
        result.problems.append(ParseProblem(
            "missing_terminal", "stream ended with no terminal 'result' event"))
    return result
