"""Typed envelope + schema versioning for the `localcode run --json` stream.

This is the single public definition of a LocalCode automation event. Both the
native `eval/` harness and the Harbor plugin consume it through
`localcode.protocol` — neither may re-interpret the raw stream itself (see the
integration plan, §7.2.1). Keeping the envelope here means a schema change is a
review of one file with one compatibility policy.

Compatibility policy (semantic-version style, on the SCHEMA only):
  • MAJOR — a breaking change to the meaning/shape of existing fields. A reader
    built for major N MUST reject major N+1 rather than guess.
  • MINOR — purely additive: new event types or new fields on existing events.
    Older readers ignore what they don't recognise and keep working.
So a reader accepts a stream iff `stream.major == reader.major`, regardless of
minor. Unknown event types and unknown fields are always preserved verbatim on
the `Event` so a newer producer never breaks an older consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SCHEMA_MAJOR",
    "SCHEMA_MINOR",
    "SCHEMA_VERSION",
    "Event",
    "SchemaVersion",
    "UnsupportedSchemaError",
]

# The schema version this build of LocalCode emits and reads. Bump MINOR for
# additive changes (new event types/fields); bump MAJOR only for a breaking
# reshape of existing fields, and only with a documented deprecation.
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0
SCHEMA_VERSION = f"{SCHEMA_MAJOR}.{SCHEMA_MINOR}"


class UnsupportedSchemaError(Exception):
    """Raised when a stream declares a MAJOR schema this reader can't interpret.

    Explicit failure, never silent misinterpretation — a major bump means the
    meaning of existing fields changed, so guessing would produce wrong metrics.
    """

    def __init__(self, found_major: int, reader_major: int = SCHEMA_MAJOR) -> None:
        self.found_major = found_major
        self.reader_major = reader_major
        super().__init__(
            f"stream schema major {found_major} is not readable by this build "
            f"(supports major {reader_major}); upgrade localcode or the consumer"
        )


@dataclass(frozen=True)
class SchemaVersion:
    """A parsed `MAJOR.MINOR` schema version with a readable-by check."""

    major: int
    minor: int

    @classmethod
    def parse(cls, raw: Any) -> "SchemaVersion":
        """Parse `1`, `"1"`, or `"1.0"` into a SchemaVersion.

        A bare integer/`"1"` means major 1, minor 0. Anything unparseable is
        treated as the baseline (1.0) rather than crashing the stream — the
        stricter major-rejection happens in `readable_by`, driven by a value we
        could actually parse.
        """
        s = str(raw).strip()
        if not s:
            return cls(SCHEMA_MAJOR, 0)
        parts = s.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return cls(SCHEMA_MAJOR, 0)
        return cls(major, minor)

    def readable_by(self, reader_major: int = SCHEMA_MAJOR) -> bool:
        return self.major == reader_major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class Event:
    """One decoded stream record.

    `type` and `payload` are always present; `payload` holds every field the
    producer emitted for this record EXCEPT `type` (so unknown/future fields are
    preserved verbatim, never dropped). `raw` keeps the exact source dict for
    consumers that want lossless passthrough.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "Event":
        etype = str(obj.get("type") or "")
        payload = {k: v for k, v in obj.items() if k != "type"}
        return cls(type=etype, payload=payload, raw=dict(obj))

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    @property
    def is_terminal(self) -> bool:
        """The single mandatory closing record of every run."""
        return self.type == "result"
