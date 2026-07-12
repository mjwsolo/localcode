"""Deterministic evidence requirements and observations.

Evidence is keyed by content, command, and environment hashes rather than by
human-readable claims.  A verification result therefore becomes stale as soon
as the file, command, or relevant environment changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    """Return a stable hash for a file, or a distinct marker if absent."""
    try:
        return _digest(path.read_bytes())
    except FileNotFoundError:
        return _digest(b"<missing>")


def command_hash(command: str | Sequence[str]) -> str:
    value = command if isinstance(command, str) else list(command)
    return _digest(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def environment_hash(environment: Mapping[str, str], keys: Sequence[str] | None = None) -> str:
    selected = keys if keys is not None else sorted(environment)
    payload = [(key, environment.get(key, "")) for key in sorted(set(selected))]
    return _digest(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


@dataclass(frozen=True)
class EvidenceKey:
    file_hashes: tuple[tuple[str, str], ...]
    command_hash: str
    environment_hash: str

    @classmethod
    def build(
        cls,
        *,
        files: Sequence[Path],
        command: str | Sequence[str],
        environment: Mapping[str, str],
        environment_keys: Sequence[str] | None = None,
    ) -> "EvidenceKey":
        hashes = tuple(sorted((str(path), file_hash(path)) for path in files))
        return cls(hashes, command_hash(command), environment_hash(environment, environment_keys))


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    files: tuple[Path, ...]
    command: str | tuple[str, ...]
    environment_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceRecord:
    requirement: str
    key: EvidenceKey
    passed: bool
    output: str = ""


@dataclass
class EvidenceRegistry:
    requirements: dict[str, EvidenceRequirement] = field(default_factory=dict)
    records: dict[str, EvidenceRecord] = field(default_factory=dict)

    def require(self, requirement: EvidenceRequirement) -> None:
        self.requirements[requirement.name] = requirement

    def key_for(self, name: str, environment: Mapping[str, str]) -> EvidenceKey:
        requirement = self.requirements[name]
        keys = requirement.environment_keys or None
        return EvidenceKey.build(
            files=requirement.files,
            command=requirement.command,
            environment=environment,
            environment_keys=keys,
        )

    def record(self, name: str, *, environment: Mapping[str, str], passed: bool, output: str = "") -> EvidenceRecord:
        record = EvidenceRecord(name, self.key_for(name, environment), passed, output[:4000])
        self.records[name] = record
        return record

    def satisfied(self, name: str, environment: Mapping[str, str]) -> bool:
        record = self.records.get(name)
        return bool(record and record.passed and record.key == self.key_for(name, environment))

    def missing(self, environment: Mapping[str, str]) -> list[str]:
        return [name for name in self.requirements if not self.satisfied(name, environment)]
