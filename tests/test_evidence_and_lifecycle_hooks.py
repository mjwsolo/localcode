from pathlib import Path

from localcode.evidence import EvidenceRegistry, EvidenceRequirement


def test_evidence_invalidates_when_file_or_environment_changes(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text("print('a')\n")
    registry = EvidenceRegistry()
    registry.require(EvidenceRequirement("tests", (source,), "pytest -q", ("PATH",)))
    env = {"PATH": "/bin"}
    registry.record("tests", environment=env, passed=True)
    assert registry.satisfied("tests", env)
    source.write_text("print('b')\n")
    assert not registry.satisfied("tests", env)
    source.write_text("print('a')\n")
    assert not registry.satisfied("tests", {"PATH": "/usr/bin"})
