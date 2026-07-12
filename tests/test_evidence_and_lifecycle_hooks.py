from pathlib import Path

from localcode.evidence import EvidenceRegistry, EvidenceRequirement
from localcode.hooks import HookConfig, HookRunner


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


def test_lifecycle_hook_environment(monkeypatch, tmp_path: Path):
    runner = HookRunner(str(tmp_path), "session", "model")
    runner.config = HookConfig(post_edit="post", pre_completion="pre", post_compaction="compact")
    calls = []
    monkeypatch.setattr(runner, "_run", lambda command, env: calls.append((command, env)) or type("R", (), {"blocked": False})())
    runner.on_post_edit("a.py", "ok")
    runner.on_pre_completion("done", "completed")
    runner.on_post_compaction(12, 5)
    assert calls[0][1]["EDIT_PATH"] == "a.py"
    assert calls[1][1]["COMPLETION_STATUS"] == "completed"
    assert calls[2][1] == {"COMPACTION_BEFORE_COUNT": "12", "COMPACTION_AFTER_COUNT": "5"}
