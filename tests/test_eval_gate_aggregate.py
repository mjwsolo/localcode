"""The gate blocks a clear aggregate drop even with no task going all-pass to all-fail."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _rows(passes, tasks=10):
    return "".join(
        json.dumps(dict(task=f"t{i}", model="test", trial=t, purpose="regression", **{"pass": ok})) + "\n"
        for i in range(tasks) for t, ok in enumerate(passes, 1))


def _gate(tmp_path, base, cand):
    b, c = tmp_path / "b", tmp_path / "c"
    b.write_text(base)
    c.write_text(cand)
    return subprocess.run([sys.executable, str(ROOT / "evals/gate.py"), str(b), str(c)],
                          capture_output=True, text=True)


def test_half_the_trials_lost_on_every_task_blocks(tmp_path):
    assert _gate(tmp_path, _rows([True, True]), _rows([True, False])).returncode == 1


def test_prefill_collapse_blocks_even_when_decode_is_unchanged(tmp_path):
    def rows(prefill):
        return "".join(json.dumps(dict(task="t", model="test", trial=1, purpose="regression", **{"pass": True},
                                       decode_tps=50.0, prefill_tps=prefill)) + "\n")
    assert _gate(tmp_path, rows(500.0), rows(500.0)).returncode == 0
    assert _gate(tmp_path, rows(500.0), rows(100.0)).returncode == 1


def test_identical_runs_pass(tmp_path):
    assert _gate(tmp_path, _rows([True, True]), _rows([True, True])).returncode == 0
