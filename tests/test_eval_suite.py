"""The Inspect AI suite must cover every model task on disk, without loading a model."""
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")
ROOT = Path(__file__).resolve().parents[1]


def test_every_task_directory_is_a_sample_of_exactly_one_task():
    import tomllib

    from localcode_evals.suite import regression, samples, tracking

    on_disk = {d.name: tomllib.loads((d / "task.toml").read_text()) for d in (ROOT / "evals/tasks").iterdir() if (d / "task.toml").is_file()}
    model_tasks = {name for name, meta in on_disk.items() if "kind=check" not in meta.get("notes", "")}
    reg = {s.id for s in regression().dataset}
    trk = {s.id for s in tracking().dataset}
    assert reg and trk and not (reg & trk)
    assert reg | trk == model_tasks
    one = samples(ids="fix-one-bug")[0]
    assert "app/calc.py" in one.files and one.metadata["purpose"] == "regression"
    with pytest.raises(ValueError):
        samples(ids="no-such-task")


def test_model_provider_names_a_gguf_and_refuses_to_generate():
    import asyncio

    from inspect_ai.model import get_model

    model = get_model("localcode/some-model")
    assert model.name == "some-model" and model.api.__class__.__name__ == "LocalCodeAPI"
    with pytest.raises(Exception):
        asyncio.run(model.generate("hi"))


def test_gate_reads_inspect_logs(tmp_path):
    import subprocess
    import sys

    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
    from inspect_ai.scorer import Score

    def log(path, passed):
        samples = [EvalSample(id="fix", epoch=1, input="x", target="t", metadata={"purpose": "regression"},
                              scores={"hidden_tests": Score(value="C" if passed else "I"),
                                      "measurements": Score(value={"decode_tps": 50.0, "steps": 3.0, "wall_s": 9.0})})]
        write_eval_log(EvalLog(status="success", eval=EvalSpec(created="2026-01-01T00:00:00+00:00", task="regression", dataset=EvalDataset(),
                                                                model="localcode/m", config=EvalConfig()), samples=samples), str(path))
    log(tmp_path / "b.eval", True); log(tmp_path / "ok.eval", True); log(tmp_path / "bad.eval", False)
    run = lambda c: subprocess.run([sys.executable, str(ROOT / "evals/gate.py"), str(tmp_path / "b.eval"), str(tmp_path / c)], capture_output=True, text=True)
    assert run("ok.eval").returncode == 0
    assert run("bad.eval").returncode == 1


def _harnessbench_task(root, tid, **extra):
    import yaml

    d = root / "tasks" / tid
    (d / "workspace/in").mkdir(parents=True); (d / "tests/reference").mkdir(parents=True)
    (d / "task.yaml").write_text(yaml.safe_dump({"task_id": tid, "domain": "Software Engineering", "difficulty": "hard", "timeout_sec": 7200, **extra}))
    (d / "instruction.md").write_text("Write out.txt"); (d / "workspace/in/a.txt").write_text("a")
    (d / "tests/cases.yaml").write_text("x"); (d / "tests/test_outcome.py").write_text("x")
    return d


def test_harnessbench_refuses_missing_or_old_data_and_never_fetches(tmp_path, monkeypatch):
    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    with pytest.raises(benchmarks.MissingData, match="Nothing is downloaded"):
        benchmarks.harnessbench_samples()
    (tmp_path / "harness-bench/tasks").mkdir(parents=True)          # a checkout from before 3.0
    with pytest.raises(benchmarks.MissingData, match="3.0"):
        benchmarks.harnessbench_samples()


def test_harnessbench_samples_from_local_data(tmp_path, monkeypatch):
    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    root = tmp_path / "harness-bench"
    (root / "src/harnessbench/grading").mkdir(parents=True); (root / "src/harnessbench/grading/outcome_cases.py").write_text("")
    plain = _harnessbench_task(root, "001-a")
    image = _harnessbench_task(root, "002-b"); (image / "environment").mkdir(); (image / "environment/Dockerfile").write_text("FROM x")
    hooked = _harnessbench_task(root, "003-c"); (hooked / "environment").mkdir(); (hooked / "environment/hooks.py").write_text("")
    data, left_out = benchmarks.harnessbench_samples()
    assert [s.id for s in data] == ["001-a"] and data[0].files == {"in/a.txt": str(plain / "workspace/in/a.txt")}
    assert left_out == {"needs a container image": 1, "needs setup hooks or a service": 1}
    # the local default budget is shorter than the benchmark's; the official one is recorded and can be asked for
    assert data[0].metadata["budget_seconds"] == benchmarks.LOCAL_BUDGET_SECONDS and data[0].metadata["official_timeout_sec"] == 7200
    assert benchmarks.harnessbench_samples(budget_seconds=7200)[0][0].metadata["budget_seconds"] == 7200
    assert benchmarks.harnessbench_samples(domains="Finance")[0] == [] and len(benchmarks.harnessbench_samples(domains="software")[0]) == 1
