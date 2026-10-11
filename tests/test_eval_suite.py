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


def test_harnessbench_refuses_missing_data_and_never_fetches(tmp_path, monkeypatch):
    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    with pytest.raises(benchmarks.MissingData, match="Nothing is downloaded"):
        benchmarks.harnessbench_samples(None, None, None)


def test_harnessbench_samples_from_local_data(tmp_path, monkeypatch):
    import json

    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    task_dir = tmp_path / "harness-bench/tasks/001-a"; (task_dir / "fixtures/in").mkdir(parents=True)
    (task_dir / "task.yaml").write_text(json.dumps({"task_id": "001-a", "class": "Office", "timeout_sec": 60}))
    (task_dir / "prompt.txt").write_text("Read $WORKSPACE/in/a.txt"); (task_dir / "fixtures/in/a.txt").write_text("a")
    tunnel = tmp_path / "harness-bench/tasks/002-b"; tunnel.mkdir()
    (tunnel / "task.yaml").write_text(json.dumps({"task_id": "002-b"})); (tunnel / "prompt.txt").write_text("x"); (tunnel / "hooks.py").write_text("# trycloudflare")
    data, left_out = benchmarks.harnessbench_samples(None, None, None)
    assert [s.id for s in data] == ["001-a"] and data[0].files == {"in/a.txt": str(task_dir / "fixtures/in/a.txt")}
    assert data[0].metadata["budget_seconds"] == 60 and left_out["needs a public tunnel"] == 1
    assert benchmarks.harnessbench_samples(None, "SRE", None)[0] == []
