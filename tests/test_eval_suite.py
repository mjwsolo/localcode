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


def test_benchmarks_refuse_missing_data_and_never_fetch(tmp_path, monkeypatch):
    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    for load in (lambda: benchmarks.polyglot_samples("python", None, None), lambda: benchmarks.livecodebench_samples("2000-01-01", None, None, None),
                 lambda: benchmarks.swebench_samples("pallets/flask", None, None), lambda: benchmarks.harnessbench_samples(None, None, None)):
        with pytest.raises(benchmarks.MissingData, match="Nothing is downloaded"):
            load()


def test_benchmark_samples_from_local_data(tmp_path, monkeypatch):
    import json

    from localcode_evals import benchmarks

    monkeypatch.setattr(benchmarks, "DATA", tmp_path)
    ex = tmp_path / "polyglot-benchmark/python/exercises/practice/two-fer"
    (ex / ".meta").mkdir(parents=True); (ex / ".docs").mkdir()
    (ex / ".meta/config.json").write_text(json.dumps({"files": {"solution": ["two_fer.py"], "test": ["two_fer_test.py"], "example": [".meta/example.py"]}}))
    (ex / ".meta/example.py").write_text("answer"); (ex / ".docs/instructions.md").write_text("Say it.")
    (ex / "two_fer.py").write_text("stub"); (ex / "two_fer_test.py").write_text("test")
    monkeypatch.setitem(benchmarks.POLYGLOT_TESTS["python"], "needs", [])
    [s] = benchmarks.polyglot_samples("python", None, None)
    assert s.id == "python/two-fer" and set(s.files) == {"two_fer.py", "two_fer_test.py"}      # the answer never reaches the workspace
    assert s.metadata["reference"]["files"] == {"two_fer.py": str(ex / ".meta/example.py")}

    (tmp_path / "livecodebench").mkdir()
    (tmp_path / "livecodebench/index.jsonl").write_text("".join(json.dumps({"id": i, "platform": "x", "date": d, "content": "c", "cases": [{"input": "1", "output": "1"}]}) + "\n"
                                                                 for i, d in (("old", "2023-01-01"), ("new", "2025-06-01"))))
    assert [s.id for s in benchmarks.livecodebench_samples("2025-01-01", None, None, None)] == ["new"]

    task_dir = tmp_path / "harness-bench/tasks/001-a"; (task_dir / "fixtures/in").mkdir(parents=True)
    (task_dir / "task.yaml").write_text(json.dumps({"task_id": "001-a", "class": "Office", "timeout_sec": 60}))
    (task_dir / "prompt.txt").write_text("Read $WORKSPACE/in/a.txt"); (task_dir / "fixtures/in/a.txt").write_text("a")
    tunnel = tmp_path / "harness-bench/tasks/002-b"; tunnel.mkdir()
    (tunnel / "task.yaml").write_text(json.dumps({"task_id": "002-b"})); (tunnel / "prompt.txt").write_text("x"); (tunnel / "hooks.py").write_text("# trycloudflare")
    data, left_out = benchmarks.harnessbench_samples(None, None, None)
    assert [s.id for s in data] == ["001-a"] and data[0].files == {"in/a.txt": str(task_dir / "fixtures/in/a.txt")}
    assert left_out["needs a public tunnel"] == 1
