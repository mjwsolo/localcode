"""localcode bench: maths, safety checks and an end-to-end run against a fake server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localcode import bench


class _Fake(BaseHTTPRequestHandler):
    prompts: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).prompts.append(body["messages"][0]["content"])
        assert body["cache_prompt"] is False and body["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for tok in ("a", "b", "c"):
            self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": tok}}]})}\n\n'.encode())
        final = {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 3},
                 "timings": {"prompt_n": 1000, "prompt_per_second": 500.0, "predicted_n": 3, "predicted_per_second": 40.0}}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture()
def fake_server():
    _Fake.prompts = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_median_and_spread():
    assert bench.median([3, 1, 2]) == 2
    assert bench.median([1, 2, 3, 4]) == 2.5
    assert bench.spread([3, 1, 2]) == (1, 3)
    assert bench.median([]) is None


def test_summarize_prefers_server_timings():
    r = bench.summarize_run(0.0, 2.0, 4.0, timings={"prompt_per_second": 500, "predicted_per_second": 40, "predicted_n": 10},
                            usage={"prompt_tokens": 1000}, chunks=10)
    assert (r.ttft_s, r.prefill_tps, r.gen_tps, r.prompt_tokens) == (2.0, 500.0, 40.0, 1000)


def test_summarize_wall_clock_fallback():
    r = bench.summarize_run(0.0, 2.0, 6.0, timings=None, usage={"prompt_tokens": 1000, "completion_tokens": 21}, chunks=21)
    assert r.prefill_tps == 500.0
    assert r.gen_tps == pytest.approx(20 / 4)


def test_throttle_flags():
    assert bench.throttle_flags([40, 38], 100, 100) == []
    assert "slowed" in bench.throttle_flags([40, 30], 100, 100)[0]
    assert "speed limit" in bench.throttle_flags([40, 40], 100, 80)[0]


def test_prompts_are_unique_and_deterministic():
    a, b = bench.synthetic_prompt(1024, "x"), bench.synthetic_prompt(1024, "y")
    assert a != b and a.split("\n", 1)[1] == b.split("\n", 1)[1]
    assert len(bench.synthetic_prompt(8192, "x")) > 6 * len(a)


def test_fits_skips_large_sizes():
    assert bench.fits(1024, 4096) and not bench.fits(8192, 8192) and bench.fits(32768, 65536)
    assert bench.ctx_from_command(["s", "--ctx-size", "40000", "--parallel", "2"]) == 20000


def test_stream_request_against_fake(fake_server):
    base = f"http://127.0.0.1:{fake_server.server_address[1]}/v1"
    r = bench.stream_request(base, "", "m", "hi")
    assert r.prefill_tps == 500.0 and r.gen_tps == 40.0 and r.completion_tokens == 3


def _patch_env(monkeypatch, tmp_path, files):
    monkeypatch.setattr(bench, "serving_reason", lambda: None)
    monkeypatch.setattr(bench, "downloaded_models", lambda: files)
    monkeypatch.setattr(bench, "power_warning", lambda: None)
    monkeypatch.setattr(bench, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(bench, "COOLDOWN_S", 0)


def test_refuses_when_serving(monkeypatch, capsys):
    monkeypatch.setattr(bench, "serving_reason", lambda: "a LocalCode session is serving a model")
    assert bench.main(["--yes"]) == 1
    assert "serving" in capsys.readouterr().err


def test_no_answer_launches_nothing(monkeypatch, tmp_path, capsys):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _patch_env(monkeypatch, tmp_path, {"m": gguf})
    launched = []
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: launched.append(a))
    assert bench.main(["--models", "m"], input_fn=lambda _: "n") == 0
    assert launched == [] and "Cancelled" in capsys.readouterr().err


def test_missing_model_is_reported_not_downloaded(monkeypatch, tmp_path, capsys):
    _patch_env(monkeypatch, tmp_path, {})
    assert bench.main(["--models", "ghost", "--yes"]) == 1
    assert "no downloaded model" in capsys.readouterr().err


def test_full_run_json_and_skip_large(monkeypatch, tmp_path, fake_server, capsys):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    small = tmp_path / "tiny.gguf"
    small.write_bytes(b"x")
    _patch_env(monkeypatch, tmp_path, {"m": gguf, "tiny": small})
    port = fake_server.server_address[1]
    monkeypatch.setattr(bench, "free_port", lambda: port)
    monkeypatch.setattr(bench, "thermal_snapshot", lambda: 100)

    def fake_cmd(path, p, alias=None):
        return ["llama-server", "--ctx-size", "2048" if "tiny" in path else "16384"]
    monkeypatch.setattr("localcode.ui.server_cmd.server_command", fake_cmd)

    class P:
        pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: P())
    monkeypatch.setattr(bench.os, "killpg", lambda *a: None)
    monkeypatch.setattr(bench.os, "getpgid", lambda p: p)
    monkeypatch.setattr(bench.MemorySampler, "sample", lambda self: 2048.0)

    out_path = tmp_path / "out.json"
    assert bench.main(["--models", "m,tiny,ghost", "--runs", "2", "--yes", "--compare", "--json", str(out_path)]) == 0
    text = capsys.readouterr().out
    assert "Gen tok/s" in text and "1.00x" in text
    raw = out_path.read_text()
    data = json.loads(raw)
    assert data["schema"] == 1 and data["runs_per_size"] == 2
    by = {m["model"]: m for m in data["models"]}
    assert set(by["m"]["sizes"]) == {"1024", "8192"} and by["m"]["skipped_sizes"] == [32768]
    assert by["m"]["sizes"]["1024"]["gen_tps"] == 40.0
    assert by["m"]["peak_memory_mb"] == 2048.0
    assert set(by["tiny"]["sizes"]) == {"1024"}
    assert by["ghost"]["status"] == "not downloaded"
    assert "/Users/" not in raw and str(tmp_path) not in raw
    assert list(tmp_path.glob("bench-*.json"))
    assert len({p.splitlines()[0] for p in _Fake.prompts}) == len(_Fake.prompts)


def test_server_command_value_error_is_skip(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")

    def boom(*a, **k):
        raise ValueError("Insufficient RAM")
    monkeypatch.setattr("localcode.ui.server_cmd.server_command", boom)
    r = bench.bench_model(gguf, 1)
    assert r.status == "skipped" and "RAM" in r.reason
