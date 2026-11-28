"""localcode bench: maths, safety checks and an end-to-end run against a fake server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localcode import bench


@pytest.fixture(autouse=True)
def _results_in_tmp(monkeypatch, tmp_path):
    """No test may write into the real ~/.localcode/bench: the history and the
    /bench comparison read it, so fake runs showed up as a model called "m"."""
    monkeypatch.setattr(bench, "results_dir", lambda: tmp_path)


def test_footprint_counts_mapped_model_pages():
    out = """llama-server [1]: 64-bit    Footprint: 9779 MB (16384 bytes per page)
  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
    0 B      21 GB          0 B         11    mapped file
9779 MB      21 GB        47 MB       5284    TOTAL
"""
    assert bench.footprint_resident_mb(out) == 9779 + 21 * 1024
    assert bench.footprint_resident_mb("x [1]: 64-bit    Footprint: 2 GB") == 2048
    assert bench.footprint_resident_mb("nothing") is None


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
    monkeypatch.setattr("localcode.ui.ports.find_running", lambda *a, **k: None)
    monkeypatch.setattr(bench, "serving_reason", lambda: None)
    monkeypatch.setattr(bench, "downloaded_models", lambda: files)
    monkeypatch.setattr(bench, "power_warning", lambda: None)
    monkeypatch.setattr(bench, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(bench, "COOLDOWN_S", 0)


def test_refuses_when_serving(monkeypatch, capsys):
    monkeypatch.setattr("localcode.ui.ports.find_running", lambda *a, **k: None)
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


def test_bench_running_uses_the_given_server_and_launches_nothing(monkeypatch, fake_server):
    port = fake_server.server_address[1]
    monkeypatch.setattr(bench, "thermal_snapshot", lambda: 100)
    launched = []
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k: launched.append(a))
    rep = bench.bench_running("m", f"http://127.0.0.1:{port}/v1", "", 16384, None, runs=2, sizes=(1024, 8192, 32768))
    assert launched == []
    assert rep.status == "ok" and set(rep.sizes) == {1024, 8192} and rep.skipped_sizes == [32768]
    assert rep.sizes[1024]["runs"] == 2 and rep.sizes[1024]["gen_tps"] == 40.0
    again = bench.ModelReport.from_json(json.loads(json.dumps(rep.to_json())))
    assert again.sizes[8192]["prefill_tps"] == 500.0 and "Gen tok/s" in bench.render_table([again], False)


def test_load_time_is_carried_through_json_table_and_history(monkeypatch, fake_server):
    port = fake_server.server_address[1]
    monkeypatch.setattr(bench, "thermal_snapshot", lambda: 100)
    rep = bench.bench_running("m", f"http://127.0.0.1:{port}/v1", "", 16384, None, runs=1, sizes=(1024, 8192),
                              load_s=27.4)
    again = bench.ModelReport.from_json(json.loads(json.dumps(rep.to_json())))
    assert again.load_s == 27.4 and again.first_reply_s is None
    lines = bench.render_table([again], False).splitlines()
    assert "Load s" in lines[0] and "1st reply s" in lines[0]
    assert "27.4" in lines[2] and "27.4" not in lines[3]       # once per model, on its first row
    data = bench.build_json([again], 1, [])
    row = bench.current_row(again, data)
    assert row["load_s"] == 27.4
    prev = dict(row, id="old", load_s=54.8)
    assert bench.compare_with_history(row, [prev])[0]["delta"]["load_pct"] == 100   # twice as fast to load


def test_supervisor_bench_endpoint_state(monkeypatch, fake_server):
    from localcode.ui import supervisor
    monkeypatch.setattr(bench, "thermal_snapshot", lambda: 100)
    sup = supervisor.Supervisor("x", fake_server.server_address[1], __import__("pathlib").Path("."), 16384)
    assert "error" in sup.bench_start()                       # nothing loaded
    sup.current, sup.state = "m", {"state": "ready"}
    sup.proc = type("P", (), {"pid": 0})()
    assert "error" in sup.bench_start(runs=9)                 # out of range
    assert "error" in sup.bench_start(sizes=(5,))
    assert sup.bench_start(runs=1, sizes=(1024,)) == {"ok": True}
    assert "error" in sup.bench_start()                       # already running
    import time
    for _ in range(100):
        if sup.bench_status()["state"] != "running":
            break
        time.sleep(0.05)
    st = sup.bench_status()
    assert st["report"]["load_s"] is None                      # this session did not start the server
    assert st["state"] == "done" and st["report"]["sizes"]["1024"]["gen_tps"] == 40.0


def test_cli_benchmarks_the_running_session_model(monkeypatch, tmp_path, capsys):
    _patch_env(monkeypatch, tmp_path, {"m": tmp_path / "m.gguf", "other": tmp_path / "o.gguf"})
    monkeypatch.setattr("localcode.ui.ports.find_running", lambda *a, **k: (8323, {"current": "m", "port": 1}))
    monkeypatch.setattr("localcode.ui.auth.read_auth_file", lambda d: ("tok", "key"))
    rep = bench.ModelReport("m", ctx=16384)
    rep.sizes[1024] = {"runs": 1, "prompt_tokens": 1000, "ttft_s": 1.0, "ttft_range": None, "prefill_tps": 500.0,
                       "prefill_range": None, "gen_tps": 40.0, "gen_range": None}
    seen = {}
    monkeypatch.setattr(bench, "bench_via_session", lambda ctrl, tok, runs, sizes, progress=None: seen.update(c=ctrl, t=tok) or rep)
    assert bench.main(["--yes"]) == 0
    assert seen == {"c": 8323, "t": "tok"} and "Gen tok/s" in capsys.readouterr().out
    # other models need the server free, and the message says how
    assert bench.main(["--models", "other", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "Close LocalCode or run /bench" in err
    assert bench.main(["--all-downloaded", "--yes"]) == 1


# ---------------------------------------------------------------- cancel, concurrency, history

class _Slow(_Fake):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        import time as _t
        for _ in range(100):
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n')
            self.wfile.flush()
            _t.sleep(0.05)


def _loaded_sup(port):
    from localcode.ui import supervisor
    sup = supervisor.Supervisor("x", port, __import__("pathlib").Path("."), 16384)
    sup.current, sup.state = "m", {"state": "ready"}
    sup.proc = type("P", (), {"pid": 0})()
    return sup


def test_bench_cancel_concurrency_and_status(monkeypatch, tmp_path):
    import time
    monkeypatch.setattr(bench, "thermal_snapshot", lambda: 100)
    monkeypatch.setattr("localcode.paths.GLOBAL_STATE_DIR", tmp_path, raising=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        sup = _loaded_sup(srv.server_address[1])
        assert "error" in sup.bench_cancel()                       # nothing running
        assert sup.bench_start(runs=1, sizes=(1024,)) == {"ok": True}
        st = sup.bench_status()
        assert st["state"] == "running" and st["total"] == 1 and "elapsed_s" in st
        second = sup.bench_start()
        assert "already running" in second["error"] and second["status"]["state"] == "running"
        assert sup.bench_cancel() == {"ok": True}
        assert sup.bench_status()["state"] == "cancelled"
        time.sleep(0.3)
        assert sup.bench_status()["state"] == "cancelled"          # the worker never overwrites it
        assert sup.bench_start(runs=1, sizes=(1024,)) == {"ok": True}   # a new run may start
        sup.bench_cancel()
    finally:
        srv.shutdown()


def _write_run(d, name, created, model, gen, version="0.5.0", notes=(), warnings=(), machine="Apple M3, 32 GB"):
    sizes = {"1024": {"ttft_s": 1.0, "prefill_tps": 500.0, "gen_tps": gen},
             "8192": {"ttft_s": 9.0, "prefill_tps": 450.0, "gen_tps": gen - 2}}
    data = {"schema": 1, "localcode_version": version, "created": created, "notes": list(notes), "machine": machine,
            "on_battery": any("battery" in n for n in notes),
            "models": [{"model": model, "status": "ok", "sizes": sizes, "peak_memory_mb": 20480.0, "warnings": list(warnings)}]}
    (d / name).write_text(json.dumps(data))


@pytest.fixture()
def results(monkeypatch, tmp_path):
    d = tmp_path / "bench"
    d.mkdir()
    monkeypatch.setattr(bench, "results_dir", lambda: d)
    return d


def test_history_is_newest_first_and_not_deduplicated(results):
    _write_run(results, "bench-20260101-100000.json", "2026-01-01T10:00:00", "Qwen3.8-27B-UD-Q4_K_XL", 30)
    _write_run(results, "bench-20260102-100000.json", "2026-01-02T10:00:00", "Qwen3.8-27B-UD-Q4_K_XL", 32)
    _write_run(results, "bench-20260103-100000.json", "2026-01-03T10:00:00", "gemma-4-12B-Q8_0", 50)
    (results / "bench-broken.json").write_text("{nope")
    rows = bench.load_history()
    assert [r["gen_1k"] for r in rows] == [50, 32, 30]
    assert rows[0]["display"] == "gemma-4-12B" and rows[0]["quant"] == "Q8_0"
    assert rows[1]["quant"] == "UD-Q4_K_XL" and rows[1]["machine"] == "Apple M3, 32 GB"
    assert rows[1]["peak_memory_mb"] == 20480.0
    assert "Gen 1K" in bench.render_history(rows) and "No saved" in bench.render_history([])


def test_compare_deltas_and_condition_marks(results):
    _write_run(results, "bench-20260101-100000.json", "2026-01-01T10:00:00", "A", 40, version="0.4.0")
    _write_run(results, "bench-20260102-100000.json", "2026-01-02T10:00:00", "B", 20, notes=["on battery power"])
    _write_run(results, "bench-20260103-100000.json", "2026-01-03T10:00:00", "B", 25, warnings=["generation slowed during the run"])
    _write_run(results, "bench-20260104-100000.json", "2026-01-04T10:00:00", "A", 44)
    hist = bench.load_history()
    cur = hist[0]
    rows = bench.compare_with_history(cur, hist)
    assert [(r["model"], r["relation"]) for r in rows] == [("A", "previous run"), ("B", "best run")]
    prev, other = rows
    assert prev["delta"]["gen_pct"] == 10 and prev["mismatch"] == ["LocalCode 0.4.0"]
    assert other["delta"]["gen_pct"] == 76 and other["mismatch"] == ["throttled"]
    battery = bench.compare_with_history(hist[0], [hist[0], hist[2]])
    assert battery[0]["mismatch"] == ["on battery"]


def test_delete_run(results):
    _write_run(results, "bench-20260101-100000.json", "2026-01-01T10:00:00", "A", 40)
    rid = bench.load_history()[0]["id"]
    assert not bench.delete_run("../evil.json#0") and not bench.delete_run("bench-x.json#9")
    assert bench.delete_run(rid) and bench.load_history() == []
    assert not (results / "bench-20260101-100000.json").exists()


def test_cli_history_and_saved_compare(results, capsys):
    _write_run(results, "bench-20260101-100000.json", "2026-01-01T10:00:00", "other-Q4_K_M", 33)
    assert bench.main(["--history"]) == 0
    assert "other-Q4_K_M" in capsys.readouterr().out
    rep = bench.ModelReport("now", sizes={1024: {"gen_tps": 40.0}})
    text = bench.render_saved_compare([rep])
    assert "other-Q4_K_M (2026-01-01" in text and "-18% gen" in text
