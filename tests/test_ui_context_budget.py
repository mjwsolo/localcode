"""The context budget is measured on this machine, never a token literal."""
from __future__ import annotations

import json
import types

import pytest

from localcode.ui import context_budget as cb
from localcode.ui.launch import write_config

LOG = """0.01 I slot print_timing: id  0 | task 1 | prompt eval time =  1000.00 ms /  2000 tokens ( 0.5 ms per token, 2000.00 tokens per second)
0.01 I slot print_timing: id  0 | task 1 |        eval time =  1000.00 ms /   80 tokens (12.5 ms per token,   80.00 tokens per second)
0.01 I slot      release: id  0 | task 1 | stop processing: n_tokens = 4000, truncated = 0
"""


def sample(n_tokens, pp, tg, pp_n=1000, tg_n=100):
    return cb.Sample(n_tokens, pp, pp_n, tg, tg_n)


def test_parses_timing_lines_in_any_chunking():
    t = cb.TimingTable()
    for i in range(0, len(LOG), 7):          # feed in 7-byte pieces
        t.feed(LOG[i:i + 7])
    assert len(t.samples) == 1
    s = t.samples[0]
    assert s.n_tokens == 4000 and s.pp_n == 2000 and s.tg_n == 80
    assert s.pp_tps == pytest.approx(2000) and s.tg_tps == pytest.approx(80)


def test_release_without_timing_is_ignored():
    t = cb.TimingTable()
    t.feed("x | task 9 | stop processing: n_tokens = 100, truncated = 0\n")
    assert t.samples == []


def test_no_measurements_means_the_full_context_minus_headroom():
    b, info = cb.budget(65536, 4096, [])
    assert b == 65536 - 4096 and info["basis"] == "cap"


def test_prior_from_warmup_prefill_speed():
    # 300 tok/s cold prefill: 0.6 * 300 * 60 s = 10800 tokens, snapped down to the n_ctx/16 grid (2048) = 10240
    b, info = cb.budget(32768, 2048, [], prior_pp_tps=300)
    assert info["basis"] == "prior" and b == 10240
    b2, _ = cb.budget(131072, 8192, [], prior_pp_tps=1800)   # 64800 -> grid of 8192 -> 57344
    assert b2 == 57344


def test_reread_time_bounds_the_budget():
    n = 131072
    # prefill 2000 t/s at short prompts, 400 t/s from 40k on: 48k/400 = 120 s > 60 s; 40k/400 = 100 s; 32k/2000 ok
    samples = [sample(4000, 2000, 80)] * 5 + [sample(45000, 400, 60)] * 5 + [sample(70000, 300, 45)] * 5
    b, info = cb.budget(n, 8192, samples)
    assert info["basis"] == "measured" and info["limit"] == "reread"
    assert b == 32768


def test_decode_collapse_bounds_the_budget_even_when_prefill_is_fast():
    n = 65536
    samples = [sample(3000, 5000, 80)] * 5 + [sample(30000, 5000, 30)] * 5   # decode drops to 37% at 30k
    b, info = cb.budget(n, 4096, samples)
    assert info["limit"] == "decode" and b < 30000 and b >= n // 4


def test_fast_machine_keeps_the_cap():
    n = 65536
    samples = [sample(3000, 5000, 80)] * 5 + [sample(50000, 4000, 70)] * 5
    b, info = cb.budget(n, 4096, samples)
    assert b == n - 4096 and info["basis"] == "measured" and "limit" not in info


def test_budget_is_relative_to_the_loaded_context():
    small_samples = [sample(1000, 300, 12)] * 5 + [sample(6000, 100, 9)] * 5
    b16, _ = cb.budget(16384, 2048, small_samples)
    assert 16384 // 4 <= b16 <= 16384 - 2048
    b8, _ = cb.budget(8192, 1024, small_samples)
    assert 8192 // 4 <= b8 <= 8192 - 1024


def test_step_reserve_scales_with_context():
    assert cb.step_reserve(131072, 8192, 50 * 1024) == 8192 + 2 * (50 * 1024 // 4)
    assert cb.step_reserve(8192, 2048, 50 * 1024) == 4096          # capped at half the context


def test_launch_config_leaves_the_reserve_to_the_supervisor(tmp_path):
    p = tmp_path / "c.json"
    write_config(p, port=8123, ctx=40000, alias="m")
    cfg = json.loads(p.read_text())
    assert cfg["compaction"] == {"reserved": 0}
    assert cfg["provider"]["localcode"]["models"]["m"]["limit"]["context"] == 40000


def test_supervisor_status_reports_per_slot_context_and_budget(tmp_path, monkeypatch):
    from localcode.ui import supervisor as sv
    monkeypatch.setattr(sv, "HERE", tmp_path)
    (tmp_path / "server.log").write_bytes(b"")
    sup = sv.Supervisor.__new__(sv.Supervisor)
    sup.port, sup.ctx, sup.ctx_total, sup.slots = 8123, 131072, 131072, 1
    sup.pp_prior = None
    sup._timing = cb.TimingTable(); sup._log_offset = 0; sup._budget = None; sup._budget_info = {}
    import threading
    sup._budget_lock = threading.Lock()
    # /props says two slots of 65536 each
    class R:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(sv.urllib.request, "urlopen", lambda url, timeout=0: R({"default_generation_settings": {"n_ctx": 65536}, "total_slots": 2}))
    sup._probe_props()
    assert sup.ctx == 65536 and sup.slots == 2 and sup.ctx_total == 131072
    info = sup.context_info()
    assert info["ctx"] == 65536 and info["budget"] == 65536 - cb.step_reserve(65536, 8192, 50 * 1024)
    # the server logs a slow deep request: the budget comes down on the next poll
    with open(tmp_path / "server.log", "ab") as f:
        for i in range(5):
            f.write(f"x task {i} | prompt eval time = 1000.00 ms / 2000 tokens (x)\nx task {i} |        eval time = 1000.00 ms / 100 tokens (x)\nx task {i} | stop processing: n_tokens = 3000, truncated = 0\n".encode())
        for i in range(5, 10):
            f.write(f"x task {i} | prompt eval time = 10000.00 ms / 2000 tokens (x)\nx task {i} |        eval time = 1000.00 ms / 100 tokens (x)\nx task {i} | stop processing: n_tokens = 40000, truncated = 0\n".encode())
    info2 = sup.context_info()
    assert info2["budget"] < info["budget"] and info2["budget_info"]["limit"] == "reread"
