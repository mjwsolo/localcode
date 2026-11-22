"""The context budget is recomputed only when its inputs change; machine
observations run their commands together."""
import threading
import time

from localcode.ui import context_budget as cb
from localcode.ui import diagnostics
from localcode.ui import supervisor as sv


def test_budget_is_cached_until_new_samples_arrive(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "HERE", tmp_path)
    (tmp_path / "server.log").write_bytes(b"")
    sup = sv.Supervisor.__new__(sv.Supervisor)
    sup.port, sup.ctx, sup.ctx_total, sup.slots = 8123, 65536, 65536, 1
    sup.pp_prior = None
    sup._timing = cb.TimingTable(); sup._log_offset = 0; sup._budget = None; sup._budget_info = {}
    sup._budget_lock = threading.Lock()
    calls = []
    real = cb.budget
    monkeypatch.setattr(cb, "budget", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = sup.context_info()
    for _ in range(5):
        assert sup.context_info() == first
    assert len(calls) == 1
    with open(tmp_path / "server.log", "ab") as f:
        f.write(b"x task 1 | prompt eval time = 1000.00 ms / 2000 tokens (x)\n"
                b"x task 1 |        eval time = 1000.00 ms / 100 tokens (x)\n"
                b"x task 1 | stop processing: n_tokens = 3000, truncated = 0\n")
    sup.context_info()
    assert len(calls) == 2


def test_machine_observations_do_not_add_up_their_timeouts(monkeypatch):
    def slow(args, **kwargs):
        time.sleep(0.3)
        return ""
    monkeypatch.setattr(diagnostics.subprocess, "check_output", slow)
    start = time.monotonic()
    diagnostics.machine_observations()
    assert time.monotonic() - start < 0.9
