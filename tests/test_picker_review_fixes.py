"""Picker: no star when nothing fits; a failed listing fetch is retried, not
remembered as fresh for a day."""
import time
from types import SimpleNamespace

from localcode import hf_quants
from localcode.ui import supervisor
from test_ui_picker_nonblocking import _cold_sup


def test_no_star_when_nothing_fits(tmp_path, monkeypatch):
    sup = _cold_sup(tmp_path, monkeypatch, lambda _: [])
    sup.ram_gb = 16
    rows = [SimpleNamespace(filename="x-Q8_0.gguf", size_gb=40.0), SimpleNamespace(filename="x-Q4_K_M.gguf", size_gb=30.0)]
    assert sup._recommended_quant_idx(rows, "x") is None


def test_failed_fetch_is_not_stamped_fresh(tmp_path, monkeypatch):
    calls = []
    def failing(_):
        calls.append(1)
        return []   # what fetch_quants returns on a network failure with no cache
    sup = _cold_sup(tmp_path, monkeypatch, failing)
    g = supervisor.MODEL_GROUPS[0]
    sup._quant_rows(g, refresh=True)
    deadline = time.time() + 5
    while sup.__dict__.get("_quant_pending") and time.time() < deadline:
        time.sleep(0.02)
    rows, stamp = sup._quant_memory[g.hf_repo]
    assert time.time() - stamp > 86400, "a failed listing must stay due for a retry"
