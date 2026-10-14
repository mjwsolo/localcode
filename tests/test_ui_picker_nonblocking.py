import threading
import time
from pathlib import Path
from types import SimpleNamespace

from localcode import hf_quants
from localcode.ui import supervisor


def test_quant_picker_returns_before_slow_network_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('LOCALCODE_HOME', str(tmp_path))
    monkeypatch.setattr(hf_quants, '_read_cache', lambda _: None)
    release, done = threading.Event(), threading.Event()
    def slow(_):
        release.wait(timeout=1)
        done.set()
        return []
    monkeypatch.setattr(supervisor, 'fetch_quants', slow)
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir = tmp_path; sup.current = None
    sup.ram_gb = 64; sup.bandwidth = 150
    sup.state = {'state': 'idle', 'pct': None}; sup.active = {}
    try:
        start = time.monotonic()
        result = sup.quants(supervisor.MODEL_GROUPS[0].key)
        assert time.monotonic() - start < 0.5, 'picker must not wait for Hugging Face'
        assert result['quants'], 'trusted fallback choices must remain available offline'
        assert result['listing'] is True
    finally:
        release.set()
        assert done.wait(timeout=2)


def test_failed_selection_reaps_only_the_failed_new_model(tmp_path):
    (tmp_path / 'custom.gguf').write_bytes(b'GGUF')
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir = tmp_path; sup.current = None; sup.proc = None
    sup.lock = threading.Lock(); sup.lock.acquire()
    sup.cancel = threading.Event(); sup.active = {}
    stopped = []
    def fail(_):
        sup.proc = object()
        return False
    def stop():
        stopped.append(sup.proc)
        sup.proc = None
    sup.start = fail; sup.stop = stop
    sup._select_worker(SimpleNamespace(drafter=None), 'custom.gguf', 'custom')
    assert len(stopped) == 1 and sup.proc is None
    assert sup.state['state'] == 'error' and not sup.lock.locked()


def test_validation_error_keeps_the_previous_model_process(tmp_path):
    (tmp_path / 'custom.gguf').write_bytes(b'GGUF')
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    previous = SimpleNamespace(poll=lambda: None)
    sup.models_dir = tmp_path; sup.current = 'previous'; sup.proc = previous
    sup.lock = threading.Lock(); sup.lock.acquire()
    sup.cancel = threading.Event(); sup.active = {}
    sup.progress = lambda: {'phase': 'idle'}
    def reject(_):
        raise ValueError('insufficient memory before starting a new process')
    sup.start = reject
    sup.stop = lambda: (_ for _ in ()).throw(AssertionError('previous model must survive'))
    sup._select_worker(SimpleNamespace(drafter=None), 'custom.gguf', 'custom')
    assert sup.proc is previous and not sup.lock.locked()
